"""
化工厂危险行为预警系统 - 动态抽帧实现
========================================

完整实现：运动检测 + 智能抽帧 + AI分析集成

功能：
1. 从海康威视摄像头读取RTSP流
2. 运动检测（只在画面变化时抽帧）
3. 集成QwenVL分析
4. 发送预警通知

运行：python 03_动态抽帧方案.py
"""

import cv2
import numpy as np
import time
import os
import requests
import base64
from datetime import datetime
from pathlib import Path
from collections import deque
import json

# ============================================================================
# 配置部分
# ============================================================================
class Config:
    """系统配置"""

    # 摄像头配置
    CAMERAS = [
        {
            "id": "CAM001",
            "name": "临水作业区摄像头",
            "location": "化工厂东区 - 循环水池",
            "rtsp_url": "rtsp://admin:123456@192.168.1.100:554/Streaming/Channels/101",
            "scene_type": "临水作业",
        },
        {
            "id": "CAM002",
            "name": "高空作业区摄像头",
            "location": "化工厂西区 - 脚手架",
            "rtsp_url": "rtsp://admin:123456@192.168.1.101:554/Streaming/Channels/101",
            "scene_type": "高空作业",
        }
    ]

    # 运动检测配置
    MOTION_THRESHOLD = 1000      # 运动检测阈值（越小越敏感）
    MIN_INTERVAL = 2.0           # 最小抽帧间隔（秒）
    MOTION_HISTORY_SIZE = 5      # 滑动窗口大小
    RESIZE_SCALE = 0.5           # 缩放比例（加速检测）

    # AI分析配置
    QWEN_API_KEY = os.getenv("DASHSCOPE_API_KEY")  # 阿里云API Key
    USE_AI = True                # 是否启用AI分析（测试时可关闭）

    # 输出配置
    OUTPUT_DIR = Path("./frames")
    EVIDENCE_DIR = Path("./evidence")

    # 预警配置
    ALERT_WEBHOOK = os.getenv("DINGTALK_WEBHOOK")  # 钉钉机器人Webhook


# ============================================================================
# 运动检测器
# ============================================================================
class MotionDetector:
    """
    运动检测器 - 基于帧差法

    特点：
    - 简单高效
    - 实时性好
    - 无需训练
    """

    def __init__(self, threshold=1000, history_size=5):
        """
        Args:
            threshold: 运动检测阈值
            history_size: 滑动窗口大小（用于平滑）
        """
        self.threshold = threshold
        self.history_size = history_size
        self.prev_frame = None
        self.motion_history = deque(maxlen=history_size)

    def detect(self, frame):
        """
        检测运动

        Args:
            frame: 当前帧（numpy数组）

        Returns:
            (是否检测到运动, 运动分数)
        """
        # 预处理
        small_frame = cv2.resize(frame, None, fx=Config.RESIZE_SCALE, fy=Config.RESIZE_SCALE)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # 第一帧
        if self.prev_frame is None:
            self.prev_frame = gray
            return False, 0

        # 计算帧差
        frame_diff = cv2.absdiff(self.prev_frame, gray)
        motion_score = np.sum(frame_diff > 25)

        # 更新前一帧
        self.prev_frame = gray

        # 判断是否超过阈值
        motion_detected = motion_score > self.threshold

        # 平滑处理（滑动窗口投票）
        self.motion_history.append(motion_detected)
        smoothed_motion = sum(self.motion_history) > len(self.motion_history) / 2

        return smoothed_motion, motion_score


# ============================================================================
# AI分析器（集成QwenVL）
# ============================================================================
class AIAnalyzer:
    """
    AI分析器 - 调用QwenVL视觉模型

    功能：
    - 识别人员数量
    - 检测防护装备
    - 分析危险行为
    """

    def __init__(self, api_key):
        """
        Args:
            api_key: 阿里云DashScope API Key
        """
        self.api_key = api_key
        self.api_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"

    def analyze(self, image_path, scene_type):
        """
        分析图片

        Args:
            image_path: 图片路径
            scene_type: 场景类型（如"临水作业"）

        Returns:
            分析结果字典
        """
        try:
            # 读取图片并转base64
            with open(image_path, 'rb') as f:
                image_base64 = base64.b64encode(f.read()).decode('ascii')

            # 构建请求
            prompt = f"""分析这张图片中的安全违规行为。

场景: {scene_type}

请按以下格式回答：
1. 人员数量: X人
2. 防护装备: 列举看到的装备（安全帽、救生衣、手套等）
3. 违规行为: 是否有违规，具体说明
4. 风险等级: 高/中/低

特别注意：
- 临水作业是否穿救生衣
- 高空作业是否戴安全帽
- 化学品操作是否戴防护手套"""

            payload = {
                "model": "qwen-vl-plus",
                "input": {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"image": f"data:image/jpeg;base64,{image_base64}"},
                                {"text": prompt}
                            ]
                        }
                    ]
                }
            }

            # 调用API
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            response = requests.post(self.api_url, json=payload, headers=headers, timeout=10)

            if response.status_code == 200:
                result = response.json()
                content = result["output"]["choices"][0]["message"]["content"][0]["text"]

                # 解析结果
                return self._parse_result(content)
            else:
                print(f"⚠️ API调用失败: {response.status_code}")
                return {"error": str(response.text)}

        except Exception as e:
            print(f"❌ AI分析失败: {e}")
            return {"error": str(e)}

    def _parse_result(self, content):
        """解析AI返回的结果"""
        # 简化解析（实际应该用正则或LLM再解析一次）
        return {
            "raw_response": content,
            "violation_detected": "违规" in content or "未" in content,
            "risk_level": "高" if "高" in content else ("中" if "中" in content else "低")
        }


# ============================================================================
# 预警系统
# ============================================================================
class AlertSystem:
    """
    预警系统 - 多渠道通知

    支持：
    - 钉钉机器人
    - 现场大屏（TODO）
    - 短信通知（TODO）
    """

    def __init__(self, webhook_url):
        """
        Args:
            webhook_url: 钉钉机器人Webhook URL
        """
        self.webhook_url = webhook_url
        self.recent_alerts = deque(maxlen=100)  # 用于去重

    def send_alert(self, alert_data):
        """
        发送预警

        Args:
            alert_data: 预警数据字典
        """
        # 检查重复（5分钟内相同摄像头+违规类型）
        if self._is_duplicate(alert_data):
            print("⏭️ 跳过重复预警")
            return

        # 生成预警报告
        report = self._format_report(alert_data)

        # 发送钉钉通知
        if self.webhook_url:
            self._send_dingtalk(report)

        # 记录预警历史
        self.recent_alerts.append({
            "camera_id": alert_data["camera_id"],
            "violation_type": alert_data["violation_type"],
            "timestamp": time.time()
        })

        print(f"🚨 预警已发送: {alert_data['violation_type']}")

    def _is_duplicate(self, alert_data):
        """检查是否重复预警"""
        current_time = time.time()

        for alert in self.recent_alerts:
            if (alert["camera_id"] == alert_data["camera_id"] and
                alert["violation_type"] == alert_data["violation_type"] and
                current_time - alert["timestamp"] < 300):  # 5分钟
                return True

        return False

    def _format_report(self, alert_data):
        """格式化预警报告"""
        return f"""
🚨 **安全预警报告**

📍 **位置**: {alert_data['location']}
📷 **摄像头**: {alert_data['camera_name']}
⏰ **时间**: {alert_data['timestamp']}
⚠️ **违规行为**: {alert_data['violation_type']}
🎯 **风险等级**: {alert_data['risk_level']}
📋 **场景**: {alert_data['scene_type']}

💡 **建议**:
{chr(10).join([f'- {s}' for s in alert_data['suggestions']])}

📸 **证据图片**: {alert_data['image_path']}
"""

    def _send_dingtalk(self, report):
        """发送钉钉通知"""
        payload = {
            "msgtype": "markdown",
            "markdown": {
                "title": "安全预警",
                "text": report
            }
        }

        try:
            response = requests.post(self.webhook_url, json=payload, timeout=5)
            if response.status_code == 200:
                print("✅ 钉钉通知发送成功")
            else:
                print(f"⚠️ 钉钉通知失败: {response.status_code}")
        except Exception as e:
            print(f"❌ 钉钉通知异常: {e}")


# ============================================================================
# 主监控器
# ============================================================================
class SafetyMonitor:
    """
    安全监控主类

    整合：运动检测 + AI分析 + 预警发送
    """

    def __init__(self, camera_config):
        """
        Args:
            camera_config: 摄像头配置字典
        """
        self.camera_id = camera_config["id"]
        self.camera_name = camera_config["name"]
        self.location = camera_config["location"]
        self.rtsp_url = camera_config["rtsp_url"]
        self.scene_type = camera_config["scene_type"]

        # 初始化组件
        self.motion_detector = MotionDetector(
            threshold=Config.MOTION_THRESHOLD,
            history_size=Config.MOTION_HISTORY_SIZE
        )

        if Config.USE_AI and Config.QWEN_API_KEY:
            self.ai_analyzer = AIAnalyzer(Config.QWEN_API_KEY)
        else:
            self.ai_analyzer = None
            print("⚠️ AI分析未启用")

        if Config.ALERT_WEBHOOK:
            self.alert_system = AlertSystem(Config.ALERT_WEBHOOK)
        else:
            self.alert_system = None
            print("⚠️ 预警系统未启用")

        # 状态
        self.cap = None
        self.last_save_time = 0
        self.frame_count = 0

    def start(self, max_frames=100):
        """开始监控"""
        print(f"\n{'='*70}")
        print(f"启动监控: {self.camera_name}")
        print(f"位置: {self.location}")
        print(f"场景: {self.scene_type}")
        print('='*70)

        # 连接RTSP流
        self.cap = cv2.VideoCapture(self.rtsp_url)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            print(f"❌ 无法连接: {self.rtsp_url}")
            return

        print("✅ RTSP连接成功")

        # 创建输出目录
        Config.OUTPUT_DIR.mkdir(exist_ok=True)
        Config.EVIDENCE_DIR.mkdir(exist_ok=True)

        try:
            while self.frame_count < max_frames:
                ret, frame = self.cap.read()

                if not ret:
                    print("⚠️ 读取帧失败")
                    break

                current_time = time.time()

                # 步骤1: 运动检测
                motion_detected, motion_score = self.motion_detector.detect(frame)

                # 步骤2: 检查是否可以抽帧
                can_save = (motion_detected and
                           (current_time - self.last_save_time >= Config.MIN_INTERVAL))

                if can_save:
                    # 步骤3: 保存抽帧
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{self.camera_id}_{timestamp}.jpg"
                    frame_path = Config.OUTPUT_DIR / filename
                    cv2.imwrite(str(frame_path), frame)

                    print(f"\n🎯 运动 #{self.frame_count + 1}: 分数={motion_score:.0f}")
                    print(f"   保存: {filename}")

                    # 步骤4: AI分析
                    if self.ai_analyzer:
                        print("   🔍 AI分析中...")
                        analysis_result = self.ai_analyzer.analyze(
                            str(frame_path),
                            self.scene_type
                        )

                        if "error" not in analysis_result:
                            print(f"   结果: {analysis_result['raw_response'][:100]}...")

                            # 步骤5: 判断是否违规
                            if analysis_result.get("violation_detected"):
                                # 保存证据
                                evidence_path = Config.EVIDENCE_DIR / f"evidence_{timestamp}.jpg"
                                cv2.imwrite(str(evidence_path), frame)

                                # 步骤6: 发送预警
                                alert_data = {
                                    "camera_id": self.camera_id,
                                    "camera_name": self.camera_name,
                                    "location": self.location,
                                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "violation_type": "检测到违规行为",
                                    "risk_level": analysis_result.get("risk_level", "中"),
                                    "scene_type": self.scene_type,
                                    "image_path": str(evidence_path),
                                    "suggestions": [
                                        "立即停止作业",
                                        "检查防护装备",
                                        "安排安全员现场监督"
                                    ]
                                }

                                if self.alert_system:
                                    self.alert_system.send_alert(alert_data)

                    self.last_save_time = current_time
                    self.frame_count += 1

                # 显示预览
                self._show_preview(frame, motion_detected, motion_score)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n用户中断")
                    break

        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            print(f"\n✅ 监控结束: 共保存 {self.frame_count} 帧")

    def _show_preview(self, frame, motion_detected, motion_score):
        """显示预览窗口"""
        display_frame = frame.copy()

        status = "⚠️ 运动!" if motion_detected else "静态"
        color = (0, 0, 255) if motion_detected else (0, 255, 0)

        cv2.putText(display_frame, f"{self.camera_name}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.putText(display_frame, f"Status: {status}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.putText(display_frame, f"Score: {motion_score:.0f}", (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.putText(display_frame, f"Saved: {self.frame_count}", (10, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow(f'Safety Monitor - {self.camera_id}', display_frame)


# ============================================================================
# 主程序
# ============================================================================
def main():
    """主程序"""
    print("""
╔════════════════════════════════════════════════════════════╗
║  化工厂危险行为预警系统 - 动态抽帧                          ║
║  Version: 1.0                                              ║
╚════════════════════════════════════════════════════════════╝
    """)

    # 环境检查
    if Config.USE_AI and not Config.QWEN_API_KEY:
        print("⚠️ 未设置 DASHSCOPE_API_KEY，AI分析已禁用")
        Config.USE_AI = False

    if not Config.ALERT_WEBHOOK:
        print("⚠️ 未设置 DINGTALK_WEBHOOK，预警通知已禁用")

    # 选择摄像头
    print("\n可用摄像头:")
    for i, cam in enumerate(Config.CAMERAS, 1):
        print(f"  {i}. {cam['name']} ({cam['location']})")

    try:
        choice = int(input("\n请选择摄像头编号: ")) - 1
        if 0 <= choice < len(Config.CAMERAS):
            camera_config = Config.CAMERAS[choice]

            # 创建监控器
            monitor = SafetyMonitor(camera_config)

            # 开始监控
            monitor.start(max_frames=100)

        else:
            print("❌ 无效选择")

    except ValueError:
        print("❌ 请输入数字")
    except KeyboardInterrupt:
        print("\n\n用户中断")


if __name__ == "__main__":
    # 示例：直接监控第一个摄像头
    if len(Config.CAMERAS) > 0:
        monitor = SafetyMonitor(Config.CAMERAS[0])
        monitor.start(max_frames=100)
    else:
        print("❌ 未配置摄像头，请修改 Config.CAMERAS")
