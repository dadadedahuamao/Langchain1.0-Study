"""
化工厂危险行为预警系统 - 核心伪代码
====================================

这是主要算法的逻辑框架，可作为实际开发的参考

目录:
1. 主流程伪代码
2. 运动检测伪代码
3. AI分析伪代码
4. 预警决策伪代码
"""

# ============================================================================
# 1. 主流程伪代码
# ============================================================================
"""
FUNCTION main():
    """
    主流程：系统启动和运行
    """
    INIT system_config
    INIT camera_list
    INIT knowledge_base
    INIT ai_analyzer
    INIT alert_system

    FOREACH camera IN camera_list:
        START thread monitor_camera(camera)

    WHILE system_running:
        WAIT 1 second
        CHECK system_health
        LOG statistics

END FUNCTION


FUNCTION monitor_camera(camera):
    """
    监控单个摄像头
    """
    CONNECT camera.rtsp_url

    WHILE camera.online:
        frame = READ frame FROM camera

        # 步骤1: 运动检测
        IF detect_motion(frame):
            # 步骤2: 保存抽帧
            frame_path = SAVE frame TO disk

            # 步骤3: AI分析
            result = analyze_frame(frame_path)

            # 步骤4: 判断是否违规
            IF result.violation_detected:
                # 步骤5: 发送预警
                SEND alert TO alert_system
                SAVE evidence

    DISCONNECT camera
END FUNCTION
"""

# ============================================================================
# 2. 运动检测伪代码
# ============================================================================
"""
FUNCTION detect_motion(frame):
    """
    检测画面中的运动

    算法: 帧差法
    输入: 当前帧
    输出: 是否检测到运动 (True/False)
    """

    # 步骤1: 预处理
    gray_frame = CONVERT frame TO grayscale
    small_frame = RESIZE gray_frame BY 0.5
    blurred_frame = APPLY gaussian_blur TO small_frame

    # 步骤2: 如果是第一帧，保存并返回False
    IF prev_frame IS None:
        SET prev_frame = blurred_frame
        RETURN False

    # 步骤3: 计算帧差
    frame_diff = ABSOLUTE_DIFFERENCE(prev_frame, blurred_frame)

    # 步骤4: 二值化
    _, binary_diff = THRESHOLD frame_diff AT 25

    # 步骤5: 统计白色像素数量（运动区域）
    motion_score = COUNT white pixels IN binary_diff

    # 步骤6: 更新前一帧
    SET prev_frame = blurred_frame

    # 步骤7: 判断是否超过阈值
    IF motion_score > threshold:
        RETURN True, motion_score
    ELSE:
        RETURN False, motion_score

END FUNCTION


FUNCTION smooth_detection(motion_history):
    """
    平滑检测结果（避免抖动）

    算法: 滑动窗口投票
    输入: 最近N次检测结果
    输出: 平滑后的结果
    """

    # 计算True的比例
    true_count = COUNT True IN motion_history
    ratio = true_count / LENGTH(motion_history)

    # 超过50%才认为有运动
    IF ratio > 0.5:
        RETURN True
    ELSE:
        RETURN False

END FUNCTION
"""

# ============================================================================
# 3. AI分析伪代码
# ============================================================================
"""
FUNCTION analyze_frame(frame_path):
    """
    分析图片中的危险行为

    算法: QwenVL + RAG + Agent
    输入: 图片路径
    输出: 分析结果 {violation, description, rules, risk_level}
    """

    # 步骤1: 视觉分析（调用QwenVL）
    vision_result = CALL qwenvl_api(
        image = frame_path,
        question = "描述图片中的安全违规行为"
    )

    EXTRACT vision_info FROM vision_result:
        - person_count: 人员数量
        - equipment: 装备情况（安全帽、救生衣等）
        - activity: 活动描述
        - location: 位置

    # 步骤2: 检索安全规范（RAG）
    context = CONCATENATE(vision_info.location, vision_info.activity)

    safety_rules = QUERY knowledge_base WITH context:
        - 使用向量相似度搜索
        - 返回Top-3相关规范

    # 步骤3: Agent决策（对比实际vs规范）
    agent_result = CALL langchain_agent WITH:
        - vision_info: 从QwenVL提取的信息
        - safety_rules: 检索到的安全规范
        - system_prompt: "你是安全监控助手..."

    # 步骤4: 解析Agent结果
    PARSE agent_result:
        - violation_detected: 是否违规
        - violation_type: 违规类型
        - risk_level: 风险等级（高/中/低）
        - suggestions: 建议措施

    RETURN {
        violation: violation_detected,
        description: vision_info,
        rules: safety_rules,
        risk_level: risk_level,
        suggestions: suggestions
    }

END FUNCTION


FUNCTION query_knowledge_base(query):
    """
    查询安全规范知识库（RAG）

    算法: 向量相似度搜索
    输入: 查询文本
    输出: 相关安全规范列表
    """

    # 步骤1: 向量化查询
    query_vector = EMBED query USING model

    # 步骤2: 向量相似度搜索
    similar_docs = SEARCH knowledge_base:
        - vector = query_vector
        - top_k = 3
        - threshold = 0.7

    # 步骤3: 返回结果
    RETURN similar_docs

END FUNCTION
"""

# ============================================================================
# 4. 预警决策伪代码
# ============================================================================
"""
FUNCTION send_alert(result, camera_info):
    """
    发送预警通知

    算法: 根据风险等级选择通知渠道
    输入: 分析结果、摄像头信息
    """

    # 步骤1: 判断风险等级
    risk_level = result.risk_level

    # 步骤2: 生成预警报告
    alert_report = FORMAT """
    🚨 安全预警报告
    ================
    📍 位置: {camera_info.location}
    📷 摄像头: {camera_info.name}
    ⏰ 时间: {current_time}
    ⚠️ 违规行为: {result.violation_type}
    📋 违反规范: {result.rules}
    🎯 风险等级: {risk_level}
    💡 建议措施: {result.suggestions}
    📸 图片: {evidence_path}
    """

    # 步骤3: 根据风险等级选择通知渠道
    IF risk_level == "高":
        SEND alert_report VIA:
            - DingTalk (钉钉机器人)
            - SMS (短信通知)
            - Display (现场大屏)

    ELSE IF risk_level == "中":
        SEND alert_report VIA:
            - DingTalk (钉钉机器人)
            - Display (现场大屏)

    ELSE:  # 低风险
        LOG alert_report TO database  # 仅记录

    # 步骤4: 记录预警历史
    SAVE alert TO database:
        - timestamp
        - camera_id
        - risk_level
        - violation_type
        - evidence_path

END FUNCTION


FUNCTION check_duplicate_alert(new_alert):
    """
    检查重复预警（避免刷屏）

    算法: 时间窗口 + 相似度
    输入: 新预警
    输出: 是否发送 (True/False)
    """

    # 查询最近N分钟的预警
    recent_alerts = QUERY alerts FROM database:
        - camera_id = new_alert.camera_id
        - time_range = last 5 minutes

    # 检查是否有相同类型的违规
    FOREACH alert IN recent_alerts:
        IF alert.violation_type == new_alert.violation_type:
            # 发现相同违规，不再发送
            RETURN False

    # 没有重复，发送预警
    RETURN True

END FUNCTION
"""

# ============================================================================
# 5. 完整工作流示例
# ============================================================================
"""
完整工作流（单次检测循环）
========================

1. 读取摄像头帧
   ↓
2. 运动检测
   ├─ 有运动 → 继续
   └─ 无运动 → 跳过（回到步骤1）
   ↓
3. 保存抽帧
   ├─ 保存到 ./frames/camera1_timestamp.jpg
   └─ 记录元数据（时间、摄像头ID）
   ↓
4. 视觉分析（QwenVL）
   ├─ 输入: 图片 + 提示词
   ├─ 输出: "检测到2人，临水作业，未穿救生衣"
   └─ 提取: {person_count:2, location:"临水", equipment:[...]}
   ↓
5. 检索安全规范（RAG）
   ├─ 查询: "临水作业 救生衣"
   ├─ 向量搜索: Top-3
   └─ 结果: "临水作业必须穿救生衣（风险等级：高）"
   ↓
6. Agent决策
   ├─ 输入: 视觉信息 + 安全规范
   ├─ 推理: "检测到临水作业但未穿救生衣，违反规范"
   └─ 输出: {violation:True, risk_level:"高", suggestions:"立即停止作业"}
   ↓
7. 预警判断
   ├─ 检查重复（5分钟内是否有相同预警）
   ├─ 风险等级: 高
   └─ 决策: 发送预警
   ↓
8. 发送通知
   ├─ 钉钉机器人
   ├─ 短信通知
   └─ 现场大屏
   ↓
9. 记录证据
   ├─ 保存图片到归档
   └─ 写入数据库
   ↓
10. 回到步骤1（继续监控）
"""

# ============================================================================
# 6. 数据结构定义
# ============================================================================
"""
摄像头配置:
{
    "camera_id": "CAM001",
    "name": "临水作业区摄像头",
    "location": "化工厂东区 - 循环水池",
    "rtsp_url": "rtsp://admin:pass@192.168.1.100:554/stream1",
    "scene_type": "临水作业",
    "motion_threshold": 1000,
    "sensitivity": "medium"
}

视觉分析结果:
{
    "person_count": 2,
    "persons": [
        {
            "id": 1,
            "equipment": ["安全帽"],  # 缺少救生衣
            "activity": "施工"
        },
        {
            "id": 2,
            "equipment": [],  # 无任何装备
            "activity": "观察"
        }
    ],
    "location": "水池边",
    "activity": "施工作业"
}

安全规范:
{
    "rule_id": "RULE001",
    "scenario": "临水作业",
    "requirement": "救生衣",
    "risk_level": "高",
    "description": "在水池、河边等临水区域作业时，必须穿着救生衣"
}

预警报告:
{
    "alert_id": "ALT20250115143025",
    "timestamp": "2025-01-15 14:30:25",
    "camera_id": "CAM001",
    "location": "化工厂东区 - 循环水池",
    "violation_type": "临水作业未穿救生衣",
    "risk_level": "高",
    "evidence_path": "/data/evidence/ALT20250115143025.jpg",
    "suggestions": [
        "立即停止作业",
        "要求工人穿戴救生衣",
        "安排安全员现场监督"
    ],
    "status": "待处理"
}
"""

# ============================================================================
# 7. 性能优化技巧
# ============================================================================
"""
优化1: 异步处理
----------------
不使用: 同步调用（阻塞）
使用: 异步任务队列

伪代码:
    WHILE True:
        frame = READ camera
        ADD frame TO queue  # 非阻塞
        CONTINUE reading

    # 另一个线程处理队列
    WHILE queue NOT empty:
        frame = POP FROM queue
        PROCESS frame


优化2: 批量处理
--------------
不使用: 逐帧处理
使用: 累积多帧后批量分析

伪代码:
    frame_batch = COLLECT frames FOR 5 seconds
    BATCH_ANALYZE frame_batch  # 一次API调用多张图


优化3: 智能跳过
--------------
不使用: 每5秒固定抽帧
使用: 只在运动时段抽帧

伪代码:
    IF NOT in_motion_window:
        SKIP frame  # 静态时段不处理
    ELSE:
        PROCESS frame  # 运动时段才处理


优化4: 缓存复用
--------------
不使用: 重复查询知识库
使用: 缓存常见查询

伪代码:
    cache_key = HASH(query)
    IF cache_key IN cache:
        RETURN cache[cache_key]
    ELSE:
        result = QUERY knowledge_base
        cache[cache_key] = result
        RETURN result
"""

# ============================================================================
# 8. 错误处理
# ============================================================================
"""
错误处理策略
============

1. RTSP连接中断
   → 尝试重连（最多3次）
   → 记录日志
   → 发送告警通知

2. QwenVL API失败
   → 降级：只做运动检测，不做AI分析
   → 记录失败帧，稍后重试
   → 发送API异常告警

3. 知识库查询失败
   → 使用默认规范
   → 记录错误日志
   → 不影响主流程

4. 预警发送失败
   → 重试3次
   → 失败后入队重试
   → 记录失败预警

伪代码:
    TRY:
        result = PROCESS frame
    EXCEPT ConnectionError:
        RETRY up to 3 times
        IF still failed:
            LOG error
            SEND system alert
    EXCEPT APIError:
        USE fallback logic
        LOG error
    FINALLY:
            CLEANUP resources
"""

# ============================================================================
# 总结
# ============================================================================
"""
核心算法总结
============

1. 主流程: 监控 → 检测 → 分析 → 预警
2. 运动检测: 帧差法 + 平滑处理
3. AI分析: QwenVL视觉 + RAG检索 + Agent决策
4. 预警发送: 多级通知 + 去重机制

关键点:
✅ 只在运动时抽帧（节省70%成本）
✅ 向量检索相关规范（准确匹配）
✅ Agent自动决策（无需人工判断）
✅ 多渠道预警（及时响应）

下一步:
→ 参考 03_动态抽帧方案.py 实现代码
→ 参考 01_实现路径.md 了解详细步骤
"""
