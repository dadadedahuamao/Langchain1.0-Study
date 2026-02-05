"""
LangChain 1.0 - 消息类型与对话管理
====================================

本模块重点讲解：
1. 三种消息类型的实际使用
2. 对话历史管理（核心难点）
3. 消息的修剪和优化
"""

import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here_replace_this":
    raise ValueError("请先设置 GROQ_API_KEY")

model = init_chat_model("groq:llama-3.3-70b-versatile", api_key=GROQ_API_KEY)


# ============================================================================
# 示例 1：三种消息类型
# ============================================================================
def example_1_message_types():
    """
    三种消息类型：SystemMessage, HumanMessage, AIMessage

    重点：字典格式 vs 消息对象（推荐用字典）
    """
    print("\n" + "="*70)
    print("示例 1：三种消息类型对比")
    print("="*70)

    # 方式 1：消息对象（啰嗦）
    print("\n【方式 1：消息对象】")
    messages_obj = [
        SystemMessage(content="你是 Python 导师"),
        HumanMessage(content="什么是列表？")
    ]
    response = model.invoke(messages_obj)
    print(f"回复: {response.content[:100]}...")

    # 方式 2：字典格式（推荐，简洁）
    print("\n【方式 2：字典格式（推荐）】")
    messages_dict = [
        {"role": "system", "content": "你是 Python 导师"},
        {"role": "user", "content": "什么是列表？"}
    ]
    response = model.invoke(messages_dict)
    print(f"回复: {response.content[:100]}...")

    print("\n💡 推荐：直接用字典，更简洁！")


# ============================================================================
# 示例 2：对话历史管理（核心难点）
# ============================================================================
def example_2_conversation_history():
    """
    难点：如何正确管理对话历史

    关键：每次调用都要传递完整历史！
    """
    print("\n" + "="*70)
    print("示例 2：对话历史管理（重点）")
    print("="*70)

    # 初始化对话历史
    conversation = [
        {"role": "system", "content": "你是一个简洁的助手，回答限制在50字内"}
    ]

    # 第一轮
    print("\n【第 1 轮】")
    conversation.append({"role": "user", "content": "什么是 Python？"})
    print(f"用户: {conversation[-1]['content']}")

    r1 = model.invoke(conversation)
    print(f"AI: {r1.content}")

    # 关键：保存 AI 回复到历史
    conversation.append({"role": "assistant", "content": r1.content})

    # 第二轮（测试记忆）
    print("\n【第 2 轮】")
    conversation.append({"role": "user", "content": "它有什么特点？"})
    print(f"用户: {conversation[-1]['content']}")

    r2 = model.invoke(conversation)
    print(f"AI: {r2.content}")

    conversation.append({"role": "assistant", "content": r2.content})

    # 第三轮（测试上下文）
    print("\n【第 3 轮】")
    conversation.append({"role": "user", "content": "我第一个问题问的是什么？"})
    print(f"用户: {conversation[-1]['content']}")

    r3 = model.invoke(conversation)
    print(f"AI: {r3.content}")

    print(f"\n💡 对话历史共 {len(conversation)} 条消息")
    print("   AI 记住了之前的内容，因为每次都传递了完整历史！")


# ============================================================================
# 示例 3：错误示范 - AI 失忆
# ============================================================================
def example_3_wrong_way():
    """
    错误示范：不保存对话历史

    结果：AI 会"失忆"
    """
    print("\n" + "="*70)
    print("示例 3：错误示范 - AI 失忆")
    print("="*70)

    print("\n❌ 错误做法：不保存历史")

    # 第一次
    r1 = model.invoke("我叫张三")
    print(f"用户: 我叫张三")
    print(f"AI: {r1.content[:50]}...")

    # 第二次（没有传递历史）
    r2 = model.invoke("我叫什么名字？")
    print(f"\n用户: 我叫什么名字？")
    print(f"AI: {r2.content[:80]}...")
    print("\n❌ AI 不记得你叫张三！")


# ============================================================================
# 示例 4：对话历史的优化
# ============================================================================
def example_4_optimize_history():
    """
    难点：对话历史太长怎么办？

    解决方案：
    1. 只保留最近 N 条
    2. 总是保留 system 消息
    """
    print("\n" + "="*70)
    print("示例 4：优化对话历史（避免太长）")
    print("="*70)

    def keep_recent_messages(messages, max_pairs=3):
        """
        保留最近的 N 轮对话

        参数:
            messages: 完整消息列表
            max_pairs: 保留的对话轮数

        返回:
            优化后的消息列表
        """
        # 分离 system 消息和对话消息
        system_msgs = [m for m in messages if m.get("role") == "system"]
        conversation_msgs = [m for m in messages if m.get("role") != "system"]

        # 只保留最近的消息（每轮 = user + assistant）
        max_messages = max_pairs * 2
        recent_msgs = conversation_msgs[-max_messages:]

        # 返回：system + 最近对话
        return system_msgs + recent_msgs

    # 模拟长对话
    long_conversation = [
        {"role": "system", "content": "你是助手"},
        {"role": "user", "content": "我叫小明"},
        {"role": "assistant", "content": "收到···"},
        {"role": "user", "content": "我18岁"},
        {"role": "assistant", "content": "了解"},
        {"role": "user", "content": "我是男生"},
        {"role": "assistant", "content": "明白了"},
        {"role": "user", "content": "我是中国人"},
        {"role": "assistant", "content": "知道了"},
        {"role": "user", "content": "给出我的信息"},
    ]

    print(f"原始消息数: {len(long_conversation)}")
     # 使用未优化后的历史
    response1 = model.invoke(long_conversation)
    print(f"\nAI 回复: {response1.content[:100]}...")

    # 优化：只保留最近 2 轮
    optimized = keep_recent_messages(long_conversation, max_pairs=2)
    print(f"优化后消息数: {len(optimized)}")
    print(f"保留的内容: system + 最近2轮对话")

    # 使用优化后的历史
    response = model.invoke(optimized)
    print(f"\nAI 回复: {response.content[:100]}...")

    print("\n💡 技巧：对话太长时，只保留最近的几轮即可")


# ============================================================================
# 示例 5：实战 - 简单聊天机器人
# ============================================================================
def example_5_simple_chatbot():
    """
    实战：构建一个记住对话的聊天机器人

    对比有/无历史的差异，直观展示"遗忘"现象
    """
    print("\n" + "="*70)
    print("示例 5：实战 - 简单聊天机器人")
    print("="*70)

    # === 第一部分：保存历史（能记住） ===
    print("\n【有历史】AI 能记住对话：")
    conversation = [
        {"role": "system", "content": "你是一个友好的助手，回答尽量简洁"}
    ]

    questions = [
        "我叫李明，今年25岁",
        "我喜欢编程",
        "请问我叫什么名字？我今年多大？我喜欢什么？"
    ]

    for i, q in enumerate(questions, 1):
        print(f"\n--- 第 {i} 轮 ---")
        print(f"用户: {q}")

        conversation.append({"role": "user", "content": q})
        response = model.invoke(conversation)

        print(f"AI: {response.content}")
        conversation.append({"role": "assistant", "content": response.content})

    print(f"\n总共 {len(conversation)} 条消息，AI 完美记住了所有信息！")

    # === 第二部分：不保存历史（会遗忘） ===
    print("\n" + "-"*70)
    print("\n【无历史】每次单独调用，AI 会遗忘：")

    questions_no_history = [
        "我叫李明，今年25岁，我喜欢编程",
        "请问我叫什么名字？"  # 单独发送，没有上文
    ]

    for i, q in enumerate(questions_no_history, 1):
        print(f"\n--- 第 {i} 轮（独立调用） ---")
        print(f"用户: {q}")
        # 每次都是独立调用，不传历史
        response = model.invoke([
            {"role": "system", "content": "你是一个友好的助手，回答尽量简洁"},
            {"role": "user", "content": q}
        ])
        print(f"AI: {response.content}")

    print("\n对比结论：")
    print("  有历史：AI 记住了名字、年龄、爱好")
    print("  无历史：AI 不知道你是谁（因为没有传递之前的对话）")


# ============================================================================
# 主程序
# ============================================================================
def main():
    print("\n" + "="*70)
    print(" LangChain 1.0 - 消息类型与对话管理")
    print("="*70)

    try:
        example_1_message_types()
        input("\n按 Enter 继续...")

        example_2_conversation_history()
        input("\n按 Enter 继续...")

        example_3_wrong_way()
        input("\n按 Enter 继续...")

        example_4_optimize_history()
        input("\n按 Enter 继续...")

        example_5_simple_chatbot()

        print("\n" + "="*70)
        print(" 完成！")
        print("="*70)
        print("\n核心要点：")
        print("  ✅ 推荐用字典格式，不用消息对象")
        print("  ✅ 对话历史必须每次都传递完整的")
        print("  ✅ 记得保存 AI 的回复到历史中")
        print("  ✅ 历史太长时只保留最近几轮")

    except KeyboardInterrupt:
        print("\n\n程序中断")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
