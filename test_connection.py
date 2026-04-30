"""
LiteLLM 连接测试 - 使用 GLM 5.1
==============================================

测试通过 LiteLLM 代理（OpenAI 兼容）调用 GLM 5.1 模型
"""

import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

# ============================================================================
# 环境配置
# ============================================================================

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_API_BASE")

if not API_KEY:
    raise ValueError("请先设置 OPENAI_API_KEY")

print("="*70)
print(" LiteLLM 连接测试 - GLM 5.1")
print("="*70)
print(f"\n代理地址: {BASE_URL}")
print(f"API Key: {API_KEY[:10]}...{API_KEY[-4:]}")

# ============================================================================
# 测试1：使用 init_chat_model 调用
# ============================================================================
print("\n" + "-"*70)
print("测试1：init_chat_model 调用 GLM 5.1")
print("-"*70)

model = init_chat_model(
    "openai:glm-5.1",
    api_key=API_KEY,
    base_url=BASE_URL,
    temperature=0.7,
)

messages = [
    SystemMessage(content="你是一个友好的助手，回答简洁。"),
    HumanMessage(content="你好！用一句话介绍你自己。"),
]

response = model.invoke(messages)

print(f"\nAI 回复：{response.content}")
print(f"\n模型信息：")
print(f"  类型: {type(response).__name__}")
if hasattr(response, 'response_metadata') and response.response_metadata:
    for k, v in response.response_metadata.items():
        print(f"  {k}: {v}")

# ============================================================================
# 测试2：字典格式消息
# ============================================================================
print("\n" + "-"*70)
print("测试2：字典格式消息")
print("-"*70)

messages_dict = [
    {"role": "system", "content": "你是一个 Python 编程专家。"},
    {"role": "user", "content": "写一个快速排序的 Python 函数。"},
]

response2 = model.invoke(messages_dict)
print(f"\nAI 回复：\n{response2.content}")

# ============================================================================
# 测试3：流式输出
# ============================================================================
print("\n" + "-"*70)
print("测试3：流式输出")
print("-"*70)

print("\nAI 回复（流式）：", end="")
for chunk in model.stream("用三句话解释什么是 LangChain。"):
    print(chunk.content, end="", flush=True)
print()

# ============================================================================
# 完成
# ============================================================================
print("\n" + "="*70)
print(" 连接测试通过！")
print("="*70)
print("\n配置信息：")
print("  代理: LiteLLM (OpenAI 兼容)")
print("  模型: glm-5.1")
print("  init_chat_model 格式: openai:glm-5.1")
