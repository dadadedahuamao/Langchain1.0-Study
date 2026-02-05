# 06 - Agent Loop (Agent 执行循环)

## 核心概念

**Agent 执行循环 = 自动化的"思考-行动-观察"过程**

Agent 不是一次性调用，而是一个循环：
```
用户问题 → AI 思考 → 调用工具 → 观察结果 → 继续思考 → 最终答案
```

## 执行循环详解

### 完整流程

```
┌─────────────┐
│ 用户提问    │
│ HumanMessage│
└──────┬──────┘
       ↓
┌─────────────┐
│ AI 分析问题 │
│ 需要工具？  │
└──────┬──────┘
       ↓ 是
┌─────────────┐
│ AI 决定调用 │
│ AIMessage   │
│ (tool_calls)│
└──────┬──────┘
       ↓
┌─────────────┐
│ 执行工具    │
│ ToolMessage │
└──────┬──────┘
       ↓
┌─────────────┐
│ AI 看结果   │
│ 生成答案    │
│ AIMessage   │
└─────────────┘
```

### 消息历史示例

```python
response = agent.invoke({
    "messages": [{"role": "user", "content": "25 乘以 8"}]
})

# response['messages'] 包含：
[
    HumanMessage(content="25 乘以 8"),
    AIMessage(tool_calls=[{
        'name': 'calculator',
        'args': {'operation': 'multiply', 'a': 25, 'b': 8}
    }]),
    ToolMessage(content="25.0 multiply 8.0 = 200.0"),
    AIMessage(content="25 乘以 8 等于 200")
]
```

## 查看执行过程

### 1. 查看完整历史

```python
response = agent.invoke({"messages": [...]})

for msg in response['messages']:
    print(f"{msg.__class__.__name__}: {msg.content}")
```

### 2. 获取最终答案

```python
# 最后一条消息就是最终答案
final_answer = response['messages'][-1].content
```

### 3. 查看使用的工具

```python
used_tools = []
for msg in response['messages']:
    if hasattr(msg, 'tool_calls') and msg.tool_calls:
        for tc in msg.tool_calls:
            used_tools.append(tc['name'])

print(f"使用的工具: {used_tools}")
```

## 流式输出（Streaming）

**用于实时显示 Agent 的进度**

### 两种流式模式对比

| 模式 | 参数 | 返回格式 | 用途 |
|------|------|---------|------|
| **按步骤输出** | `stream_mode="updates"` | `dict` | 显示每步进度（工具调用、结果） |
| **逐 token 输出** | `stream_mode="messages"` | `(chunk, metadata)` 元组 | 打字机效果，实时显示文字 |

---

### 方式 1：按步骤输出（`stream_mode="updates"`）

**特点：** 每个节点（model、tools）执行完后返回一次，适合显示执行进度。

```python
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "北京天气如何？"}]},
    stream_mode="updates"  # 默认模式
):
    for node_name, node_output in chunk.items():
        if 'messages' in node_output:
            latest_msg = node_output['messages'][-1]
            msg_type = latest_msg.__class__.__name__

            if msg_type == "AIMessage":
                if hasattr(latest_msg, 'tool_calls') and latest_msg.tool_calls:
                    print(f"[{node_name}] 调用工具: {latest_msg.tool_calls[0]['name']}")
                elif latest_msg.content:
                    print(f"[{node_name}] 最终回答: {latest_msg.content}")
            elif msg_type == "ToolMessage":
                print(f"[{node_name}] 工具返回: {latest_msg.content}")
```

**输出示例：**
```
[model] 调用工具: get_weather
[tools] 工具返回: 晴天，温度 15°C...
[model] 最终回答: 北京今天天气晴朗，温度15°C
```

---

### 方式 2：逐 token 输出（`stream_mode="messages"`）

**特点：** 逐字符/token 输出，实现打字机效果。返回 `(chunk, metadata)` 元组。

```python
print("回答: ", end="", flush=True)

for chunk, metadata in agent.stream(
    {"messages": [{"role": "user", "content": "用一句话介绍 Python"}]},
    stream_mode="messages"  # 逐 token 模式
):
    # 只显示 model 节点的输出
    if metadata.get("langgraph_node") == "model":
        if hasattr(chunk, 'content') and chunk.content:
            print(chunk.content, end="", flush=True)

print()  # 换行
```

**输出效果（逐字显示）：**
```
回答: Python是一种简洁易读的编程语言...
```

---

### 两种模式的选择

| 场景 | 推荐模式 |
|------|---------|
| 显示 Agent 执行步骤 | `stream_mode="updates"` |
| 聊天界面打字机效果 | `stream_mode="messages"` |
| 调试/监控工具调用 | `stream_mode="updates"` |
| 用户体验优化 | `stream_mode="messages"` |

---

### stream vs invoke

| 方法 | 返回 | 用途 |
|-----|------|------|
| `invoke()` | 完整结果 | 等待完成后一次性获取 |
| `stream()` | 生成器 | 实时获取中间步骤/token |

## 消息类型

### HumanMessage
用户的输入

```python
HumanMessage(content="北京天气如何？")
```

### AIMessage（两种情况）

**情况1：调用工具**
```python
AIMessage(
    content="",
    tool_calls=[{
        'name': 'get_weather',
        'args': {'city': '北京'},
        'id': 'call_xxx'
    }]
)
```

**情况2：最终答案**
```python
AIMessage(content="北京今天晴天，温度 15°C")
```

### ToolMessage
工具执行的结果

```python
ToolMessage(
    content="晴天，温度 15°C",
    name="get_weather"
)
```

### SystemMessage
系统指令（通过 `system_prompt` 设置）

```python
agent = create_agent(
    model=model,
    tools=tools,
    system_prompt="你是一个helpful assistant"
)
```

## 多步骤执行

Agent 可以多次调用工具：

```python
# 问题：先算 10 + 20，然后乘以 3
response = agent.invoke({
    "messages": [{"role": "user", "content": "先算 10 + 20，然后乘以 3"}]
})

# Agent 可能会：
# 1. 调用 calculator(add, 10, 20) → 30
# 2. 调用 calculator(multiply, 30, 3) → 90
# 3. 返回最终答案
```

统计工具调用次数：
```python
tool_calls_count = sum(
    len(msg.tool_calls) if hasattr(msg, 'tool_calls') and msg.tool_calls else 0
    for msg in response['messages']
)
```

## 调试技巧

### 1. 打印所有消息

```python
for i, msg in enumerate(response['messages'], 1):
    print(f"\n--- 消息 {i}: {msg.__class__.__name__} ---")

    if hasattr(msg, 'content'):
        print(f"内容: {msg.content}")

    if hasattr(msg, 'tool_calls') and msg.tool_calls:
        for tc in msg.tool_calls:
            print(f"工具: {tc['name']}, 参数: {tc['args']}")
```

### 2. 使用 stream 查看步骤

```python
step = 0
for chunk in agent.stream(input):
    step += 1
    print(f"步骤 {step}:")
    if 'messages' in chunk:
        latest = chunk['messages'][-1]
        print(f"  类型: {latest.__class__.__name__}")
```

### 3. 检查是否使用工具

```python
has_tool_calls = any(
    hasattr(msg, 'tool_calls') and msg.tool_calls
    for msg in response['messages']
)

if has_tool_calls:
    print("Agent 使用了工具")
else:
    print("Agent 直接回答")
```

## 常见问题

### 1. 如何知道 Agent 何时完成？

**答：当 AIMessage 不包含 tool_calls 时**

```python
for msg in response['messages']:
    if isinstance(msg, AIMessage):
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            print("还在调用工具...")
        else:
            print("完成！最终答案：", msg.content)
```

### 2. Agent 可以调用多少次工具？

**答：默认没有限制，直到得到最终答案**

但可能会：
- 超时
- 达到 token 限制
- 模型决定停止

### 3. 如何限制工具调用次数？

LangChain 1.0 的 `create_agent` 默认使用 LangGraph，可以通过配置限制：

```python
# 注意：这是高级用法，后续会详细学习
config = {
    "recursion_limit": 5  # 最多 5 步
}

response = agent.invoke(input, config=config)
```

## 最佳实践

### 1. 生产环境获取答案

```python
try:
    response = agent.invoke(input)
    final_answer = response['messages'][-1].content
    return final_answer
except Exception as e:
    logger.error(f"Agent 错误: {e}")
    return "抱歉，出现错误"
```

### 2. 用户体验优化

```python
# 使用流式输出
print("正在思考...")
for chunk in agent.stream(input):
    if 'messages' in chunk:
        latest = chunk['messages'][-1]
        # 显示进度
```

### 3. 调试和监控

```python
response = agent.invoke(input)

# 记录使用的工具
tools_used = [
    tc['name']
    for msg in response['messages']
    if hasattr(msg, 'tool_calls') and msg.tool_calls
    for tc in msg.tool_calls
]

logger.info(f"工具使用: {tools_used}")
```

### 4. 错误处理

```python
try:
    response = agent.invoke(input)

    # 检查是否成功
    if not response['messages']:
        raise ValueError("没有收到响应")

    final = response['messages'][-1]
    if not hasattr(final, 'content') or not final.content:
        raise ValueError("没有最终答案")

    return final.content

except Exception as e:
    # 记录详细错误
    logger.error(f"Agent 执行失败: {e}", exc_info=True)
    return None
```

## 运行示例

```bash
# 运行所有示例
python main.py

# 测试
python test.py
```

## 核心要点总结

1. **执行循环**：问题 → 工具调用 → 结果 → 答案
2. **messages 历史**：记录完整对话过程
3. **流式输出**：`stream()` 实时显示进度
4. **消息类型**：HumanMessage、AIMessage、ToolMessage
5. **最终答案**：`response['messages'][-1].content`

## 下一步

**阶段一（基础）完成！**

已学习：
- 01: 环境搭建和模型调用
- 02: 提示词模板
- 03: 消息类型和对话
- 04: 自定义工具
- 05: Simple Agent
- 06: Agent 执行循环

**下一阶段：phase2_intermediate**
- 内存和状态管理
- 中间件架构
- 结构化输出
