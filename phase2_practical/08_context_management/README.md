# 08 - Context Management (上下文管理)

## 核心概念

**问题**：对话历史会无限增长 → 超 token、成本高、响应慢

**解决**：使用中间件自动管理上下文长度

## SummarizationMiddleware（推荐）

### 基本用法

```python
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model=model,
    tools=[],
    checkpointer=InMemorySaver(),
    middleware=[
        SummarizationMiddleware(
            model="groq:llama-3.3-70b-versatile",
            trigger=("tokens", 500)  # 超过 500 tokens 触发摘要
        )
    ]
)
```

### 工作原理

```
对话历史: [消息1, 消息2, ..., 消息20]  (超过 500 tokens)
    ↓
SummarizationMiddleware 自动触发
    ↓
摘要旧消息: "用户是张三，在北京工作，喜欢编程..."
    ↓
新历史: [摘要, 最近消息]  (减少到 300 tokens)
```

### 参数说明

| 参数 | 说明 | 示例 |
|-----|------|------|
| `model` | 生成摘要的模型（可用便宜模型） | 必需 |
| `trigger` | 触发摘要的条件 | `("tokens", 500)` 或 `("messages", 10)` |
| `keep` | 保留多少最近消息 | `("messages", 3)` |

> **注意**: `max_tokens_before_summary` 和 `messages_to_keep` 参数已废弃，请使用 `trigger` 和 `keep`。

## trim_messages（手动修剪）

### 基本用法

```python
from langchain_core.messages import trim_messages

# 只保留最近 N 条消息
trimmed = trim_messages(
    messages,
    max_tokens=100,
    strategy="last",  # 保留最后的
    token_counter=len
)
```

### 适用场景

- 只需要最近几轮对话
- 不需要保留旧信息
- 简单直接

## 策略对比

| 策略 | 优点 | 缺点 | 适用 |
|-----|------|------|------|
| **不处理** | 完整历史 | 超 token | 短对话 |
| **SummarizationMiddleware** | 自动化、保留信息 | 摘要成本 | 长对话（推荐）|
| **trim_messages** | 简单、精确 | 丢失旧信息 | 只要最近 N 轮 |

## 实际应用

### 客服机器人

```python
agent = create_agent(
    model=model,
    tools=[查询订单, 查询物流],
    system_prompt="客服助手",
    checkpointer=InMemorySaver(),
    middleware=[
        SummarizationMiddleware(
            model="groq:llama-3.3-70b-versatile",
            trigger=("tokens", 800),  # 超过 800 tokens 触发摘要
            keep=("messages", 5)       # 保留最近 5 条消息
        )
    ]
)
```

### 长期对话助手

```python
agent = create_agent(
    model=model,
    tools=[],
    middleware=[
        SummarizationMiddleware(
            model="groq:llama-3.3-70b-versatile",
            trigger=("tokens", 1000),  # 超过 1000 tokens 触发摘要
            keep=("messages", 3)       # 保留最近 3 条消息
        )
    ],
    checkpointer=InMemorySaver()
)
```

## 常见问题

### 1. 摘要会丢失信息吗？

会有一些细节丢失，但：
- 重要信息会保留（姓名、关键事实）
- 最近的消息完整保留
- 对于大部分场景足够

### 2. 如何选择 trigger 和 keep 参数？

```python
# trigger（触发条件）：
# - 模型上下文窗口 4k → trigger=("tokens", 3000)
# - 模型上下文窗口 8k → trigger=("tokens", 6000)
# - 模型上下文窗口 16k → trigger=("tokens", 12000)
# 留一些余量给工具调用和系统提示

# keep（保留消息）：
# - 短对话：keep=("messages", 3)  # 保留最近 3 条
# - 长对话：keep=("messages", 5)  # 保留最近 5 条
```

### 3. 摘要成本高吗？

- 摘要只在超过阈值时触发
- 可以使用便宜的模型（如 gpt-3.5）
- 相比传输全部历史，通常更便宜

### 4. 能自定义摘要提示词吗？

可以！使用 `summary_prompt` 参数：

```python
SummarizationMiddleware(
    model="groq:llama-3.3-70b-versatile",
    trigger=("tokens", 800),
    summary_prompt="请总结对话，重点保留：用户姓名、关键事实、待办事项"
)
```

如需更复杂的控制，可以实现自己的中间件（Module 10 会学）。

## 最佳实践

```python
# 1. 根据场景选择阈值
agent = create_agent(
    model=model,
    tools=[],
    middleware=[
        SummarizationMiddleware(
            model="groq:llama-3.3-70b-versatile",
            trigger=("tokens", 500),  # 短对话
            # trigger=("tokens", 2000),  # 长对话
            keep=("messages", 3)       # 保留最近 3 条
        )
    ],
    checkpointer=InMemorySaver()
)

# 2. 使用便宜模型摘要（降低成本）
SummarizationMiddleware(
    model="groq:llama-3.3-70b-versatile",  # 用便宜模型摘要
    trigger=("tokens", 1000)
)

# 3. 监控摘要触发频率
# 如果频繁触发 → 提高阈值
# 如果从不触发 → 降低阈值
```

## 核心要点

1. **默认问题**：对话历史无限增长
2. **推荐方案**：`SummarizationMiddleware`
3. **配置位置**：`middleware=[]` 参数
4. **触发条件**：`trigger=("tokens", N)` 或 `trigger=("messages", N)`
5. **自动化**：无需手动管理

## 下一步

**09_checkpointing** - 学习如何持久化对话状态（SQLite）
