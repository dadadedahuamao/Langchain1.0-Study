# CLAUDE.md — LangChain 1.0 学习仓库开发指南

## 项目定位

LangChain 1.0 系统学习仓库，五阶段渐进式学习路径。代码目标是让学习者按顺序运行、观察输出、理解概念并继续扩展——不是炫技。

每个模块的 main.py 由独立的 `example_N_xxx()` 函数组成，从简单到复杂逐步推进。

## 目录结构

```
Langchain1.0-Study/
├── phase1_fundamentals/              # 阶段一：基础知识 ✅ 已完成
│   ├── 01_hello_langchain/
│   ├── 02_prompt_templates/
│   ├── 03_messages/
│   ├── 04_custom_tools/
│   ├── 05_simple_agent/
│   └── 06_agent_loop/
├── phase2_practical/                 # 阶段二：实战进阶 ✅ 已完成
│   ├── 07_memory_basics/
│   ├── 08_context_management/
│   ├── 09_checkpointing/
│   ├── 10_middleware_basics/
│   ├── 11_structured_output/
│   ├── 12_validation_retry/
│   ├── 13_rag_basics/
│   └── 14_rag_advanced/
├── phase3_advanced/                  # 阶段三：LangGraph 进阶 ✅ 已完成
│   ├── 15_langgraph_low_level/       # StateGraph、节点、边、reducer ✅
│   ├── 16_multi_agent/               # create_react_agent、Supervisor、Handoff ✅
│   ├── 17_human_in_the_loop/         # 打断、审批、编辑 ✅
│   ├── 18_subgraphs/                 # 子图嵌套、图组合 ✅
│   ├── 19_streaming_and_events/      # 底层 stream、节点级流式 ✅
│   ├── 20_production_ready/          # Checkpoint、LangSmith、错误处理、成本 ✅
│   └── 21_multimodal_files/          # 图像理解、PDF 解析、多模态消息 ✅
├── phase4_frontier/                  # 阶段四：生产与前沿 🆕
│   ├── 22_mcp/                       # MCP 协议
│   ├── 23_sql_agent/                 # SQL Agent
│   ├── 24_guardrails/                # 安全护栏
│   ├── 25_long_term_memory/          # 长期记忆
│   ├── 26_langgraph_functional_api/  # 函数式 API
│   ├── 27_advanced_multi_agent/      # 高级多Agent
│   ├── 28_time_travel/               # 时间旅行
│   ├── 29_deep_agents/               # Deep Agents
│   ├── 30_testing/                   # Agent 测试
│   └── 31_local_server/              # 本地服务
├── phase5_projects/                  # 阶段五：综合项目（部分完成）
│   ├── 01_data_analysis/             # ⏳ 数据分析 Agent（待创建）
│   ├── 02_saas_agent/                # ⏳ 全栈 SaaS Agent（待创建）
│   └── 03_enterprise_rag/            # ✅ 企业级 RAG（已完成）
├── examples/                          # 扩展示例
│   └── HBRS-Chem/                    # 危险化工行为识别预警
├── docs/                              # 文档
├── asset/                             # 静态资源
└── requirements.txt
```

## 工作原则

- 修改前先阅读相关模块的 `README.md`、`main.py`、`test.py`，保持同一模块内风格一致
- 不覆盖用户已有改动；遇到无关的未提交文件或变更，保持原样
- 新增或改写 LangChain / LangGraph 代码前，必须查询官方文档核实最新 API，不能凭记忆猜 import 路径、参数名或废弃接口
- 示例代码要适合学习：步骤清楚、输出完整、错误提示友好、依赖说明明确
- 避免引入大型新依赖；确需新增依赖时，同步更新 `requirements.txt` 和对应 README

## 严格规则：必须查询真实文档

编写任何新模块代码前，**必须联网查询 LangChain 官方文档**，核实最新的 API 语法。禁止猜测或杜撰 API。

核实内容包括但不限于：
- import 路径是否正确（LangChain 1.0 的包结构变化很大）
- 函数签名、参数名、默认值
- 类的继承关系和必需方法
- 废弃的 API 替代方案

主要文档来源：
- https://docs.langchain.com/oss/python/langchain/overview
- https://docs.langchain.com/oss/python/langgraph/overview
- https://docs.langchain.com/llms.txt

## main.py 结构规范

每个 main.py 由三部分组成：**文件头+环境配置 → 一系列 example 函数 → main() 串联执行**。

main.py 的定位是"可运行课堂演示"——可以用少量注释解释关键语法，但不要把知识点长篇解释写进 print 输出；完整概念、语法拆解、类比说明统一放到 README。

### 文件头

```python
"""
LangChain 1.0 - 模块中文名 (English Name)
==============================================

本模块重点讲解：
1. 要点一
2. 要点二
3. 要点三
"""
```

### 环境配置

```python
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

# ============================================================================
# 环境配置
# ============================================================================

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_API_BASE")

if not API_KEY:
    raise ValueError("请先设置 OPENAI_API_KEY")

model = init_chat_model(
    "openai:glm-5.1",
    api_key=API_KEY,
    base_url=BASE_URL,
)
```

如果维护旧模块中已有 Groq、Pinecone、Chroma 等配置，先保持模块内一致；只有在任务明确要求统一模型时再迁移。

### 代码注释规范（重要）

注释是帮助学习者读懂代码的关键手段，**不能省略**，但要克制、精准。

**必须保留注释的场景**：
1. **新框架、新 API、新语法首次出现时**——必须加注释说明是什么、为什么用
2. **非显而易见的逻辑**——解释"这行/这段为什么存在"，不要只复述代码表面含义
3. **新知识点旁边**——例如 `StateGraph`、`START`、`END`、`add_node`、`add_edge`、`add_conditional_edges`、reducer、装饰器、类型注解、checkpointer 等
4. **关键函数的 docstring**——说明示例目的和核心概念
5. **复杂流程前**——用 1-3 行注释说明流程结构或设计意图

**注释写法原则**：
- 解释 **WHY（为什么）**，不只是 **WHAT（是什么）**
- 保持短小精悍，不写成教程段落；长解释放 README
- 用中文写注释，技术术语保留英文
- 复杂语法拆成小段注释，不要一大段注释块堆在一起

**注释示例**：

```python
# 好：解释为什么这样做
builder.add_conditional_edges("classify", route_by_sentiment)
# ↑ 条件路由：根据 classify 节点的输出决定下一步走哪个分支

# 好：解释新概念
from langgraph.graph import StateGraph, START, END
# StateGraph: LangGraph 的核心类，用来定义工作流图
# START / END: 特殊常量，标记图的入口和出口

# 坏：只复述代码
model = init_chat_model("openai:glm-5.1")  # 初始化模型
result = model.invoke("hello")  # 调用模型
```

**注释与 print 输出的分工**：
- **代码注释**：帮助阅读源码的人理解关键语法和设计意图——要充分
- **print 输出**：展示运行结果和关键结论——要克制
- **README**：完整的知识点讲解、语法拆解、背景说明——要详尽

不要为了减少 print 而把所有解释都删掉。终端输出要少，但代码注释要足够帮助学习者读懂关键语法。

### print 输出规范

终端输出要克制，避免把示例变成大段日志。详细语法解释、逐句说明、背景知识应主要写在 README，不要全部塞进 `print()`。

**输出原则**：
- 每个 example 只打印：标题、必要输入、关键中间结果、最终结果、2-3 条关键点
- 不要在终端重复打印 README 里已经详细解释过的概念
- 不要把整段教程、长表格、长映射关系、完整源码流程用 `print()` 输出
- 节点较多的工作流只打印关键节点执行，不要每个细节都打日志
- 调试类 print 仅在必要时保留，完成后删除临时排查输出
- 如果需要展示流程结构，优先用 1-3 行简短文本，例如 `START -> classify -> answer -> END`
- 单个 example 的终端输出建议控制在一屏内；复杂示例也应尽量少于 30 行

### example 函数

```python
# ============================================================================
# 示例 N：主题
# ============================================================================
def example_N_topic():
    """
    示例N：简短描述

    关键：
    1. 要点一
    2. 要点二
    """
    print("\n" + "="*70)
    print("示例 N：标题")
    print("="*70)

    # ... 具体代码 ...

    print(f"结果: {result}")
    print("\n关键点：")
    print("  - 要点1")
    print("  - 要点2")
```

**设计原则**：
- 每个 example 尽量独立可理解，包含完整 print 输出
- 编号递增 = 复杂度递增，example_1 最简单
- 每个用 `# ====...====` 注释块分隔
- 最后一个 example 通常是最佳实践总结
- 示例数量通常 4-7 个
- 如需要向后传递对象，明确 `return`，并在 `main()` 中接收

### main() 函数

```python
# ============================================================================
# 主程序
# ============================================================================
def main():
    print("\n" + "="*70)
    print(" LangChain 1.0 - 模块标题")
    print("="*70)

    try:
        example_1_xxx()
        input("\n按 Enter 继续...")
        example_2_xxx()
        input("\n按 Enter 继续...")
        # ... 依次调用
        example_N_xxx()

        print("\n" + "="*70)
        print(" 完成！")
        print("="*70)
        print("\n核心要点：")
        print("  ✅ 要点1")
        print("  ✅ 要点2")
        print("\n下一步：")
        print("  下一模块 - 学习内容")

    except KeyboardInterrupt:
        print("\n\n程序中断")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
```

- 示例之间用 `input("\n按 Enter 继续...")` 暂停
- 末尾打印"完成！"+ 核心要点 + 下一步引导
- 最后一个 example 后不暂停

## 编码约定

### 通用

- 注释、print 输出、README 全部**中文**，技术术语保留英文
- 变量名、函数名、类名使用英文
- 模型统一用 `init_chat_model`，通过 LiteLLM 代理调用，格式 `"openai:glm-5.1"`
- API Key 和 base_url 从环境变量 `OPENAI_API_KEY`、`OPENAI_API_BASE` 读取
- Agent 示例优先使用 `from langchain.agents import create_agent`

### 分隔线

| 位置 | 写法 |
|------|------|
| example 之间、环境配置区、主程序区 | `# ============================================================================` |
| print 主标题 | `print("="*70)` |
| print 次要分隔 | `print("-"*70)` |

### 输出符号

仅用于四种场景：`✅` 成功 / `❌` 失败 / `⚠️` 警告 / `💡` 提示

同一文件内保持一种风格，不要混用过多符号。代码 print 输出中也可使用纯文本 `[OK]`、`[ERROR]`、`[SKIP]`。

### 工具定义

```python
from langchain_core.tools import tool


@tool
def tool_name(param: str) -> str:
    """工具描述

    参数:
        param: 参数说明

    返回:
        返回值说明
    """
    return "结果"
```

### 测试文件

不使用 pytest，直接执行的 print 脚本。测试脚本应尽量不依赖付费 API；若必须依赖 API key，要在输出中明确提示。测试输出保持中文，展示成功、跳过、失败原因和下一步建议。

### Windows 编码

文件读写使用 UTF-8。需要处理 Windows 控制台中文乱码时，在文件头附近加入：

```python
import sys

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
```

## README 规范

模块 README 是主要教学载体，必须承担知识点解释和语法解释；`main.py` 负责可运行演示，不负责输出整篇教程。

建议结构：

1. 快速开始
2. 核心概念
3. 工作流程或执行过程
4. 关键代码片段
5. 常见问题
6. 最佳实践
7. 下一步学习

推荐教学顺序：

```text
1. 这个东西是干嘛的
2. 它像你已经知道的什么
3. 最小例子
4. 逐句语法解释
5. 完整示例
6. 常见问题和最佳实践
```

每个新模块 README 应包含：

- **为什么学这个**：说明这个模块解决什么问题，最好给一个真实场景或熟悉工具类比
- **和已有工具/概念的类比**：例如 LangGraph 可以类比 Dify 工作流，RAG 可以类比"先查资料再回答"
- **核心知识点解释**：先讲概念，再讲代码，不要直接堆 API
- **最小语法示例**：给一个最小可运行片段
- **逐句语法解释**：解释关键 import、类、函数、参数、返回值、调用流程
- **常用语法速查**：把高频 API 写成简短表格或代码块
- **完整示例说明**：说明 `main.py` 每个 example 展示什么
- **易混点/常见错误**：说明废弃导入、错误参数、状态覆盖与追加等坑
- **练习建议**：给学习者 1-3 个可以自己改的方向

### README 语法解释要求

- 不只写"这样用"，还要写"为什么这样写"
- 每个核心 API 都解释参数含义和返回值
- 复杂语法要拆成 3-8 行最小代码，不要一上来给完整大段
- 对学习者可能不懂的 Python 语法也要解释，例如 `TypedDict`、`Annotated`、装饰器、类型注解
- 如果示例用了第三方库，说明该库在当前流程中扮演什么角色

示例：

```markdown
### `StateGraph(State)` 是什么意思？

`StateGraph` 用来创建一个工作流图。

`State` 告诉 LangGraph：这个工作流有哪些变量，节点之间会传递哪些数据。

\```python
builder = StateGraph(State)
\```

可以理解成：创建一张 Dify 画布，并提前定义画布里会流动的变量。
```

README 中的技术说明要准确，涉及 LangChain 1.0 API 迁移、包名变化、废弃接口时必须写清楚正确导入方式和错误示例。

## 模块目录模板

```
编号_英文主题/
├── main.py       # 必须 —— example_1 ~ example_N + main()
├── README.md     # 必须 —— 核心概念 → 用法 → 流程 → FAQ → 最佳实践 → 下一步
├── test.py       # 可选 —— 直接执行式脚本
├── tools/        # 按需 —— 自定义工具
└── data/         # 按需 —— 示例数据
```

## 提交前检查

完成修改后至少检查：

- 新增/修改文件是否为 UTF-8
- `main.py` 是否能按顺序运行到需要外部 API 的步骤
- 没有把 `.env`、API key、数据库、向量索引、缓存目录提交进来
- 新增依赖是否已写入 `requirements.txt`
- README 中的运行命令与真实路径一致
- LangChain / LangGraph API 已按官方文档核实
- `git status --short` 中只包含本次任务相关文件

## 禁止事项

- 禁止硬编码真实 API key、token、密钥或私有 base URL
- 禁止无依据编造 LangChain API
- 禁止把学习示例写成难以阅读的生产级抽象
- 禁止随意重构无关模块
- 禁止删除用户已有文件或回滚用户未提交改动，除非用户明确要求
