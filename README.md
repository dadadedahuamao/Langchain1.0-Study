# LangChain 1.0 学习仓库

这是一个系统学习 **LangChain 1.0** 的实践仓库，涵盖从基础概念到实战项目的完整学习路径。

## 📚 关于 LangChain 1.0

LangChain 1.0 是用于构建 LLM 驱动应用程序的框架的最新主要版本（2024年9月发布）。主要特性：

- ✅ **构建在 LangGraph 运行时之上** - 提供持久化、流式处理、人在回路等能力
- ✅ **新的 `create_agent` API** - 简化 Agent 创建流程
- ✅ **中间件架构** - 提供细粒度的执行控制（before_model、after_model、wrap_model_call 等）
- ✅ **多模态支持** - 处理文本、图像、视频、文件
- ✅ **结构化输出** - 使用 Pydantic 模型定义输出格式
- ✅ **语义化版本控制** - 1.x 系列保证 API 稳定

## 🚀 快速开始

### 环境要求

- Python 3.10 或更高版本（不支持 Python 3.9）
- pip 或 uv 包管理器

### 安装步骤

1. **克隆仓库**
```bash
git clone https://github.com/Mason-zy/Langchain1.0-Study.git
cd langchain_v1_study
```

2. **创建虚拟环境**
```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Unix/macOS:
source venv/bin/activate
```

3. **安装依赖**
```bash
pip install -r requirements.txt
```

4. **配置环境变量**
```bash
# 复制环境变量模板
cp .env.example .env

# 编辑 .env 文件，填入你的 API Keys
```

需要的 API Keys：
- `OPENAI_API_KEY` - OpenAI API 密钥（https://platform.openai.com/api-keys）
- `ANTHROPIC_API_KEY` - Anthropic API 密钥（https://console.anthropic.com/）
- `LANGSMITH_API_KEY` - LangSmith API 密钥（可选，用于可观测性）

### 验证安装

运行第一个示例：
```bash
python phase1_fundamentals/01_hello_langchain/main.py
```

## 📖 学习路径

本仓库采用**五阶段渐进式学习**，共 31 个模块 + 3 个综合项目：

### 阶段一：基础知识（第1-2周）
📂 `phase1_fundamentals/`

| 模块 | 主题 | 学习内容 |
|------|------|----------|
| 01 | Hello LangChain | 第一次 LLM 调用，使用 `init_chat_model` |
| 02 | Prompt Templates | 创建和使用提示词模板 |
| 03 | Messages | 消息类型（System、Human、AI） |
| 04 | Custom Tools | 自定义工具（天气、计算器、搜索） |
| 05 | Simple Agent | 使用 `create_agent` 创建 Agent |
| 06 | Agent Loop | 理解 ReAct 模式执行循环 |

### 阶段二：实战进阶（第3-4周）
📂 `phase2_practical/`

| 模块 | 主题 | 学习内容 |
|------|------|----------|
| 07 | Memory Basics | 使用 InMemorySaver 实现短期内存 |
| 08 | Context Management | 消息修剪和摘要 |
| 09 | Checkpointing | 使用 SQLite/Postgres 持久化状态 |
| 10 | Middleware Basics | before_model 和 after_model 钩子 |
| 11 | Structured Output | 使用 Pydantic 定义输出模式 |
| 12 | Validation Retry | 优雅地处理验证失败 |
| 13 | RAG Basics | 文档加载、向量存储、检索增强生成 |
| 14 | RAG Advanced | 混合检索、重排序、高级 RAG 策略 |

### 阶段三：LangGraph 进阶（第5-6周）✅ 已完成
📂 `phase3_advanced/`

| 模块 | 主题 | 学习内容 |
|------|------|----------|
| 15 | LangGraph Low Level | StateGraph、节点、边、reducer、条件路由 |
| 16 | Multi-Agent | create_react_agent、Supervisor、Handoff |
| 17 | Human-in-the-Loop | 打断、审批、编辑（LangGraph 独有） |
| 18 | Subgraphs | 子图嵌套、父图调用子图、图组合 |
| 19 | Streaming & Events | 底层 stream()、节点级流式、事件过滤 |
| 20 | Production Ready | Checkpoint 进阶、LangSmith、错误处理、成本 |
| 21 | Multimodal & Files | 图像理解、PDF 解析、多模态消息 |

### 阶段四：生产与前沿（第7-8周）🆕 待开发
📂 `phase4_frontier/`

| 模块 | 主题 | 学习内容 |
|------|------|----------|
| 22 | MCP (Model Context Protocol) | Agent 通过标准协议连接外部工具和数据源 |
| 23 | SQL Agent | 自然语言查询数据库，Text-to-SQL |
| 24 | Guardrails | 安全护栏：内容过滤、输出校验、敏感信息屏蔽 |
| 25 | Long-Term Memory | 跨会话持久化记忆（Store API） |
| 26 | LangGraph Functional API | 函数式 API，与 Graph API 互补的新写法 |
| 27 | Advanced Multi-Agent | Router / Skills / Subagents 三种高级模式 |
| 28 | Time Travel & Durable Execution | 回退历史状态、断点续跑 |
| 29 | Deep Agents | LangChain 2025 重点推出的"全装电池"Agent 框架 |
| 30 | Agent Testing | pytest 单元测试、集成测试、Agent 评估 |
| 31 | Local Server & RemoteGraph | 启动 API 服务、RemoteGraph 远程调用 |

### 阶段五：综合项目（第9-10周）🔨 部分完成
📂 `phase5_projects/`

| 项目 | 描述 | 核心技术 | 状态 |
|------|------|----------|------|
| 01 Data Analysis Agent | CSV/Excel 读取→分析→可视化→报告 | Deep Agents、Sandbox、Skills、Streaming | ⏳ 待创建 |
| 02 全栈 SaaS Agent | 用户管理、对话历史、API 服务、Agent 后台 | LangGraph Server、RemoteGraph、FastAPI | ⏳ 待创建 |
| 03 Enterprise RAG | 企业级文档问答系统 | RAG、LangGraph、子图、可观测性 | ✅ 已完成 |

## 📁 项目结构

```
Langchain1.0-Study/
├── phase1_fundamentals/     # 阶段一：基础知识 ✅
├── phase2_practical/        # 阶段二：实战进阶 ✅
├── phase3_advanced/         # 阶段三：LangGraph 进阶 ✅
│   ├── 15_langgraph_low_level/      ✅
│   ├── 16_multi_agent/              ✅
│   ├── 17_human_in_the_loop/        ✅
│   ├── 18_subgraphs/                ✅
│   ├── 19_streaming_and_events/     ✅
│   ├── 20_production_ready/         ✅
│   └── 21_multimodal_files/         ✅
├── phase4_frontier/         # 阶段四：生产与前沿 🆕
│   ├── 22_mcp/                      待创建
│   ├── 23_sql_agent/                待创建
│   ├── 24_guardrails/               待创建
│   ├── 25_long_term_memory/         待创建
│   ├── 26_langgraph_functional_api/ 待创建
│   ├── 27_advanced_multi_agent/     待创建
│   ├── 28_time_travel/              待创建
│   ├── 29_deep_agents/              待创建
│   ├── 30_testing/                  待创建
│   └── 31_local_server/             待创建
├── phase5_projects/         # 阶段五：综合项目（部分完成）
│   ├── 01_data_analysis/             ⏳
│   ├── 02_saas_agent/                ⏳
│   └── 03_enterprise_rag/            ✅
├── examples/                # 扩展示例项目
├── docs/                    # 学习笔记和文档
├── asset/                   # 静态资源
└── requirements.txt         # Python 依赖
```


## 🎯 使用指南

### 运行单个模块

```bash
# 进入模块目录
cd phase1_fundamentals/01_hello_langchain

# 运行主程序
python main.py

# 运行测试（如果有）
python test.py
```

### 运行综合项目

```bash
# 进入项目目录
cd phase5_projects/03_enterprise_rag

# 安装项目特定依赖
pip install -r requirements.txt

# 运行项目
python main.py
```

### 使用 Jupyter Notebook

```bash
# 安装 Jupyter
pip install jupyter

# 启动 Notebook
jupyter notebook notebooks/
```

## 📝 学习建议

1. **按顺序学习** - 从阶段一开始，每个模块都基于前面的知识
2. **动手实践** - 每个模块都有可运行的代码，修改参数观察效果
3. **记录笔记** - 在 `docs/learning_notes/` 中记录你的学习心得
4. **查看 README** - 每个模块都有独立的 README.md 说明核心概念
5. **完成测试** - 运行测试文件验证你的理解
6. **做综合项目** - 前三阶段完成后，通过 Phase 4 项目巩固所学
7. **前沿拓展** - Phase 4 涵盖 MCP、Deep Agents、SQL Agent 等最新特性
8. **综合实战** - Phase 5 通过完整项目融会贯通所有知识点

## 🔧 常用命令

```bash
# 查看已安装的包
pip list

# 更新某个包
pip install --upgrade langchain

# 激活 LangSmith 追踪（可选）
export LANGSMITH_TRACING=true  # Unix/macOS
set LANGSMITH_TRACING=true     # Windows

# 运行全局测试
pytest tests/
```

## 📚 重要资源

- **官方文档**: https://docs.langchain.com/oss/python/langchain/overview
- **迁移指南**: https://docs.langchain.com/oss/python/migrate/langchain-v1
- **LangGraph 文档**: https://docs.langchain.com/oss/python/langgraph/overview
- **LangSmith 平台**: https://smith.langchain.com
- **GitHub 仓库**: https://github.com/langchain-ai/langchain

## 🆘 常见问题

### 1. 导入错误：ModuleNotFoundError

确保虚拟环境已激活并且安装了所有依赖：
```bash
pip install -r requirements.txt
```

### 2. API Key 错误

检查 `.env` 文件是否正确配置，确保 API Keys 有效。

### 3. Python 版本不兼容

LangChain 1.0 需要 Python 3.10+：
```bash
python --version  # 检查版本
```


## 🤝 贡献

这是个人学习仓库，欢迎提交问题和改进建议！

## 📄 许可证

MIT License

## 🎓 关于作者

正在学习 LangChain 1.0 的开发者，记录学习过程供参考。如果感觉有用的话，请作者喝杯咖啡吧。😊
![alt text](asset/41954bb0679c01a8c04e888f28e18207.jpg)
---

**开始学习之旅** 👉 [01_hello_langchain](./phase1_fundamentals/01_hello_langchain/)
