# 企业级 RAG 系统知识总结

> 本文件整理自对该项目的深入学习问答，涵盖架构设计、核心概念、评估方法等关键知识点。

---

## 目录

1. [双图架构：入库图与检索图](#1-双图架构入库图与检索图)
2. [LangGraph 是必须的吗](#2-langgraph-是必须的吗)
3. [文档格式支持](#3-文档格式支持)
4. [表格检测机制](#4-表格检测机制)
5. [文本切分机制](#5-文本切分机制)
6. [检索管线六步全流程](#6-检索管线六步全流程)
7. [重排序（Reranking）详解](#7-重排序reranking详解)
8. [CRAG：纠正性 RAG](#8-crag纠正性-rag)
9. [如何评估 RAG 效果](#9-如何评估-rag-效果)
10. [主流评估框架对比](#10-主流评估框架对比)

---

## 1. 双图架构：入库图与检索图

### 整体类比

把 RAG 系统想象成一座图书馆：

| 角色 | 类比 | 对应组件 |
|------|------|----------|
| 入库图 | 图书采购员 — 把新书分类、贴标签、上架 | Ingestion Graph |
| 向量库 | 书架 — 存放已处理好的文档 | ChromaDB |
| 检索图 | 咨询台 — 接受读者提问，去书架找书，整理答案 | Retrieval Graph |

### 入库图（Ingestion Graph）

```
PDF 文件 → 解析(parse) → 切片(chunk) → 向量化+入库(embed & store) → ChromaDB
```

**职责**：把原始文档变成可检索的向量。具体做了三件事：

1. **解析（parse）**：用 PyPDFLoader 把 PDF 按页读取，每页生成一个 Document 对象
2. **切片（chunk）**：用 RecursiveCharacterTextSplitter 把长文本切成 500 字符的小块
3. **入库（store）**：调用 Embedding 模型把每个块转成向量，存入 ChromaDB

对应代码：`ingestion.py` 中的 `build_ingestion_graph()`，流程为 `START → parse → chunk → store → END`。

### 检索图（Retrieval Graph）

```
用户查询 → 改写(rewrite) → [向量检索 + BM25检索] → 合并(merge) → 重排序(rerank) → 评分(grade) → 生成/兜底 → 回答
                                         ↑ 并行执行
```

**职责**：拿到用户问题，找到最相关的文档，生成回答。

对应代码：`retrieval.py` 中的 `build_retrieval_graph()`，包含 fan-out（并行检索）和条件路由（评分后走生成或兜底）。

### 两图的关系

入库图和检索图**不直接通信**。它们通过 ChromaDB 这个共享存储间接连接：
- 入库图写 ChromaDB
- 检索图读 ChromaDB

这就像图书馆的采购员不需要认识咨询台的工作人员，他们通过"书架"这个中间层协作。

---

## 2. LangGraph 是必须的吗

### 短回答

不是必须的。代码中所有核心功能（解析、切片、检索、重排序等）都是普通 Python 函数，可以单独调用。LangGraph 是可选的编排层。

### 用与不用的区别

| 维度 | 直接调用函数 | 用 LangGraph 编排 |
|------|-------------|-------------------|
| 代码量 | 更少，更直接 | 多一些定义（State、节点、边） |
| 并行执行 | 需要手写 `threading` / `asyncio` | 自动 fan-out 并行（向量检索和 BM25 同时跑） |
| 条件路由 | 手写 `if/else` | 声明式 `add_conditional_edges` |
| 状态管理 | 手动传参 | State TypedDict + 自动 reducer 合并 |
| 可观测性 | 需自己加日志 | 每个节点的输入/输出自动可追踪 |
| 复杂度 | 线性流程很简洁 | 简单线性流程有点"杀鸡用牛刀" |

### 什么时候值得用 LangGraph

- **需要并行**：向量检索 + BM25 同时跑，LangGraph 的 fan-out 天然支持
- **需要条件分支**：评分够就生成，不够就走兜底，用 `add_conditional_edges` 很清晰
- **需要 reducer**：并行结果自动合并（`Annotated[list[Document], operator.add]`），不用手写合并逻辑
- **需要可观测性**：每个节点自动追踪，方便调试和 LangSmith 集成

**简单线性流程（入库图）**：LangGraph 带来的收益不大，直接调用函数也完全 OK。

**有并行和分支的流程（检索图）**：LangGraph 的收益明显，代码更清晰。

---

## 3. 文档格式支持

### 当前支持

| 格式 | 状态 | 对应 Loader |
|------|------|------------|
| PDF（文本型） | 已支持 | `PyPDFLoader` |
| PDF（含表格/图片） | 可选支持 | `UnstructuredPDFLoader`（需额外依赖） |

### 如何扩展到其他格式

只需替换解析器函数，后面的切片 → 向量化 → 入库完全不变：

| 格式 | 对应 Loader | 安装方式 |
|------|------------|---------|
| Markdown | `UnstructuredMarkdownLoader` 或 `TextLoader` | 内置 |
| HTML | `UnstructuredHTMLLoader` 或 `BSHTMLLoader` | `pip install beautifulsoup4` |
| Word (.docx) | `Docx2txtLoader` 或 `UnstructuredWordDocumentLoader` | `pip install docx2txt` |
| CSV/Excel | `CSVLoader` / `DataFrameLoader` | 内置 / `pip install pandas` |
| 纯文本 | `TextLoader` | 内置 |

**扩展方式**：在 `ingestion.py` 中新增一个 `parse_xxx()` 函数，在入库图的 parse 节点中选择调用即可。

### 扫描件 PDF（图片型）

扫描件 PDF 的每一页实际是图片，PyPDFLoader 提取不出文字。解决方案：

| 方案 | 原理 | 适用场景 |
|------|------|---------|
| `pytesseract` + `pdf2image` | 传统 OCR，把 PDF 转图片再识别文字 | 简单扫描件 |
| `unstructured` OCR 模式 | 内置 OCR 能力 | 中等复杂度 |
| 多模态 LLM（GPT-4o / Qwen-VL） | 用视觉模型直接"看"PDF 页面 | 复杂排版、手写体 |

---

## 4. 表格检测机制

### 核心原理

表格检测**不是靠分析内容**，而是靠解析器给文档打标签。

- `PyPDFLoader`：**无法检测表格**，它只把每页当纯文本输出
- `UnstructuredPDFLoader`（mode="elements"）：能识别文档结构，给每个元素标注类型

### UnstructuredPDFLoader 的标注方式

解析后的每个 Document 在 metadata 中包含 `element_type` 字段：

```python
# 正文段落
Document(metadata={"element_type": "NarrativeText", ...})

# 表格
Document(metadata={"element_type": "Table", ...})

# 标题
Document(metadata={"element_type": "Title", ...})
```

### 代码中的分流处理

```python
# ingestion.py - separate_by_element_type()
for doc in documents:
    elem_type = doc.metadata.get("element_type", "text")
    if elem_type == "Table":
        table_docs.append(doc)    # 表格：保留原文，不切片
    else:
        text_docs.append(doc)     # 文本：走正常切片
```

**设计意图**：表格切成碎片会丢失结构信息，所以表格类元素不切片，整块保留。

---

## 5. 文本切分机制

### 切分不是按页，是按字符数 + 语义边界

**常见误解**：以为切分就是按页切（PyPDFLoader 按页读取 → 所以切分也按页）。

**实际情况**：
1. PyPDFLoader 按页读取 → 每页一个 Document（这是"读取"，不是"切分"）
2. RecursiveCharacterTextSplitter 对每个 Document 再切分 → 这才是真正的"切分"

### RecursiveCharacterTextSplitter 工作原理

```
目标大小 = 500 字符（CHUNK_SIZE）
重叠大小 = 50 字符（CHUNK_OVERLAP）

切分顺序（从粗到细）：
1. 先尝试按 "\n\n"（段落）切 → 每块 ≤ 500 字符？OK 就停
2. 还是太长？按 "\n"（换行）切
3. 还是太长？按 "。" "！" "？"（中文句号）切
4. 还是太长？按 "."（英文句号）切
5. 还是太长？按 " "（空格）切
6. 最后兜底：按单个字符切
```

**关键参数**：
- `chunk_size=500`：目标块大小（字符数，不是 token 数）
- `chunk_overlap=50`：相邻块重叠 50 字符，防止关键信息被截断在两个块的边界

### 为什么需要 overlap

```
块1: "...LangChain 1.0 发布于 2024 年，引入了|全新架构"  ← 500字符截止
块2: "全新架构，支持 LangGraph 和 |LangSmith..."         ← 下一个500字符

如果没有 overlap，"全新架构"的上下文就丢了。
加了 overlap 后：
块1: "...引入了全新架构"    ← 多带50字符
块2: "引入了全新架构，支持..."  ← 往前多带50字符
→ 关键信息在两块中都能找到
```

---

## 6. 检索管线六步全流程

用户提问后的完整检索流程，每一步对应 `retrieval.py` 中的一个函数：

### Step 1：查询改写（rewrite_query）

**目的**：把一个问题变成多个问法，扩大召回范围。

```python
# 用户问："哪个模型 MMLU 分数最高？"
# 改写后：
#   1. "哪个模型 MMLU 分数最高？"          ← 原始
#   2. "各模型在 MMLU 基准测试中的排名"    ← 变体1
#   3. "MMLU benchmark 最高分模型是哪个"   ← 变体2
```

**为什么需要**：用户的表达方式和文档的表达方式可能不同。改写增加了"命中"的概率。

### Step 2：多路检索（vector_search + bm25_search）

两路检索**并行执行**（LangGraph fan-out）：

| 检索方式 | 原理 | 擅长 | 短板 |
|----------|------|------|------|
| 向量检索 | Embedding → 余弦相似度 | 语义相似（同义词、改写） | 精确关键词匹配弱 |
| BM25 检索 | 词频统计 | 精确关键词、专业术语 | 不理解语义 |

**互补性**：向量检索能找到"性能优化"即"提升速度"，BM25 能精确匹配 "MiniLM-L-6-v2" 这样的专有名词。

### Step 3：合并去重（merge_and_deduplicate）

两路检索的结果可能有重叠（同一个文档被两种方式都找到了）。合并时：
- 向量检索结果优先排列
- 用内容哈希（MD5）去重
- 保留先出现的文档

### Step 4：重排序（rerank_documents）

用 CrossEncoder 模型对合并后的结果做精确排序。详见[下一节](#7-重排序reranking详解)。

### Step 5：文档评分（grade_documents）

用 LLM 判断：**检索到的文档能不能回答用户的问题？**

- 评分 0.0 ~ 1.0
- ≥ 0.7：文档充分 → 走生成
- < 0.7：文档不足 → 走兜底

这就是 CRAG 的核心思想。详见[第 8 节](#8-crag纠正性-rag)。

### Step 6：生成回答（generate_answer）/ 兜底（fallback）

- **生成**：把文档拼成上下文，让 LLM 基于文档内容回答，标注来源
- **兜底**：告诉用户"知识库中没有足够信息"，建议换个问法

---

## 7. 重排序（Reranking）详解

### 中文名称

Reranking 的中文叫 **重排序** 或 **精排序**。

### 为什么需要重排序

初始检索（向量 / BM25）是"粗筛"——快速从大量文档中捞出候选集，但排序不一定准确。重排序用更精确的模型重新打分排序，把最相关的排到最前面。

### 双塔模型 vs CrossEncoder

| | 双塔模型（Bi-Encoder） | CrossEncoder |
|--|----------------------|--------------|
| **原理** | query 和 doc 分别编码成向量，算向量距离 | query 和 doc 拼接后一起编码 |
| **速度** | 快（向量可以预计算） | 慢（每次都要重新编码） |
| **精度** | 一般 | 高 |
| **适用阶段** | 初始检索（从万级文档中粗筛） | 重排序（从几十个候选中精选） |

**类比**：
- 双塔模型 = 海选：评委看一眼选手照片，快速筛选
- CrossEncoder = 决赛：评委和选手面对面深入交流，精确评分

### 本项目使用的重排序模型

```python
# config.py
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
```

这是一个轻量级的 CrossEncoder 模型（约 80MB），通过 `lru_cache` 缓存，首次加载后复用。

---

## 8. CRAG：纠正性 RAG

### 核心思想

传统 RAG 的流程是：检索 → 生成。它**默认信任检索结果**。

CRAG（Corrective RAG）在检索和生成之间加了一道"质检"：

```
传统 RAG：检索 → 生成（盲目信任检索结果）
CRAG：    检索 → 评估 → 够用？→ 生成
                   └→ 不够？→ 兜底（外部搜索/告知用户）
```

### 本项目的实现

```python
# retrieval.py - grade_documents()
# LLM 评估检索结果是否充分
score = grade_documents(query, reranked_docs, llm)

# graph_nodes.py - route_by_score()
# 根据评分路由
if score >= 0.7:    # 文档充分
    → generate      # 基于文档生成回答
else:               # 文档不足
    → fallback      # 告知用户无法回答
```

### 评分机制

LLM 看到问题和检索到的前 5 个文档，输出一个 0.0~1.0 的评分：

| 评分 | 含义 | 路由 |
|------|------|------|
| 1.0 | 完全可以回答 | generate |
| 0.7 | 基本可以回答 | generate |
| 0.4 | 有部分相关信息 | fallback |
| 0.0 | 完全无关 | fallback |

评分解析用了正则提取（`_parse_score`），容忍 LLM 输出 "评分：0.9"、"0.7/1.0" 等非标准格式。

---

## 9. 如何评估 RAG 效果

### 评估的两个维度

| 维度 | 评估什么 | 对应模块 |
|------|---------|---------|
| **检索质量** | 找到的文档相不相关？排得对不对？ | `evaluation.py`（已实现） |
| **生成质量** | 生成的回答准不准？有没有编造？ | 尚未实现 |

### 本项目已实现的检索评估指标

`evaluation.py` 中的 `RetrievalEvaluator` 类提供了以下指标：

| 指标 | 含义 | 计算方式 |
|------|------|---------|
| **Hit Rate@1** | 第一个结果就相关的比例 | 第 1 个文档相关 → 命中 |
| **Hit Rate@3** | 前 3 个结果中有相关的比例 | 前 3 个中有任何一个相关 → 命中 |
| **MRR** | 第一个相关文档的平均排名倒数 | 如果第 2 个相关 → 1/2 = 0.5 |
| **Avg Relevance Score** | LLM 整体充分性评分均值 | 0.0~1.0 |
| **Avg Latency** | 平均检索耗时 | 毫秒 |

### 评估方法：LLM-as-Judge

本项目不依赖人工标注，而是用 LLM 自己判断检索结果的质量：

- **`_judge_single_relevance()`**：逐个文档判断是否相关（Yes/No 二元判断）
- **`grade_documents()`**：整体判断文档集是否足以回答问题（0.0~1.0 评分）

### 评估报告示例

```
检索质量评估报告
  测试查询数: 3
  平均延迟:   1523 ms
  Hit Rate@1: 0.67
  Hit Rate@3: 1.00
  MRR:        0.833
  平均相关度: 0.82
```

### 指标解读

- **Hit Rate@3 ≥ 0.8**：召回效果良好
- **Hit Rate@3 在 0.5~0.8**：建议调整 chunk_size 或添加查询改写
- **Hit Rate@3 < 0.5**：检查 Embedding 模型是否匹配文档语言
- **MRR ≥ 0.7**：排序质量良好
- **MRR 在 0.4~0.7**：建议增大 CrossEncoder 的 top_n

---

## 10. 主流评估框架对比

除了本项目自建的轻量评估器，业界有成熟的 RAG 评估框架：

### RAGAS（RAG Assessment）

- **定位**：专门为 RAG 设计的评估框架，业界事实标准
- **核心指标**：
  - **Faithfulness（忠实度）**：回答是否只基于检索到的文档，没有编造
  - **Answer Relevance（答案相关性）**：回答是否真正回应了用户问题
  - **Context Precision（上下文精确度）**：检索到的文档中有多少是相关的
  - **Context Recall（上下文召回率）**：所有相关文档中有多少被检索到了
- **特点**：需要参考答案（ground truth），评估更准确
- **安装**：`pip install ragas`

### DeepEval

- **定位**：pytest 风格的评估框架，适合 CI/CD 集成
- **核心指标**：Faithfulness、Answer Relevance、Contextual Precision/Recall 等
- **特点**：
  - 用 `@metric` 装饰器定义评估，像写测试一样写评估
  - 内置可视化仪表板
  - 支持对比不同版本的 RAG 系统
- **安装**：`pip install deepeval`

### TruLens

- **定位**：LLM 应用可观测性平台，不只是评估
- **核心概念**：RAG Triad（三重评估）
  - **Context Relevance**：检索到的上下文是否相关
  - **Groundedness**：回答是否有据可依
  - **Answer Relevance**：回答是否切题
- **特点**：
  - 提供丰富的可视化界面
  - 支持追踪每次检索和生成的完整过程
  - 适合生产环境的持续监控
- **安装**：`pip install trulens-eval`

### 框架选择建议

| 场景 | 推荐框架 |
|------|---------|
| 学习 / 原型验证 | 本项目自建评估器（轻量、好理解） |
| 学术研究 / 论文实验 | RAGAS（指标全面、业界认可） |
| CI/CD 自动化评估 | DeepEval（pytest 风格、易集成） |
| 生产环境持续监控 | TruLens（可视化、可观测性强） |

---

## 附：核心文件速查

| 文件 | 职责 |
|------|------|
| `config.py` | 全局配置（模型名、超参数、工厂函数） |
| `graph_state.py` | State 定义（IngestionState、RetrievalState） |
| `ingestion.py` | 文档解析、切片、入库 |
| `retrieval.py` | 查询改写、检索、重排序、评分、生成 |
| `graph_nodes.py` | LangGraph 节点函数封装 |
| `evaluation.py` | 检索质量评估器 |
| `utils.py` | 工具函数（打印、哈希、格式化等） |
| `main.py` | 可运行示例入口 |
