# 企业级 RAG 系统 (Enterprise RAG System)

> 将 phase1-3 全部知识整合为**一套生产级 RAG 系统**。
> 双图架构、多路召回、CrossEncoder 重排序、CRAG 自适应生成、完整评估体系。

---

## 为什么学这个

前面模块 13 讲了基础 RAG（单路向量检索），模块 14 讲了混合检索（BM25 + Vector）。但一个真正的企业级 RAG 系统远不止"检索 + 生成"这么简单：

| 真实问题 | 本模块的解决方案 |
|----------|------------------|
| 检索质量不够 → 漏文档、答非所问 | 查询改写 + 多路召回 + 重排序 |
| 检索到无关内容 → LLM 胡编 | 文档评分 + 自适应生成（CRAG） |
| 文档有表格/图片 → 丢失结构化信息 | 多模态解析 + 按元素类型分流 |
| 流程复杂难维护 → 函数调用层层嵌套 | LangGraph 图编排 + 双图架构 |
| 不知道系统好不好 → 凭感觉调参 | 评估器 + Hit Rate/MRR/Latency |

### 和已有概念的类比

| 已知概念 | 企业级 RAG 中的对应 |
|----------|---------------------|
| 基础 RAG（模块 13） | 增加了查询改写和重排序的完整管线 |
| 混合检索（模块 14） | 整合进 LangGraph 工作流，支持并行 fan-out |
| StateGraph（模块 15） | 编排入库和检索两条独立流水线 |
| 条件路由（模块 15） | 按文档类型分流（表格/文本）、按评分分叉（生成/兜底） |
| 多 Agent（模块 16） | 入库 Agent + 检索 Agent 各司其职 |
| Reducer（模块 15） | 并行节点的结果通过 `Annotated[list, operator.add]` 自动合并 |

可以把企业级 RAG 想象成**搜索引擎 + AI 问答**的组合：
- **搜索引擎**负责"找"（多路召回、重排序、评分）
- **AI 问答**负责"答"（基于检索结果生成回答）
- **LangGraph** 是整个流程的"调度中心"

---

## 快速开始

### 安装依赖

```bash
pip install langchain langchain-chroma langchain-classic langgraph
pip install langchain-huggingface langchain-community langchain-text-splitters
pip install sentence-transformers rank_bm25 pypdf chromadb fpdf2 torch

# 可选：高级 PDF 解析（表格/图片识别）
pip install "unstructured[pdf]"
```

### 运行

```bash
python test.py    # 验证组件安装（14 项检查，大部分不需要 API key）
python main.py    # 运行 7 个渐进式示例
```

---

## 项目架构总览

### 双图设计

整个系统由**两条独立的 LangGraph 流水线**组成，通过共享的 ChromaDB 向量库连接：

```
┌──────────────────────────────────────────────────────────────────┐
│                      共享存储层：ChromaDB                        │
│           ./chroma_store/  (持久化向量 + 元数据)                  │
└───────────────┬──────────────────────────┬───────────────────────┘
                │                          │
        ┌───────▼───────┐          ┌───────▼───────┐
        │    入库图      │          │    检索图      │
        │ IngestionState │          │ RetrievalState │
        └───────────────┘          └───────────────┘
```

- **入库图**负责"把文档存进去"——解析 → 切片 → Embedding → ChromaDB
- **检索图**负责"从文档里找答案"——改写 → 混合检索 → 重排序 → 评分 → 生成
- 两条图可以**独立运行、独立测试、独立扩展**

### 目录结构

```
03_enterprise_rag/
├── config.py         # 集中配置：模型、路径、超参数、工厂函数
├── graph_state.py    # 状态定义：IngestionState / RetrievalState
├── ingestion.py      # 入库引擎：解析、切片、Embedding、ChromaDB
├── retrieval.py      # 检索引擎：改写、多路检索、重排序、评分、生成
├── graph_nodes.py    # 节点封装：将业务函数包装为 LangGraph 节点
├── evaluation.py     # 🆕 评估器：Hit Rate、MRR、Latency
├── utils.py          # 工具：打印、哈希、样本 PDF 生成、UTF-8 修复
├── main.py           # 入口：7 个渐进式示例
├── test.py           # 验证：14 项组件检查
├── data/samples/     # 示例 PDF（程序自动生成）
└── chroma_store/     # ChromaDB 持久化目录（自动创建）
```

---

## 线路一：文档入库管线

### 总览

入库管线负责把原始 PDF 变为可检索的向量切片。

```
                    PDF 文件
                       │
                       ▼
              ┌─────────────────┐
              │   文档解析       │  ← 支持两种解析器
              │ parse_simple /   │
              │ parse_advanced   │
              └────────┬────────┘
                       │
            ┌──────────┴──────────┐
            │                     │
            ▼                     ▼
     ┌─────────────┐      ┌─────────────┐
     │  文本元素    │      │  表格元素    │  ← 按类型分流
     │ NarrativeText│      │   Table     │     (仅多模态模式)
     └──────┬──────┘      └──────┬──────┘
            │                    │
            ▼                    ▼
     ┌─────────────┐      ┌─────────────┐
     │  文本切片    │      │  保留原文    │  ← 表格不切片
     │ chunk_size=  │      │  (不切片)   │     避免破坏结构
     │   500        │      └──────┬──────┘
     └──────┬──────┘             │
            │                    │
            └─────────┬──────────┘
                      │  (reducer 合并)
                      ▼
              ┌─────────────────┐
              │  Embedding +    │
              │  ChromaDB 入库  │
              └────────┬────────┘
                       │
                       ▼
                   入库完成
```

### 路径 A：简单入库（build_ingestion_graph）

**适用场景**：纯文本 PDF，不需要识别表格/图片。

**节点序列**：`START → parse → chunk → store → END`

**每一步做了什么**：

#### parse 节点

- 使用 `PyPDFLoader` 逐页解析 PDF
- 每页生成一个 `Document`，metadata 含 `{source, page}`
- 结果存入 `raw_documents`

#### chunk 节点

- 使用 `RecursiveCharacterTextSplitter`，分隔符优先级：
  ```
  "\n\n"（段落） > "\n"（换行） > "。"（中文句号） > "！" > "？" > "."（英文句号） > " "（空格） > ""（字符）
  ```
- `chunk_size=500`，`chunk_overlap=50`
- 重叠 50 字符确保关键信息不会正好被切在边界上

#### store 节点

- 先调用 `HuggingFaceEmbeddings` 将每个 chunk 转为 384 维向量
- 生成唯一 ID：`{文件名}_p{页码}_{内容哈希8位}_{序号}`
- 通过 `ChromaDB.add_documents(documents, ids=ids)` 持久化写入

### 路径 B：多模态入库（build_multimodal_ingestion_graph）

**适用场景**：PDF 含表格、图表等结构化元素。

**节点序列**：

```
START → parse → route_by_content → [chunk_text + extract_tables]（并行）→ store → END
```

**和路径 A 的关键区别**：

| 方面 | 路径 A | 路径 B |
|------|--------|--------|
| 解析器 | PyPDFLoader | UnstructuredPDFLoader（可选 fallback） |
| 表格处理 | 和文本一起切片 | **表格保留原文**，不切片 |
| 分支策略 | 无分支，线性 | 条件路由：有表格 → 双分支并行 |
| 元素标注 | 无 | 每个 Document 带 `element_type` metadata |

#### 为什么要保留表格原文

RecursiveCharacterTextSplitter 会把表格切碎：

```
原始表格：
┌──────┬──────────┬──────────┬──────────┐
│Model │Params    │ Context  │ MMLU     │
├──────┼──────────┼──────────┼──────────┤
│GPT-4o│1.8T(est.)│ 128K     │ 88.7%    │
│Gemini│Unknown   │ 1M       │ 90.0%    │
└──────┴──────────┴──────────┴──────────┘

切片后（两个 chunk）：
Chunk 1: "┌──────┬──────────┬──────────┬──────────┐\n│Model │Params    │"
Chunk 2: "│Gemini│Unknown   │ 1M       │ 90.0%    │\n└──────┴──────────┘"
                                  ↑ 表头丢失，无法理解含义
```

所以多模态路径用 `separate_by_element_type()` 把表格元素挑出来，原样传给 store 节点。

#### 路由逻辑（route_by_content_type）

- 检测 `raw_documents` 中是否有 `metadata["element_type"] == "Table"`
- 有表格 → 返回 `["chunk_text", "extract_tables"]`（双分支并行）
- 无表格 → 返回 `["chunk"]`（单分支）
- 两个分支**并行执行**，汇入 store 时通过 reducer 自动合并

### 入库优化详解

#### 优化 1：基于内容哈希的防重复 ID

```python
content_hash = hashlib.md5(chunk.page_content.encode("utf-8")).hexdigest()[:8]
ids.append(f"{Path(source).stem}_p{page}_{content_hash}_{i}")
```

同一文档多次入库不会产生重复向量（ChromaDB 按 ID upsert 而非追加）。

#### 优化 2：自动 fallback

多模态解析依赖未安装时自动降级到 PyPDFLoader，不阻塞流程。

#### 优化 3：reset_collection 幂等操作

集合不存在时静默跳过，示例可重复运行。

---

## 线路二：检索生成管线

### 总览

检索管线是系统的核心，从接收问题到返回回答共 7 个阶段：

```
                    用户提问
                       │
                       ▼
              ┌─────────────────┐
              │  ① 查询改写      │  LLM 生成 2-3 个语义变体
              └────────┬────────┘
                       │
            ┌──────────┴──────────┐
            │                     │
            ▼                     ▼
     ┌─────────────┐      ┌─────────────┐
     │ ② 向量检索   │      │ ② BM25 检索 │  ← 并行 fan-out
     │ 语义匹配     │      │ 关键词匹配  │
     └──────┬──────┘      └──────┬──────┘
            │                    │
            └─────────┬──────────┘
                      │  fan-in（reducer 合并）
                      ▼
              ┌─────────────────┐
              │ ③ 合并去重      │  基于 MD5 哈希去重
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ ④ 重排序        │  CrossEncoder 逐对打分
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ ⑤ LLM 评分      │  充分性 0.0~1.0
              └────────┬────────┘
                       │
           score ≥ 0.7 │ score < 0.7
           ┌───────────┴───────────┐
           │                       │
           ▼                       ▼
   ┌──────────────┐       ┌──────────────┐
   │ ⑥ 生成回答   │       │ ⑥ 兜底回复   │
   │ 有据生成+引用│       │ 告知无相关信息│
   └──────────────┘       └──────────────┘
```

### 阶段 ①：查询改写（rewrite_query）

**目的**：同一个意图用多种表达方式，扩大召回覆盖面。

```
用户问："哪个模型 MMLU 分数最高？"

原始查询 → 只能命中包含"MMLU"+"分数"+"最高"的句子
改写1→ "AI 模型评估成绩排名"     → 命中"模型评估"相关段落
改写2→ "哪个 LLM 的测评结果最好" → 命中"测评结果"相关段落
```

三个查询一起上，覆盖面从 1 条扩展到 3 条的并集。

**实现**：LLM 生成 2 个变体 + 原始查询 = 3 条，始终包含原始查询作为保底。

### 阶段 ②：多路召回（vector_search + bm25_search）

两路并行执行，互补漏洞。

#### 向量检索

- 每个查询变体 → `ChromaDB.similarity_search(k=5)` → 语义最近的 5 个 chunk
- 所有结果按 page_content 哈希去重
- 擅长同义表达（"性能优化"≈"速度提升"），不擅长精确关键词

#### BM25 检索

- 基于倒排索引的词频匹配，不需要 Embedding
- 擅长精确关键词（"128K"精准命中），不擅长语义理解

#### 为什么两路都要跑

| 场景 | 向量检索 | BM25 | 结论 |
|------|---------|------|------|
| "Python 性能优化" | 命中"代码加速技巧" | 漏掉 | 向量✅ |
| "MMLU 88.7%" | 可能漏掉（数值被稀释） | 精准命中 | BM25 ✅ |
| "RAG 2024 改进" | 命中语义相关段落 | 命中含"2024"的段落 | 双路 ✅ |

#### 召回数量说明

`RETRIEVAL_K = 5` 是 chunk 数。入库时 PDF 被切成 chunk，检索返回的每个 `Document` 就是一个 chunk。metadata 中 `source` 和 `page` 可溯源到原始 PDF。

### 阶段 ③：合并去重（merge_and_deduplicate）

同一个 chunk 可能被多个查询变体或两条路径同时命中。

**去重策略**：基于 MD5 哈希，相同内容只保留首次出现的（向量结果优先排前，因为语义匹配通常更可靠）。

### 阶段 ④：CrossEncoder 重排序（rerank_documents）

**要解决什么问题**：合并后 5-10 个 chunk 排序不精确。

```
向量检索返回（按语义相似度）：
  1. "2024 AI Technology Report..."         ← 话题相关但信息量少
  2. "Llama 3.1 405B 128K 85.2%..."         ← 包含分数
  3. "RAG improvements in 2024..."           ← 不相关但语义相近
  4. "Gemini Ultra achieves 90.0%..."        ← 最相关却排第 4
```

#### Bi-Encoder vs CrossEncoder

**Bi-Encoder（向量检索方式）**：query 和 doc **分别**编码，算余弦距离。快但粗糙，因为 query 和 chunk 从没见过面。

**CrossEncoder（重排序方式）**：query 和 doc **拼接**后一起进 Transformer。Self-attention 让 query 的每个词和 doc 的每个词交叉交互，打分更准。

```
CrossEncoder 逐对打分：

Query: "哪个模型 MMLU 分数最高？"

第 1 对 → 0.92 ★★★ Gemini Ultra achieves 90.0%...
第 2 对 → 0.74 ★★  Llama 3.1 405B 128K 85.2%...
第 3 对 → 0.68 ★   The table above shows...
─── top_n=3 截断线 ───
第 4 对 → 0.31     2024 AI Technology Report...  ✗ 丢弃
第 5 对 → 0.08     RAG improvements...           ✗ 丢弃
```

#### 模型参数

```
模型：cross-encoder/ms-marco-MiniLM-L-6-v2
训练数据：MS MARCO（微软搜索问答数据集）
架构：MiniLM，6 层 Transformer
大小：约 80MB，首次运行自动下载到 ~/.cache/huggingface/
运行环境：纯 CPU，本地执行，不需要 API
速度：5 个 chunk 约 50ms
```

### 阶段 ⑤：文档评分 / CRAG（grade_documents）

**核心问题**：重排序后的 3 个 chunk 真的能回答用户问题吗？

- 问文档里有的 → 能回答 ✅
- 问文档里没有的（如"诺贝尔奖得主"）→ chunk 再接近也是凑数，硬答就是幻觉 ❌

**CRAG = Corrective RAG**：生成之前先检查，不盲目信任检索。

#### 评分过程

LLM 被要求输出 0.0-1.0 的数字，评估检索到的文档是否足以回答用户问题。

评分标准：
```
1.0  ████████████████████  完全能回答
0.7  ██████████████▌─────  基本能回答，可能缺细节  ← 阈值线
0.4  ████████────────────  有部分相关信息
0.0  ────────────────────  完全无关
```

### 阶段 ⑥：条件路由 + 回答生成

```
            score ≥ 0.7                      score < 0.7
                │                                  │
                ▼                                  ▼
        ┌──────────────┐                  ┌──────────────┐
        │ 生成回答      │                  │ 兜底回复      │
        │ - 只基于文档  │                  │ "知识库没有   │
        │ - 标注来源    │                  │  足够信息"    │
        │ - 不编造      │                  └──────────────┘
        └──────────────┘
```

- **生成**：grounded generation（有据生成），prompt 明确要求"只基于文档内容，标注来源"
- **兜底**：当前是 stub，生产环境应接入 Tavily/Bing 网络搜索

### 检索优化详解

#### 优化 1：BM25 索引缓存

```python
_BM25_CACHE: dict[tuple[int, int], object] = {}

def _get_or_build_bm25(documents, k):
    cache_key = (id(documents), len(documents))  # id()+len() 双重校验
    if cache_key not in _BM25_CACHE:
        _BM25_CACHE[cache_key] = BM25Retriever.from_documents(documents, k=k)
    return _BM25_CACHE[cache_key]
```

相同文档列表只构建一次倒排索引，后续调用从内存缓存直接取。

#### 优化 2：CrossEncoder 单例缓存

```python
@functools.lru_cache(maxsize=1)
def _get_reranker(model_name):
    return CrossEncoderReranker(model=HuggingFaceCrossEncoder(model_name=model_name))
```

首次加载 80MB 模型后，所有调用复用同一实例，消除重复磁盘 IO。

#### 优化 3：grade 鲁棒解析

```python
def _parse_score(raw: str) -> float:
    # 正则提取 0.0~1.0 范围的首个数字，而不是直接 float() 转换
    matches = re.findall(r'\b([01](?:\.\d+)?|\.\d+)\b', raw)
    for m in matches:
        val = float(m)
        if 0.0 <= val <= 1.0:
            return val
    return 0.5
```

容忍 LLM 输出 `"评分：0.85"`、`"0.9/1.0"` 等含额外文字的格式。

#### 优化 4：并行 fan-out

```python
builder.add_edge("rewrite", "vector_search")   # ─┐ fan-out
builder.add_edge("rewrite", "bm25_search")     # ─┘ 并行
```

LangGraph 自动检测多条出边 → 并发执行两路检索。

---

## 评估指标

> 🆕 `evaluation.py` 提供完整的离线评估能力。

### 为什么需要评估

调参（chunk_size、k、阈值、权重）时，需要**可量化的指标**来判断改完是变好还是变坏。

### 支持的指标

| 指标 | 公式 / 含义 | 好值 |
|------|------------|------|
| **Hit Rate@k** | 前 k 条结果中至少有 1 条相关的查询比例 | `@1 ≥ 0.6`，`@3 ≥ 0.8` |
| **MRR** | 第一个相关文档排名倒数取均值 | `≥ 0.7` |
| **Avg Relevance Score** | LLM 评分的平均值 | `≥ 0.7` |
| **Average Latency** | 端到端检索耗时（毫秒） | `< 3000ms` |

### 指标详解

#### Hit Rate@k

最直观的用户体验指标——"搜索结果里有没有我要的？"

```
对每条查询，看前 k 个检索结果中是否有至少一个"相关"文档。
Hit Rate@3 = (前 3 个结果有相关的查询数) / 总查询数

@1 衡量首页命中率，@3 衡量翻一页找到的概率。
```

#### MRR（Mean Reciprocal Rank）

衡量排序质量——相关文档排越前，MRR 越高。

```
RR = 1 / (第一个相关文档的排名)

示例：
  查询 1：第一个相关排第 1  → RR = 1/1 = 1.000
  查询 2：第一个相关排第 3  → RR = 1/3 = 0.333
  查询 3：没有相关         → RR = 0.000

MRR = (1.000 + 0.333 + 0.000) / 3 = 0.444
```

不同于 Hit Rate（只看"有没有"），MRR 还看"排哪里"。

#### LLM-as-Judge 相关性判断

逐条判断二元（相关/不相关），不需要人工标注测试集：

```python
def _judge_single_relevance(query, doc, llm):
    prompt = f"""判断以下文档片段是否包含「有助于回答用户问题」的信息。
用户问题：{query}
文档片段：{doc.page_content[:400]}
只回答"Yes"或"No。"""
```

### 评估器的使用

```python
from evaluation import RetrievalEvaluator

evaluator = RetrievalEvaluator(vectorstore, all_documents)

test_set = {
    "哪个模型 MMLU 分数最高？": 0.7,
    "RAG 在 2024 年有哪些改进？": 0.7,
}

metrics = evaluator.evaluate(test_set)
evaluator.print_report(metrics)
```

输出示例：

```
==================== 检索质量评估报告 ====================
  测试查询数: 3
  平均延迟:   1523 ms
  Hit Rate@1: 66.67%
  Hit Rate@3: 100.00%
  MRR:        0.833
  平均相关度: 0.78

---- 逐查询详情 ----
  [1] 哪个模型 MMLU 分数最高？
      改写 3 条 | 向量 8 | BM25 5 | 重排 3
      评分 0.85 | RR 1.000 | 1450 ms
  [2] RAG 在 2024 年有哪些改进？
      改写 3 条 | 向量 10 | BM25 7 | 重排 3
      评分 0.72 | RR 0.500 | 1670 ms

---- 指标解读 ----
  ✅ 召回效果良好：80%+ 的查询能找到相关文档
  ✅ 排序质量良好：相关文档排在前列
  ✅ 检索充分性达标：LLM 认可检索结果质量
```

### 持续评估建议

1. **建立稳定测试集**：选 10-20 条有代表性的查询
2. **每次改参数后跑评估**：对比改前改后的 Hit Rate 和 MRR
3. **同时测试"能答"和"不能答"两种查询**：好的系统应正确区分
4. **监控延迟**：优化不能损害用户体验

---

## 关键代码解析

### 工厂函数 + 闭包注入

`vectorstore` 和 `all_documents` 太大且不可序列化，不适合存入 State。用工厂函数闭包捕获：

```python
def make_vector_search_node(vectorstore):
    """闭包捕获 vectorstore，避免存入 State"""
    def _node(state: RetrievalState) -> dict:
        results = vector_search(..., vectorstore)
        return {"vector_results": results}
    return _node
```

### Reducer 实现并行合并

```python
class IngestionState(TypedDict):
    text_chunks: Annotated[list[Document], operator.add]
    #                                       ↑ reducer
    # 并行节点分别返回 text_chunks 时，operator.add 拼接而非覆盖
```

### 条件路由

```python
def route_by_score(state):
    return "fallback" if state["needs_fallback"] else "generate"

builder.add_conditional_edges("grade", route_by_score)
```

---

## 调参指南

### 切片参数

| 场景 | chunk_size | chunk_overlap | 理由 |
|------|-----------|---------------|------|
| FAQ / 短文本 | 200-300 | 20-30 | 小粒度更精确 |
| 技术文档 | 500-800 | 50-100 | 平衡完整性和精度 |
| 长报告 / 论文 | 800-1200 | 100-200 | 需要更多上下文 |

### 检索权重

| 场景 | BM25 | Vector | 理由 |
|------|------|--------|------|
| 精确查找（数值、代码） | 0.6-0.7 | 0.3-0.4 | 关键词更重要 |
| 概念问答 | 0.3-0.4 | 0.6-0.7 | 语义更重要 |
| 混合 | 0.4-0.5 | 0.5-0.6 | 均衡 |

### 评分阈值

| RELEVANCE_THRESHOLD | 效果 |
|---------------------|------|
| 0.5 | 更宽松，更多走生成（可能低质量） |
| 0.7 | **推荐值** |
| 0.85 | 更严格，更多被拒绝（但回答质量更高） |

---

## 常见问题

### 1. `ModuleNotFoundError: No module named 'langchain_chroma'`

```bash
pip install langchain-chroma
```

### 2. CrossEncoder 首次运行卡住

正常现象——首次下载约 80MB 模型。之后 lru_cache 复用，再不会卡。

### 3. ChromaDB 数据重复

`reset_collection("集合名")` 清空后重新入库。

### 4. UnstructuredPDFLoader 安装失败

代码自动 fallback 到 PyPDFLoader，不影响核心流程。

### 5. Windows 中文乱码

`utils.py` 导入时自动执行 UTF-8 修复。

### 6. `similarity_search` 返回空

确认已运行入库、`collection_name` 一致、`chroma_store/` 目录存在。

### 7. 评估时延迟波动大

首次评估需加载模型，后续评估模型已缓存。v2 优化后模型加载不再是瓶颈。

---

## 最佳实践

### 企业级 RAG 检查清单

- [ ] Embedding 模型是否支持目标语言
- [ ] 切片大小是否适合文档类型
- [ ] 是否使用混合检索（BM25 + Vector）
- [ ] 是否有 CrossEncoder 重排序
- [ ] 是否有文档充分性评估（CRAG）
- [ ] 是否有 fallback 机制
- [ ] 是否有来源引用
- [ ] 是否处理了多模态内容（表格/图片）
- [ ] 是否有评估指标监控

### v2 版本优化总览

| 优化项 | 文件 | 手段 | 效果 |
|--------|------|------|------|
| CrossEncoder 缓存 | retrieval.py | `lru_cache` | 首次加载后零磁盘IO |
| BM25 索引缓存 | retrieval.py | 模块级 dict | 相同文档集不重建索引 |
| grade 鲁棒解析 | retrieval.py | 正则提取 | 容忍 LLM 输出额外文字 |
| 多模态图正常调用 | main.py | `graph.invoke()` | example_3 完整演示 |
| 评估工具 | evaluation.py | 新模块 | 可量化衡量检索质量 |

---

## 常用 API 速查

```python
# ── 文档解析 ──
from langchain_community.document_loaders import PyPDFLoader
pages = PyPDFLoader("file.pdf").load()

# ── 文本切片 ──
from langchain_text_splitters import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(pages)

# ── Embedding ──
from langchain_huggingface import HuggingFaceEmbeddings
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ── 向量库 ──
from langchain_chroma import Chroma
vs = Chroma(collection_name="docs", embedding_function=emb, persist_directory="./db")
vs.add_documents(chunks)
results = vs.similarity_search("query", k=5)

# ── BM25 ──
from langchain_community.retrievers import BM25Retriever
bm25 = BM25Retriever.from_documents(chunks, k=5)

# ── 混合检索 ──
from langchain_classic.retrievers import EnsembleRetriever
ensemble = EnsembleRetriever(retrievers=[bm25, vs.as_retriever(k=5)], weights=[0.4, 0.6])

# ── 重排序 ──
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
reranker = CrossEncoderReranker(
    model=HuggingFaceCrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2"),
    top_n=3
)

# ── LangGraph ──
from langgraph.graph import StateGraph, START, END
builder = StateGraph(MyState)
builder.add_node("name", node_func)
builder.add_edge(START, "name")
graph = builder.compile()
result = graph.invoke({"field": "value"})

# ── 评估器 (新) ──
from evaluation import RetrievalEvaluator
evaluator = RetrievalEvaluator(vectorstore, all_docs)
metrics = evaluator.evaluate({"查询": 0.7})
evaluator.print_report(metrics)
```

---

## 进一步学习

- **向量模型选型**：[MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- **生产部署**：ChromaDB → Milvus / Weaviate（分布式），加缓存层和 API 层
- **高级 RAG**：多跳检索、自适应检索、查询融合（Query Fusion）、HyDE（假设文档嵌入）
- **可观测性**：接入 LangSmith / LangFuse 追踪完整链路
- **多模态 RAG**：图像理解、OCR、视觉语言模型结合
- **评估体系**：扩展 evaluation.py，加入 ROUGE、BLEU、忠实度（Faithfulness）评估
