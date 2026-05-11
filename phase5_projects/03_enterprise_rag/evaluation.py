"""
企业级 RAG 系统 - 检索质量评估模块
=====================================

提供离线评估能力，帮助在无人工标注的情况下
衡量检索管线的质量。

支持指标：
- Hit Rate@k    : 前 k 个结果中命中正确答案的比例
- MRR           : Mean Reciprocal Rank，第一个正确答案的排名倒数均值
- Recall@k      : 前 k 个结果中召回的相关文档比例
- Latency       : 平均检索耗时（ms）

设计说明：
- 不需要人工标注：用 LLM 自评估（LLM-as-Judge）判断
  检索到的文档是否包含答案所需信息
- 轻量级：不依赖 ragas、deepeval 等第三方评估框架
- 可扩展：评估器通过组合方式，可灵活添加新指标
"""

import time
from langchain_core.documents import Document
from langchain_chroma import Chroma

from retrieval import (
    rewrite_query,
    vector_search,
    bm25_search,
    merge_and_deduplicate,
    rerank_documents,
    grade_documents,
)
from config import RETRIEVAL_K, RERANK_TOP_N, get_llm


# ============================================================================
# 检索评估器
# ============================================================================


class RetrievalEvaluator:
    """
    RAG 检索管线评估器

    对一组测试查询执行完整检索，收集各项指标，
    用于对比不同配置（Embedding 模型、k 值、权重等）
    对检索质量的影响。

    用法:
        evaluator = RetrievalEvaluator(vectorstore, all_docs)
        # test_set = {"查询文本": 期望相关度阈值, ...}
        metrics = evaluator.evaluate(test_set)
        evaluator.print_report(metrics)
    """

    def __init__(
        self,
        vectorstore: Chroma,
        all_documents: list[Document],
        k: int = RETRIEVAL_K,
        top_n: int = RERANK_TOP_N,
    ):
        """
        参数:
            vectorstore: ChromaDB 向量库实例
            all_documents: 用于构建 BM25 索引的完整文档集
            k: 每路检索召回数
            top_n: 重排序保留数
        """
        self.vectorstore = vectorstore
        self.all_documents = all_documents
        self.llm = get_llm()
        self.k = k
        self.top_n = top_n

    def evaluate(self, test_queries: dict[str, float]) -> dict:
        """
        对测试集执行完整检索管线并计算指标

        参数:
            test_queries: {查询文本: 期望相关度阈值}
                阈值含义：这些查询"应该能回答"，给 0.0 表示"不应该能回答"
                通常设为 0.7（config 的 RELEVANCE_THRESHOLD）

        返回:
            指标字典:
            {
                "num_queries": int,             # 测试查询数
                "hit_rate@1": float,            # 第一个结果相关的比例
                "hit_rate@3": float,            # 前 3 个中有相关的比例
                "mrr": float,                   # Mean Reciprocal Rank
                "avg_relevance_score": float,    # 平均 LLM 相关度评分
                "avg_latency_ms": float,         # 平均检索耗时（毫秒）
                "per_query": list[dict],         # 每查询详情
            }
        """
        per_query = []
        total_latency_ms = 0.0
        sum_hit_at_1 = 0
        sum_hit_at_3 = 0
        sum_rr = 0.0       # Reciprocal Rank
        sum_relevance = 0.0
        n = len(test_queries)

        for query, _expected in test_queries.items():
            detail = self._eval_one(query)
            per_query.append(detail)

            total_latency_ms += detail["latency_ms"]
            sum_relevance += detail["relevance_score"]

            # Hit Rate@k——前 k 条中是否有相关文档
            if detail["hit_at_1"]:
                sum_hit_at_1 += 1
            if detail["hit_at_3"]:
                sum_hit_at_3 += 1

            # MRR——第一个相关文档出现的位置
            sum_rr += detail["reciprocal_rank"]

        return {
            "num_queries": n,
            "hit_rate@1": sum_hit_at_1 / n if n > 0 else 0.0,
            "hit_rate@3": sum_hit_at_3 / n if n > 0 else 0.0,
            "mrr": sum_rr / n if n > 0 else 0.0,
            "avg_relevance_score": sum_relevance / n if n > 0 else 0.0,
            "avg_latency_ms": total_latency_ms / n if n > 0 else 0.0,
            "per_query": per_query,
        }

    def _eval_one(self, query: str) -> dict:
        """对单条查询执行检索+评估"""
        t0 = time.perf_counter()

        # 完整检索管线
        queries = rewrite_query(query, self.llm)
        vec_results = vector_search(queries, self.vectorstore, k=self.k)
        bm25_results = bm25_search(queries, self.all_documents, k=self.k)
        merged = merge_and_deduplicate(vec_results, bm25_results)

        if merged:
            reranked = rerank_documents(query, merged, top_n=self.top_n)
        else:
            reranked = []

        # LLM 整体充分性评分（调用 grade_documents 已有功能）
        relevance_score = grade_documents(query, reranked, self.llm)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        # 逐条判断相关性（hit: 相关度 >= 0.5 视为相关）
        hit_at_1 = False
        hit_at_3 = False
        reciprocal_rank = 0.0

        for rank, doc in enumerate(reranked, start=1):
            is_relevant = _judge_single_relevance(query, doc, self.llm)
            if is_relevant:
                if rank == 1:
                    hit_at_1 = True
                if rank <= 3:
                    hit_at_3 = True
                if reciprocal_rank == 0.0:
                    reciprocal_rank = 1.0 / rank
                break  # 找到了第一个相关文档就停（RR 只需要第一个）

        return {
            "query": query,
            "num_rewritten": len(queries),
            "num_vector": len(vec_results),
            "num_bm25": len(bm25_results),
            "num_merged": len(merged),
            "num_reranked": len(reranked),
            "relevance_score": relevance_score,
            "reciprocal_rank": reciprocal_rank,
            "hit_at_1": hit_at_1,
            "hit_at_3": hit_at_3,
            "latency_ms": elapsed_ms,
        }

    def print_report(self, metrics: dict):
        """打印评估报告"""
        from utils import print_title, print_section, print_points, print_warning

        print_title("检索质量评估报告")

        print(f"  测试查询数: {metrics['num_queries']}")
        print(f"  平均延迟:   {metrics['avg_latency_ms']:.0f} ms")
        print(f"  Hit Rate@1: {metrics['hit_rate@1']:.2%}")
        print(f"  Hit Rate@3: {metrics['hit_rate@3']:.2%}")
        print(f"  MRR:        {metrics['mrr']:.3f}")
        print(f"  平均相关度: {metrics['avg_relevance_score']:.2f}")

        # 分查询详情
        print_section("逐查询详情")
        for i, q in enumerate(metrics["per_query"], start=1):
            print(f"  [{i}] {q['query'][:50]}")
            print(f"      改写 {q['num_rewritten']} 条 | "
                  f"向量 {q['num_vector']} | BM25 {q['num_bm25']} | "
                  f"重排 {q['num_reranked']}")
            print(f"      评分 {q['relevance_score']:.2f} | "
                  f"RR {q['reciprocal_rank']:.3f} | "
                  f"{q['latency_ms']:.0f} ms")

        # 解读
        print_section("指标解读")
        if metrics["hit_rate@3"] >= 0.8:
            print("  ✅ 召回效果良好：80%+ 的查询能找到相关文档")
        elif metrics["hit_rate@3"] >= 0.5:
            print("  ⚠️  召回效果一般：建议调整 chunk_size 或添加查询改写")
        else:
            print("  ❌ 召回效果差：检查 Embedding 模型是否匹配文档语言")

        if metrics["mrr"] >= 0.7:
            print("  ✅ 排序质量良好：相关文档排在前列")
        elif metrics["mrr"] >= 0.4:
            print("  ⚠️  排序质量一般：建议增大 CrossEncoder 的 top_n")

        if metrics["avg_relevance_score"] >= 0.7:
            print("  ✅ 检索充分性达标：LLM 认可检索结果质量")
        elif metrics["avg_relevance_score"] >= 0.4:
            print("  ⚠️  检索充分性偏低：部分查询可能触发 fallback")
        else:
            print("  ❌ 检索充分性差：知识库可能缺少这些文档的相关内容")


# ============================================================================
# 辅助函数
# ============================================================================


def _judge_single_relevance(
    query: str,
    doc: Document,
    llm=None,
) -> bool:
    """
    判断单个文档是否与查询相关

    用 LLM 做二元判断（相关/不相关），比泛泛的相似度分数更精确。

    参数:
        query: 用户查询
        doc: 单个文档
        llm: LLM 实例

    返回:
        True 表示文档有助于回答该查询
    """
    if llm is None:
        llm = get_llm()

    prompt = f"""判断以下文档片段是否包含「有助于回答用户问题」的信息。

用户问题：{query}

文档片段：{doc.page_content[:400]}

只回答"Yes"或"No"。Yes = 包含有帮助的信息，No = 不相关或信息量不足。"""

    response = llm.invoke(prompt)
    raw = response.content.strip().lower()
    return raw.startswith("yes") or raw == "是"


# ============================================================================
# 端到端评估示例
# ============================================================================


def run_evaluation_demo(vectorstore: Chroma, all_documents: list[Document]):
    """
    评估器使用演示

    设计两组测试查询，分别评估检索质量，
    展示评估器的完整用法。

    参数:
        vectorstore: ChromaDB 实例
        all_documents: BM25 文档集
    """
    from utils import print_title, print_section, print_tip

    print_title("评估器演示")

    evaluator = RetrievalEvaluator(vectorstore, all_documents)

    # 测试集 1：知识库能回答的问题
    test_set_good = {
        "哪个模型 MMLU 分数最高？": 0.7,
        "RAG 在 2024 年有哪些改进？": 0.7,
        "LangChain 1.0 引入了哪些变化？": 0.7,
    }

    print_section("测试集 1：知识库能回答的查询")
    metrics_good = evaluator.evaluate(test_set_good)
    evaluator.print_report(metrics_good)

    # 测试集 2：知识库无法回答的问题
    test_set_bad = {
        "2024 年诺贝尔物理学奖得主是谁？": 0.3,
        "今天天气怎么样？": 0.3,
    }

    print_section("测试集 2：知识库无法回答的查询")
    metrics_bad = evaluator.evaluate(test_set_bad)
    evaluator.print_report(metrics_bad)

    print_tip(
        "对比两组指标：good 组要求高评分+高HitRate，"
        "bad 组期望低评分+低HitRate（正确拒绝无关查询）"
    )
