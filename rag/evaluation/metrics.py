# ============================================================================
# 模块：rag/evaluation/metrics.py
# RAG 评估指标：检索指标 + 生成指标
# ============================================================================
# 【评估指标一览】
# ┌─────────────────────┬──────────────────────────────────────────────────┐
# │ 指标                │ 说明                                              │
# ├─────────────────────┼──────────────────────────────────────────────────┤
# │ 检索指标（无需LLM）  │                                                   │
# │ HitRate@K           │ Top-K 结果中是否至少命中一条相关文档              │
# │ Recall@K            │ Top-K 结果中命中的相关文档 / 总相关文档数         │
# │ Precision@K         │ Top-K 结果中相关文档数 / K                        │
# │ MRR                 │ 第一个相关文档排名的倒数均值，衡量"第一个对的在哪" │
# │ NDCG@K              │ 归一化折损累计增益，考虑排名和相关性分数           │
# ├─────────────────────┼──────────────────────────────────────────────────┤
# │ 生成指标（LLM-as-Judge）                                                │
# │ Faithfulness        │ 回答是否忠实于检索到的上下文（不编造）             │
# │ AnswerRelevance     │ 回答是否与用户问题相关                            │
# │ ContextRelevance    │ 检索到的上下文是否与问题相关                       │
# └─────────────────────┴──────────────────────────────────────────────────┘
# ============================================================================

import math
from typing import List, Dict, Any, Set


class RetrievalMetrics:
    """
    检索质量评估指标，不需要 LLM，只需要标注数据。

    【标注数据格式】
        test_case = {
            "question": "兴业银行2024年营收多少？",
            "relevant_docs": ["兴业银行_2024年年度报告.pdf"],  # 相关文档ID列表
        }

    【使用方式】
        results = store.search("兴业银行2024年营收多少？", top_k=5)
        doc_ids = [r['file_name'] for r in results]
        hit = RetrievalMetrics.hit_rate(doc_ids, ["兴业银行_2024年年度报告.pdf"], k=5)
    """

    @staticmethod
    def hit_rate(retrieved_docs: List[str], relevant_docs: List[str], k: int = None) -> float:
        """
        HitRate@K：Top-K 结果中是否至少命中一条相关文档。

        【参数】
            retrieved_docs: List[str] — 检索返回的文档ID列表（按排名排序）
            relevant_docs: List[str] — 标注的相关文档ID列表
            k: int | None — 截取前 K 个结果，None 表示全部

        【返回值】
            float — 0.0 到 1.0，越高越好

        【示例】
            >>> RetrievalMetrics.hit_rate(['A','B','C'], ['C','D'], k=3)
            1.0
            >>> RetrievalMetrics.hit_rate(['A','B','C'], ['D'], k=2)
            0.0
        """
        docs = retrieved_docs[:k] if k else retrieved_docs
        relevant_set = set(relevant_docs)
        return 1.0 if any(d in relevant_set for d in docs) else 0.0

    @staticmethod
    def recall(retrieved_docs: List[str], relevant_docs: List[str], k: int = None) -> float:
        """
        Recall@K：Top-K 结果中命中的相关文档数 / 总相关文档数。

        【参数】
            retrieved_docs: List[str] — 检索返回的文档ID列表
            relevant_docs: List[str] — 标注的相关文档ID列表
            k: int | None — 截取前 K 个结果

        【返回值】
            float — 0.0 到 1.0，越高越好

        【示例】
            >>> RetrievalMetrics.recall(['A','B','C'], ['A','D'], k=3)
            0.5  # 2个相关文档中命中了1个
        """
        if not relevant_docs:
            return 0.0
        docs = retrieved_docs[:k] if k else retrieved_docs
        relevant_set = set(relevant_docs)
        hit_count = sum(1 for d in docs if d in relevant_set)
        return hit_count / len(relevant_docs)

    @staticmethod
    def precision(retrieved_docs: List[str], relevant_docs: List[str], k: int = None) -> float:
        """
        Precision@K：Top-K 结果中相关文档数 / K。

        【参数】
            retrieved_docs: List[str] — 检索返回的文档ID列表
            relevant_docs: List[str] — 标注的相关文档ID列表
            k: int | None — 截取前 K 个结果

        【返回值】
            float — 0.0 到 1.0，越高越好
        """
        docs = retrieved_docs[:k] if k else retrieved_docs
        if not docs:
            return 0.0
        relevant_set = set(relevant_docs)
        hit_count = sum(1 for d in docs if d in relevant_set)
        return hit_count / len(docs)

    @staticmethod
    def mrr(retrieved_docs: List[str], relevant_docs: List[str]) -> float:
        """
        MRR（Mean Reciprocal Rank）：第一个相关文档排名的倒数。

        【公式】
            MRR = 1 / rank_of_first_relevant_doc
            如果 Top-K 中没有相关文档，MRR = 0

        【参数】
            retrieved_docs: List[str] — 检索返回的文档ID列表
            relevant_docs: List[str] — 标注的相关文档ID列表

        【返回值】
            float — 0.0 到 1.0，越高越好（1.0 表示第一个就命中）

        【示例】
            >>> RetrievalMetrics.mrr(['A','B','C'], ['C','D'])
            0.3333  # 第3个命中，1/3
            >>> RetrievalMetrics.mrr(['A','B','C'], ['E'])
            0.0     # 没命中
        """
        relevant_set = set(relevant_docs)
        for i, doc in enumerate(retrieved_docs):
            if doc in relevant_set:
                return 1.0 / (i + 1)
        return 0.0

    @staticmethod
    def ndcg(retrieved_docs: List[str], relevant_docs: Dict[str, float], k: int = None) -> float:
        """
        NDCG@K（Normalized Discounted Cumulative Gain）：归一化折损累计增益。

        【公式】
            DCG@K = Σ(rel_i / log2(i+2))  for i in 0..K-1
            IDCG@K = 理想排序下的 DCG（所有相关文档排在最前面）
            NDCG@K = DCG@K / IDCG@K

        【参数】
            retrieved_docs: List[str] — 检索返回的文档ID列表
            relevant_docs: Dict[str, float] — 文档ID到相关性分数的映射
            k: int | None — 截取前 K 个结果

        【返回值】
            float — 0.0 到 1.0，越高越好

        【示例】
            >>> RetrievalMetrics.ndcg(['A','B','C'], {'A':3,'B':2,'C':1}, k=3)
            1.0  # 完美排序
        """
        docs = retrieved_docs[:k] if k else retrieved_docs
        if not docs:
            return 0.0

        dcg = 0.0
        for i, doc in enumerate(docs):
            rel = relevant_docs.get(doc, 0)
            if rel > 0:
                dcg += rel / math.log2(i + 2)

        ideal_rels = sorted(relevant_docs.values(), reverse=True)
        ideal_rels = ideal_rels[:len(docs)]
        idcg = 0.0
        for i, rel in enumerate(ideal_rels):
            if rel > 0:
                idcg += rel / math.log2(i + 2)

        if idcg == 0:
            return 0.0
        return dcg / idcg

    @staticmethod
    def f1(retrieved_docs: List[str], relevant_docs: List[str], k: int = None) -> float:
        """
        F1@K：精确率和召回率的调和平均。

        【公式】
            F1 = 2 * (Precision * Recall) / (Precision + Recall)

        【参数】
            retrieved_docs: List[str] — 检索返回的文档ID列表
            relevant_docs: List[str] — 标注的相关文档ID列表
            k: int | None — 截取前 K 个结果

        【返回值】
            float — 0.0 到 1.0，越高越好
        """
        p = RetrievalMetrics.precision(retrieved_docs, relevant_docs, k)
        r = RetrievalMetrics.recall(retrieved_docs, relevant_docs, k)
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)


class GenerationMetrics:
    """
    生成质量评估指标，使用 LLM-as-Judge 方式评估。

    【LLM-as-Judge 是什么？】
    用一个更强大的 LLM（或同一 LLM）来评判 RAG 生成的回答质量。
    通过对回答打分（1-5分）来量化忠实度、相关性等指标。
    比人工评估快，比规则评估准，是目前 RAG 评估的主流方法。

    【使用方式】
        judge_llm = LLMService()
        score = GenerationMetrics.faithfulness(answer, contexts, judge_llm)
    """

    # 忠实度评估 Prompt
    FAITHFULNESS_PROMPT = """你是一个严格的评估专家。请根据以下标准，对 AI 回答的"忠实度"进行评分。

【评估标准】
检查 AI 回答中的每一条陈述，是否都能在参考资料中找到明确依据。
- 5分：所有陈述都能在参考资料中找到直接依据，没有任何编造
- 4分：绝大多数陈述有依据，只有极少数合理推断
- 3分：大部分有依据，但有一些不能确认的陈述
- 2分：有较多无法在资料中找到依据的陈述
- 1分：大部分内容与参考资料无关或编造

【参考资料】
{context}

【AI 回答】
{answer}

请给出评分（1-5的整数）和简要理由。
输出格式：评分: X分\n理由: xxx"""

    # 回答相关性评估 Prompt
    ANSWER_RELEVANCE_PROMPT = """你是一个严格的评估专家。请评估 AI 回答与用户问题的"相关性"。

【评估标准】
- 5分：完全针对问题回答，直接、准确、全面
- 4分：基本回答了问题，但略有冗余或不够全面
- 3分：部分回答了问题，但有一些偏离或遗漏
- 2分：只有少部分内容与问题相关
- 1分：回答与问题几乎无关

【用户问题】
{question}

【AI 回答】
{answer}

请给出评分（1-5的整数）和简要理由。
输出格式：评分: X分\n理由: xxx"""

    # 上下文相关性评估 Prompt
    CONTEXT_RELEVANCE_PROMPT = """你是一个严格的评估专家。请评估检索到的参考资料与用户问题的"相关性"。

【评估标准】
- 5分：参考资料完全针对问题，包含了回答所需的所有关键信息
- 4分：参考资料大部分相关，包含了主要信息但略有不足
- 3分：参考资料部分相关，但缺少一些关键信息
- 2分：参考资料只有少量与问题相关
- 1分：参考资料与问题几乎无关

【用户问题】
{question}

【参考资料】
{context}

请给出评分（1-5的整数）和简要理由。
输出格式：评分: X分\n理由: xxx"""

    @staticmethod
    def _parse_score(text: str) -> int:
        """从 LLM 输出中解析评分"""
        import re
        match = re.search(r'评分[：:]\s*(\d)', text)
        if match:
            return int(match.group(1))
        match = re.search(r'(\d)\s*分', text)
        if match:
            return int(match.group(1))
        return 3

    @staticmethod
    def faithfulness(answer: str, contexts: List[str], judge_llm) -> Dict[str, Any]:
        """
        评估回答的忠实度：回答是否忠实于检索到的上下文。

        【参数】
            answer: str — LLM 生成的回答
            contexts: List[str] — 检索到的上下文列表
            judge_llm: LLMService — 用于评判的 LLM 服务

        【返回值】
            Dict: {"score": int, "reason": str}
        """
        context = '\n\n---\n\n'.join(contexts[:5])
        prompt = GenerationMetrics.FAITHFULNESS_PROMPT.format(
            context=context[:4000],
            answer=answer[:2000],
        )
        try:
            response = judge_llm.generate(
                [{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=256,
            )
            score = GenerationMetrics._parse_score(response)
            return {"score": score, "reason": response.strip()}
        except Exception as e:
            return {"score": 3, "reason": f"评估失败: {e}"}

    @staticmethod
    def answer_relevance(question: str, answer: str, judge_llm) -> Dict[str, Any]:
        """
        评估回答相关性：回答是否与用户问题相关。

        【参数】
            question: str — 用户问题
            answer: str — LLM 生成的回答
            judge_llm: LLMService — 用于评判的 LLM 服务

        【返回值】
            Dict: {"score": int, "reason": str}
        """
        prompt = GenerationMetrics.ANSWER_RELEVANCE_PROMPT.format(
            question=question,
            answer=answer[:2000],
        )
        try:
            response = judge_llm.generate(
                [{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=256,
            )
            score = GenerationMetrics._parse_score(response)
            return {"score": score, "reason": response.strip()}
        except Exception as e:
            return {"score": 3, "reason": f"评估失败: {e}"}

    @staticmethod
    def context_relevance(question: str, contexts: List[str], judge_llm) -> Dict[str, Any]:
        """
        评估上下文相关性：检索到的上下文是否与问题相关。

        【参数】
            question: str — 用户问题
            contexts: List[str] — 检索到的上下文列表
            judge_llm: LLMService — 用于评判的 LLM 服务

        【返回值】
            Dict: {"score": int, "reason": str}
        """
        context = '\n\n---\n\n'.join(contexts[:5])
        prompt = GenerationMetrics.CONTEXT_RELEVANCE_PROMPT.format(
            question=question,
            context=context[:4000],
        )
        try:
            response = judge_llm.generate(
                [{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=256,
            )
            score = GenerationMetrics._parse_score(response)
            return {"score": score, "reason": response.strip()}
        except Exception as e:
            return {"score": 3, "reason": f"评估失败: {e}"}