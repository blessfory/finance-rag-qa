# ============================================================================
# 模块：rag/evaluation/evaluator.py
# RAG 评估器：编排检索评估 + 生成评估，输出评估报告
# ============================================================================
# 【评估流程】
#   1. 加载测试数据集（问题 + 标注相关文档 + 期望回答）
#   2. 对每个问题：检索 → 计算检索指标
#   3. 对每个问题：生成回答 → LLM-as-Judge 评估生成质量
#   4. 汇总所有指标，输出评估报告
#
# 【使用方式】
#   from rag.evaluation import RAGEvaluator
#   evaluator = RAGEvaluator()
#   report = evaluator.evaluate(test_data)
#   print(report)
#   # 或保存到文件
#   evaluator.evaluate_and_save(test_data, "evaluation_report.txt")
# ============================================================================

import os
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

from base.config import Config
from base.logger import setup_logger
from rag.evaluation.metrics import RetrievalMetrics, GenerationMetrics
from rag.evaluation.prompts import REPORT_TEMPLATE

cfg = Config()
logger = setup_logger('RAG.evaluation', log_file=cfg.log_file)


class RAGEvaluator:
    """
    RAG 系统评估器：检索评估 + 生成评估 + 综合报告。

    【使用方式】
        # 方式一：代码调用
        evaluator = RAGEvaluator()
        test_data = [
            {
                "question": "兴业银行2024年营收和净利润分别是多少？",
                "relevant_docs": ["兴业银行_2024年年度报告.pdf"],
                "expected_answer": "营收约XX亿元，净利润约XX亿元",
            },
        ]
        report = evaluator.evaluate(test_data)

        # 方式二：从 CSV 加载测试数据
        test_data = RAGEvaluator.load_test_data("test_questions.csv")
        evaluator.evaluate_and_save(test_data, "report.txt")

        # 方式三：仅评估检索（不需要 LLM，更快）
        scores = evaluator.evaluate_retrieval_only(test_data)
    """

    def __init__(self, rag_service=None, judge_llm=None):
        """
        【参数】
            rag_service: RAGService | None — RAG 服务实例，不传则自动创建
            judge_llm: LLMService | None — 评估用的 LLM，不传则复用 RAG 的 LLM
        """
        self._rag = rag_service
        self._judge_llm = judge_llm
        self._rag_created = False
        self._judge_created = False

    @property
    def rag(self):
        if self._rag is None:
            from rag.rag_service import RAGService
            self._rag = RAGService()
            self._rag_created = True
            logger.info("RAG 服务已自动创建")
        return self._rag

    @property
    def judge_llm(self):
        if self._judge_llm is None:
            from rag.core.llm import LLMService
            self._judge_llm = LLMService()
            self._judge_created = True
            logger.info("评估 LLM 已自动创建")
        return self._judge_llm

    def close(self):
        """释放资源"""
        if self._rag_created and self._rag:
            self._rag.close()
        if self._judge_created and self._judge_llm:
            pass

    # ========================================================================
    # 数据加载
    # ========================================================================

    @staticmethod
    def load_test_data(csv_path: str) -> List[Dict[str, Any]]:
        """
        从 CSV 文件加载测试数据。

        【CSV 格式】
            必须包含列：question
            可选列：relevant_docs（多个用 | 分隔）、expected_answer

        【示例 CSV】
            question,relevant_docs,expected_answer
            兴业银行2024年营收多少？,兴业银行_2024年年度报告.pdf,约XX亿元
            联化科技主营业务？,联化科技_2024年年度报告.pdf,农药和医药中间体
        """
        import csv
        test_data = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                case = {"question": row.get("question", "").strip()}
                if "relevant_docs" in row and row["relevant_docs"].strip():
                    case["relevant_docs"] = [
                        d.strip() for d in row["relevant_docs"].split("|")
                    ]
                if "expected_answer" in row and row["expected_answer"].strip():
                    case["expected_answer"] = row["expected_answer"].strip()
                if case["question"]:
                    test_data.append(case)
        logger.info("从 %s 加载了 %d 条测试数据", csv_path, len(test_data))
        return test_data

    @staticmethod
    def generate_default_test_data() -> List[Dict[str, Any]]:
        """
        生成默认测试数据集（基于项目中的金融文档）。

        【返回值】
            List[Dict] — 预设的金融领域测试问题
        """
        return [
            {
                "question": "兴业银行2024年营收和净利润分别是多少？",
                "relevant_docs": ["兴业银行_2024年年度报告.pdf"],
                "expected_answer": "营收和净利润的具体数值",
            },
            {
                "question": "兴业银行2025年的经营情况如何？",
                "relevant_docs": ["兴业银行_2025年年度报告.pdf"],
                "expected_answer": "2025年经营情况概述",
            },
            {
                "question": "联化科技的主营业务是什么？",
                "relevant_docs": ["联化科技_2024年年度报告.pdf"],
                "expected_answer": "农药和医药中间体",
            },
            {
                "question": "兴业银行有哪些主要产品？",
                "relevant_docs": ["兴业银行产品手册.md"],
                "expected_answer": "各类银行产品列表",
            },
            {
                "question": "银行业面临的主要风险有哪些？",
                "relevant_docs": ["银行业行业知识.md"],
                "expected_answer": "信用风险、市场风险、操作风险等",
            },
            {
                "question": "兴业银行2022年与2023年的业绩对比如何？",
                "relevant_docs": [
                    "兴业银行_2022年年度报告.pdf",
                    "兴业银行_2023年年度报告.pdf",
                ],
                "expected_answer": "两年业绩对比",
            },
            {
                "question": "联化科技2025年的研发投入情况？",
                "relevant_docs": ["联化科技_2025年年度报告.pdf"],
                "expected_answer": "研发投入金额和占比",
            },
            {
                "question": "兴业银行2023年的资产规模是多少？",
                "relevant_docs": ["兴业银行_2023年年度报告.pdf"],
                "expected_answer": "资产总额具体数值",
            },
            {
                "question": "银行的资本充足率是什么？有什么要求？",
                "relevant_docs": ["银行业行业知识.md"],
                "expected_answer": "资本充足率定义和监管要求",
            },
            {
                "question": "兴业银行的企业贷款产品有哪些特点？",
                "relevant_docs": ["兴业银行产品手册.md"],
                "expected_answer": "企业贷款产品特点",
            },
        ]

    # ========================================================================
    # 检索评估
    # ========================================================================

    def evaluate_retrieval(
        self, test_data: List[Dict[str, Any]], top_k: int = 5
    ) -> Dict[str, Any]:
        """
        评估检索质量：对每条测试数据进行检索，计算各项检索指标。

        【参数】
            test_data: List[Dict] — 测试数据，每条需包含 question 和 relevant_docs
            top_k: int — 检索返回数量

        【返回值】
            Dict — 包含各指标均值和每个用例的详细结果
        """
        hit_rates_3, hit_rates_5 = [], []
        recalls_3, recalls_5 = [], []
        precisions_3, precisions_5 = [], []
        mrrs = []
        ndcgs_3, ndcgs_5 = [], []
        details = []

        for i, case in enumerate(test_data):
            question = case.get("question", "")
            relevant = case.get("relevant_docs", [])
            if not question or not relevant:
                continue

            try:
                results = self.rag.search_only(question, top_k=top_k)
                doc_ids = [r.get('file_name', '') for r in results]
                # NDCG 需要「文档 -> 相关性分数」，这里按二元相关性（相关=1）构造
                rel_map = {d: 1.0 for d in relevant}

                detail = {
                    "question": question[:60],
                    "retrieved": doc_ids,
                    "relevant": relevant,
                    "hit_rate_3": RetrievalMetrics.hit_rate(doc_ids, relevant, 3),
                    "hit_rate_5": RetrievalMetrics.hit_rate(doc_ids, relevant, 5),
                    "recall_3": RetrievalMetrics.recall(doc_ids, relevant, 3),
                    "recall_5": RetrievalMetrics.recall(doc_ids, relevant, 5),
                    "precision_3": RetrievalMetrics.precision(doc_ids, relevant, 3),
                    "precision_5": RetrievalMetrics.precision(doc_ids, relevant, 5),
                    "mrr": RetrievalMetrics.mrr(doc_ids, relevant),
                    "ndcg_3": RetrievalMetrics.ndcg(doc_ids, rel_map, 3),
                    "ndcg_5": RetrievalMetrics.ndcg(doc_ids, rel_map, 5),
                }

                hit_rates_3.append(detail["hit_rate_3"])
                hit_rates_5.append(detail["hit_rate_5"])
                recalls_3.append(detail["recall_3"])
                recalls_5.append(detail["recall_5"])
                precisions_3.append(detail["precision_3"])
                precisions_5.append(detail["precision_5"])
                mrrs.append(detail["mrr"])
                ndcgs_3.append(detail["ndcg_3"])
                ndcgs_5.append(detail["ndcg_5"])
                details.append(detail)

                logger.info(
                    "[%d/%d] 检索评估: %s → Hit@3=%s",
                    i + 1, len(test_data), question[:30], detail["hit_rate_3"],
                )

            except Exception as e:
                logger.error("检索评估失败 [%d]: %s", i + 1, e)

        return {
            "hit_rate_3": sum(hit_rates_3) / len(hit_rates_3) if hit_rates_3 else 0,
            "hit_rate_5": sum(hit_rates_5) / len(hit_rates_5) if hit_rates_5 else 0,
            "recall_3": sum(recalls_3) / len(recalls_3) if recalls_3 else 0,
            "recall_5": sum(recalls_5) / len(recalls_5) if recalls_5 else 0,
            "precision_3": sum(precisions_3) / len(precisions_3) if precisions_3 else 0,
            "precision_5": sum(precisions_5) / len(precisions_5) if precisions_5 else 0,
            "mrr": sum(mrrs) / len(mrrs) if mrrs else 0,
            "ndcg_3": sum(ndcgs_3) / len(ndcgs_3) if ndcgs_3 else 0,
            "ndcg_5": sum(ndcgs_5) / len(ndcgs_5) if ndcgs_5 else 0,
            "details": details,
        }

    # ========================================================================
    # 生成评估
    # ========================================================================

    def evaluate_generation(
        self, test_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        评估生成质量：对每条测试数据进行 RAG 问答，LLM-as-Judge 评估。

        【参数】
            test_data: List[Dict] — 测试数据，每条需包含 question

        【返回值】
            Dict — 包含忠实度、回答相关性、上下文相关性的均值和详情
        """
        faithfulness_scores = []
        answer_relevance_scores = []
        context_relevance_scores = []
        details = []

        for i, case in enumerate(test_data):
            question = case.get("question", "")
            if not question:
                continue

            try:
                result = self.rag.ask(question, top_k=5, rerank_top=3)
                answer = result.get("answer", "")
                contexts = result.get("contexts", [])

                logger.info("[%d/%d] 正在评估: %s", i + 1, len(test_data), question[:30])

                faith = GenerationMetrics.faithfulness(
                    answer, contexts, self.judge_llm
                )
                relevance = GenerationMetrics.answer_relevance(
                    question, answer, self.judge_llm
                )
                ctx_rel = GenerationMetrics.context_relevance(
                    question, contexts, self.judge_llm
                )

                detail = {
                    "question": question[:60],
                    "answer": answer[:200],
                    "faithfulness": faith["score"],
                    "faithfulness_reason": faith["reason"],
                    "answer_relevance": relevance["score"],
                    "answer_relevance_reason": relevance["reason"],
                    "context_relevance": ctx_rel["score"],
                    "context_relevance_reason": ctx_rel["reason"],
                }

                faithfulness_scores.append(faith["score"])
                answer_relevance_scores.append(relevance["score"])
                context_relevance_scores.append(ctx_rel["score"])
                details.append(detail)

                logger.info(
                    "  → 忠实度:%d 回答相关性:%d 上下文相关性:%d",
                    faith["score"], relevance["score"], ctx_rel["score"],
                )

            except Exception as e:
                logger.error("生成评估失败 [%d]: %s", i + 1, e)

        return {
            "faithfulness": (
                sum(faithfulness_scores) / len(faithfulness_scores)
                if faithfulness_scores else 0
            ),
            "answer_relevance": (
                sum(answer_relevance_scores) / len(answer_relevance_scores)
                if answer_relevance_scores else 0
            ),
            "context_relevance": (
                sum(context_relevance_scores) / len(context_relevance_scores)
                if context_relevance_scores else 0
            ),
            "details": details,
        }

    # ========================================================================
    # 综合评估
    # ========================================================================

    def evaluate(
        self,
        test_data: List[Dict[str, Any]] = None,
        top_k: int = 5,
        enable_generation_eval: bool = True,
    ) -> str:
        """
        完整评估：检索评估 + 生成评估 → 综合报告。

        【参数】
            test_data: List[Dict] | None — 测试数据，不传则使用默认数据集
            top_k: int — 检索返回数量
            enable_generation_eval: bool — 是否启用生成评估（需要 LLM，耗时较长）

        【返回值】
            str — 格式化的评估报告
        """
        if test_data is None:
            test_data = self.generate_default_test_data()

        logger.info("=" * 60)
        logger.info("开始 RAG 系统评估，共 %d 条测试数据", len(test_data))
        logger.info("=" * 60)

        start_time = time.time()

        # Step 1：检索评估
        logger.info("阶段 1/2：检索质量评估...")
        retrieval_result = self.evaluate_retrieval(test_data, top_k=top_k)

        # Step 2：生成评估
        gen_result = {"faithfulness": 0, "answer_relevance": 0, "context_relevance": 0}
        if enable_generation_eval:
            logger.info("阶段 2/2：生成质量评估（LLM-as-Judge）...")
            gen_result = self.evaluate_generation(test_data)

        # 计算综合得分
        retrieval_score = (
            retrieval_result["hit_rate_5"] * 40
            + retrieval_result["mrr"] * 30
            + retrieval_result["recall_5"] * 30
        ) * 100

        generation_score = (
            gen_result["faithfulness"] * 40
            + gen_result["answer_relevance"] * 30
            + gen_result["context_relevance"] * 30
        ) / 5 * 100

        overall_score = retrieval_score * 0.5 + generation_score * 0.5
        grade = self._get_grade(overall_score)

        elapsed = time.time() - start_time

        # 生成报告
        report = REPORT_TEMPLATE.format(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            total_cases=len(test_data),
            model=cfg.llm_model,
            hit_rate_3=retrieval_result["hit_rate_3"],
            hit_rate_5=retrieval_result["hit_rate_5"],
            recall_3=retrieval_result["recall_3"],
            recall_5=retrieval_result["recall_5"],
            mrr=retrieval_result["mrr"],
            precision_3=retrieval_result["precision_3"],
            precision_5=retrieval_result["precision_5"],
            faithfulness=gen_result["faithfulness"],
            answer_relevance=gen_result["answer_relevance"],
            context_relevance=gen_result["context_relevance"],
            retrieval_score=retrieval_score,
            generation_score=generation_score,
            overall_score=overall_score,
            grade=grade,
        )

        report += f"评估耗时：{elapsed:.1f} 秒\n"

        logger.info("评估完成，耗时 %.1f 秒，综合得分: %.1f", elapsed, overall_score)
        return report

    def evaluate_and_save(
        self,
        test_data: List[Dict[str, Any]] = None,
        output_path: str = "evaluation_report.txt",
        top_k: int = 5,
        enable_generation_eval: bool = True,
    ) -> str:
        """
        评估并保存报告到文件。

        【参数】
            test_data: List[Dict] | None — 测试数据
            output_path: str — 报告输出路径
            top_k: int — 检索返回数量
            enable_generation_eval: bool — 是否启用生成评估

        【返回值】
            str — 评估报告文本
        """
        report = self.evaluate(test_data, top_k, enable_generation_eval)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info("评估报告已保存到：%s", output_path)

        return report

    def evaluate_retrieval_only(
        self, test_data: List[Dict[str, Any]] = None, top_k: int = 5
    ) -> Dict[str, Any]:
        """
        仅评估检索（不需要 LLM，速度快）。

        【参数】
            test_data: List[Dict] | None — 测试数据
            top_k: int — 检索返回数量

        【返回值】
            Dict — 检索指标汇总
        """
        if test_data is None:
            test_data = self.generate_default_test_data()
        return self.evaluate_retrieval(test_data, top_k)

    @staticmethod
    def _get_grade(score: float) -> str:
        """根据综合得分返回等级"""
        if score >= 90:
            return "🏆 优秀（A）"
        elif score >= 80:
            return "👍 良好（B）"
        elif score >= 70:
            return "✅ 合格（C）"
        elif score >= 60:
            return "⚠️ 待改进（D）"
        else:
            return "❌ 不合格（F）"


# ============================================================================
# 命令行入口
# ============================================================================
if __name__ == '__main__':
    import sys

    print("=" * 60)
    print("RAG 系统评估工具")
    print("=" * 60)

    evaluator = RAGEvaluator()

    # 使用默认测试数据
    test_data = RAGEvaluator.generate_default_test_data()

    if len(sys.argv) > 1 and sys.argv[1] == '--retrieval-only':
        print("\n仅评估检索质量（不需要 LLM）...\n")
        result = evaluator.evaluate_retrieval_only(test_data, top_k=5)
        print(f"\n检索评估结果：")
        print(f"  HitRate@3: {result['hit_rate_3']:.2%}")
        print(f"  HitRate@5: {result['hit_rate_5']:.2%}")
        print(f"  Recall@3:  {result['recall_3']:.2%}")
        print(f"  Recall@5:  {result['recall_5']:.2%}")
        print(f"  MRR:       {result['mrr']:.2%}")
        print(f"  Precision@3: {result['precision_3']:.2%}")
        print(f"  Precision@5: {result['precision_5']:.2%}")
    else:
        print("\n完整评估（检索 + 生成）...\n")
        report = evaluator.evaluate_and_save(test_data, "evaluation_report.txt")
        print(report)

    evaluator.close()