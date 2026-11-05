# ============================================================================
# 模块：rag/evaluation/__init__.py
# RAG 评估模块
# ============================================================================
# 【评估维度】
#   1. 检索评估：评估向量检索的质量（召回率、命中率、MRR等）
#   2. 生成评估：评估 LLM 生成回答的质量（忠实度、相关性等）
#   3. 端到端评估：综合评估整个 RAG 流程的效果
#
# 【使用方式】
#   from rag.evaluation import RAGEvaluator
#   evaluator = RAGEvaluator()
#   report = evaluator.evaluate(test_data)
#   print(report)
# ============================================================================

from rag.evaluation.evaluator import RAGEvaluator
from rag.evaluation.metrics import RetrievalMetrics, GenerationMetrics

__all__ = ['RAGEvaluator', 'RetrievalMetrics', 'GenerationMetrics']