# ============================================================================
# 模块：rag/core/reranker.py
# Reranker 重排序服务：对粗排检索结果做精排
# ============================================================================

import os
from typing import List, Dict, Any

from base.config import Config
from base.logger import setup_logger

cfg = Config()
logger = setup_logger('RAG.reranker', log_file=cfg.log_file)


class RerankerService:
    """重排序服务：对向量检索的粗排结果做精排，提高最终回答质量。"""

    def __init__(self, model_path: str = None, device: str = None):
        try:
            from FlagEmbedding import FlagReranker

            if model_path is None:
                model_path = os.path.join(
                    os.path.dirname(__file__), '..', '..', 'models',
                    'bge-reranker-large'
                )
            self.reranker = FlagReranker(model_path, use_fp16=True, device=device)
            self._available = True
            logger.info("Reranker 模型已加载：%s", model_path)
        except ImportError:
            logger.warning(
                "FlagEmbedding 未安装，Reranker 不可用。"
                "请执行：pip install FlagEmbedding"
            )
            self.reranker = None
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_n: int = 3,
    ) -> List[Dict[str, Any]]:
        """对检索结果重排序，返回 top_n 个最相关的结果"""
        if not self._available or not documents:
            return documents[:top_n]

        pairs = [[query, doc.get('text', '')] for doc in documents]
        scores = self.reranker.compute_score(pairs, normalize=True)

        if not isinstance(scores, list):
            scores = [scores]

        for i, score in enumerate(scores):
            documents[i]['rerank_score'] = float(score)

        documents.sort(key=lambda x: x.get('rerank_score', 0), reverse=True)
        return documents[:top_n]