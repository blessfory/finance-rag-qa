# ============================================================================
# 模块：rag/rag_service.py
# RAG 完整问答流程：意图识别 → 分流 → 检索 → Rerank → 生成回答
# ============================================================================
# 【RAG 问答全流程】
# ┌──────────────────────────────────────────────────────────────────┐
# │ 用户输入问题                                                     │
# │     ↓                                                           │
# │ 意图识别（IntentRecognizer）                                     │
# │     ↓              ↓              ↓              ↓              │
# │  闲聊回复       FAQ匹配        知识问答        投诉处理          │
# │  (直接回)    (mysql_qa)    (RAG检索)      (转人工)            │
# │                                 ↓                               │
# │                          向量检索（粗排）                        │
# │                                 ↓                               │
# │                          Reranker（精排）                        │
# │                                 ↓                               │
# │                          Prompt 模板填充                         │
# │                                 ↓                               │
# │                          LLM 生成回答                            │
# │                                 ↓                               │
# │                          返回给用户                              │
# └──────────────────────────────────────────────────────────────────┘
# ============================================================================

from typing import List, Dict, Any, Optional

from base.config import Config
from base.logger import setup_logger
from rag.core.embedding import EmbeddingService
from rag.core.llm import LLMService
from rag.core.reranker import RerankerService
from rag.core.intent import IntentRecognizer
from rag.prompts import get_prompt
from rag.repo.milvus_repo import MilvusVectorStore, COLLECTION_NAME

cfg = Config()
logger = setup_logger('RAG.service', log_file=cfg.log_file)


class RAGService:
    """
    RAG 问答服务：意图识别 → 分流 → 检索 → 重排 → 生成 的完整流程。

    【使用方式】
        rag = RAGService()
        result = rag.ask("兴业银行2024年营收和净利润分别是多少？")
        print(result['answer'])
        rag.close()

    【依赖注入】
        所有核心服务都支持外部传入，换模型/换数据库只需改注入的参数。
    """

    def __init__(
        self,
        embedder: EmbeddingService = None,
        llm: LLMService = None,
        reranker: RerankerService = None,
        store: MilvusVectorStore = None,
    ):
        self.embedder = embedder or EmbeddingService()
        self.llm = llm or LLMService()
        self.reranker = reranker or RerankerService()
        self.store = store or MilvusVectorStore(
            embedding_service=self.embedder,
            collection_name=COLLECTION_NAME,
        )
        self.intent_recognizer = IntentRecognizer(llm_service=self.llm)
        logger.info("RAG 服务已初始化")

    def ask(
        self,
        question: str,
        top_k: int = 10,
        rerank_top: int = 3,
        enable_rerank: bool = True,
        enable_rewrite: bool = True,
    ) -> Dict[str, Any]:
        """
        RAG 问答：意图识别 → 分流 → Query改写 → 检索 → 重排 → 生成

        【参数】
            question: str — 用户问题
            top_k: int — 粗排召回数量（默认 10）
            rerank_top: int — 精排保留数量（默认 3）
            enable_rerank: bool — 是否启用 Reranker
            enable_rewrite: bool — 是否启用 Query 改写

        【返回值】
            Dict 包含：
            - answer: str — LLM 生成的最终回答
            - intent: str — 意图类型
            - contexts: List[str] — 使用的上下文
            - sources: List[Dict] — 引用来源
            - rewritten_query: str | None — 改写后的查询（仅当 enable_rewrite=True）
        """
        # Step 1：意图识别
        intent = self.intent_recognizer.recognize(question)
        logger.info("意图识别结果：%s → %s", question[:30], intent)

        # Step 2：根据意图分流处理
        if intent == 'chitchat':
            return self._handle_chitchat(question)

        # FAQ 通道：非闲聊、非投诉的问题都先走 FAQ（MySQL + BM25，毫秒级），
        # 命中即返回；未命中自动降级到下方向量检索，不影响知识问答。
        if intent in ('faq', 'knowledge'):
            faq_result = self._handle_faq(question)
            if faq_result:
                return faq_result

        if intent == 'complaint':
            return self._handle_complaint(question)

        # Step 3：知识问答（RAG 检索 + LLM 生成）
        return self._handle_knowledge(question, top_k, rerank_top, enable_rerank, enable_rewrite)

    # ========================================================================
    # 分流处理
    # ========================================================================

    def _handle_chitchat(self, question: str) -> Dict[str, Any]:
        """闲聊处理：直接回复，不检索"""
        prompt = get_prompt('chitchat', history='', question=question)
        answer = self.llm.generate(
            [{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=256,
        )
        return {
            'answer': answer,
            'intent': 'chitchat',
            'contexts': [],
            'sources': [],
            'rewritten_query': None,
        }

    def _handle_faq(self, question: str) -> Optional[Dict[str, Any]]:
        """FAQ 处理：查 MySQL 关键词匹配"""
        try:
            from mysql_qa.sql_main import MySQLQASystem
            with MySQLQASystem() as faq:
                # 阈值 0.70：实测 0.85 会漏掉「什么是 LPR」这类改写问题，
                # 0.70 可命中 FAQ 且不会误命中「兴业银行 2024 年营业收入」等知识类问题
                answer = faq.query(question, threshold=0.70)
                if answer and "未找到" not in answer:
                    return {
                        'answer': answer,
                        'intent': 'faq',
                        'contexts': [],
                        'sources': [{'file_name': 'FAQ知识库', 'score': 1.0, 'text': answer[:200]}],
                        'rewritten_query': None,
                    }
        except Exception as e:
            logger.warning("FAQ 查询失败：%s", e)
        return None

    def _handle_complaint(self, question: str) -> Dict[str, Any]:
        """投诉处理：安抚用户 + 转人工"""
        prompt = get_prompt(
            'customer_service',
            context='无相关参考资料',
            question=question,
            phone=cfg.customer_service_phone or '400-xxx-xxxx',
        )
        answer = self.llm.generate(
            [{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=512,
        )
        return {
            'answer': answer,
            'intent': 'complaint',
            'contexts': [],
            'sources': [],
            'rewritten_query': None,
        }

    def _handle_knowledge(
        self, question: str, top_k: int, rerank_top: int, enable_rerank: bool, enable_rewrite: bool = True
    ) -> Dict[str, Any]:
        """
        知识问答：Query改写 → RAG 检索 → Rerank → LLM 生成

        【参数】
            question: str — 用户问题
            top_k: int — 粗排召回数量
            rerank_top: int — 精排保留数量
            enable_rerank: bool — 是否启用 Reranker 重排序
            enable_rewrite: bool — 是否启用 Query 改写，默认 True
        """
        rewritten_query = None
        search_query = question

        if enable_rewrite:
            rewritten_query = self._rewrite_query(question)
            if rewritten_query:
                search_query = rewritten_query
                logger.info("Query改写：%s → %s", question[:30], rewritten_query[:50])

        candidates = self.store.search(search_query, top_k=top_k)
        if not candidates:
            return {
                'answer': '抱歉，当前知识库中未找到相关信息。请尝试换个问法，或联系人工客服。',
                'intent': 'knowledge',
                'contexts': [],
                'sources': [],
                'rewritten_query': rewritten_query,
            }

        if enable_rerank and self.reranker.available:
            results = self.reranker.rerank(question, candidates, top_n=rerank_top)
        else:
            results = candidates[:rerank_top]

        contexts = [r.get('parent_text') or r.get('text', '') for r in results]
        sources = [
            {
                'file_name': r.get('file_name', ''),
                'score': r.get('rerank_score', r.get('score', 0)),
                'text': r.get('text', '')[:200],
            }
            for r in results
        ]

        prompt = get_prompt('rag_qa', context='\n\n---\n\n'.join(contexts), question=question)
        answer = self.llm.generate(
            [{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1024,
        )

        return {
            'answer': answer,
            'intent': 'knowledge',
            'contexts': contexts,
            'sources': sources,
            'rewritten_query': rewritten_query,
        }

    # ========================================================================
    # Query 改写
    # ========================================================================

    def _rewrite_query(self, question: str) -> Optional[str]:
        """
        将用户口语化问题改写为适合向量检索的查询语句

        【改写示例】
            "那个贷款利率多少" → "贷款利率 LPR 利率水平"
            "兴业银行去年赚了多少" → "兴业银行 2024年 净利润 营收"
            "有没有利息低的" → "低利率 贷款产品 优惠利率"

        【参数】
            question: str — 用户原始问题

        【返回值】
            str | None — 改写后的查询语句，失败时返回 None（降级用原问题）
        """
        try:
            prompt = get_prompt('query_rewrite', question=question)
            rewritten = self.llm.generate(
                [{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=128,
            )
            rewritten = rewritten.strip().strip('"').strip("'")
            if not rewritten or rewritten == question:
                return None
            return rewritten
        except Exception as e:
            logger.warning("Query改写失败，降级使用原问题：%s", e)
            return None

    # ========================================================================
    # 便捷方法
    # ========================================================================

    def search_only(self, question: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """仅检索，不生成回答（用于调试和评估）"""
        return self.store.search(question, top_k=top_k)

    def close(self):
        """释放资源"""
        self.store.close()