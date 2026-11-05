# -*- coding: utf-8 -*-
"""
回归测试：验证 3 个 Bug 的修复

运行方式（项目根目录）：
    python -m unittest tests.test_bug_fixes -v

3 个 Bug 对应 3 组测试，全部使用 unittest.mock，不依赖
MySQL / Redis / Milvus / LLM 等外部服务。
"""
import os
import sys
import unittest
from unittest import mock

import numpy as np
from langchain_core.documents import Document

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ============================================================================
# Bug 1：bm25.py 使用包内裸导入，导致从项目根目录无法 import
#   修复前：from preprocess import ...  → ModuleNotFoundError
#   修复后：from .preprocess import ...  → 包内相对导入
# ============================================================================
class TestBug1Bm25PackageImport(unittest.TestCase):

    def test_bm25_importable_from_project_root(self):
        """从项目根目录直接 import mysql_qa.bm25 不应抛 ModuleNotFoundError"""
        import mysql_qa.bm25 as bm25_module
        self.assertIsNotNone(getattr(bm25_module, 'BM25Search', None))
        self.assertIn('mysql_qa.bm25', sys.modules)


# ============================================================================
# Bug 2：FAQ 未命中时返回提示语字符串，被 _handle_faq 误判为命中
#   修复前：返回 "未在FAQ知识库中找到匹配答案..."（不包含子串"未找到"）
#   修复后：返回 None，上层可正确降级到向量检索
# ============================================================================
class TestBug2FaqMissReturnsNone(unittest.TestCase):

    def _make_system(self):
        """绕过 __init__（避免真实连接 MySQL/Redis），只注入流程所需依赖"""
        from mysql_qa.sql_main import MySQLQASystem
        system = MySQLQASystem.__new__(MySQLQASystem)
        system.logger = mock.MagicMock()
        system.bm25_search = mock.MagicMock()
        return system

    def test_unmatched_question_returns_none(self):
        """BM25 置信度不足（need_rerank=True）时 query() 应返回 None"""
        system = self._make_system()
        system.bm25_search.search.return_value = (None, True)

        self.assertIsNone(system.query("知识库中不存在的问题"))

    def test_matched_question_returns_answer(self):
        """BM25 命中时 query() 应返回答案字符串（原有行为不变）"""
        system = self._make_system()
        system.bm25_search.search.return_value = ("标准答案", False)

        self.assertEqual(system.query("已收录的问题"), "标准答案")


# ============================================================================
# Bug 3：父块文本只存在内存，跨进程（FastAPI/Streamlit）检索时 parent_text 为空
#   修复：parent_text 写入 Milvus 标量字段，检索时直接从 entity 读取
# ============================================================================
class _FakeEmbedder:
    """假 Embedding 服务，避免加载 BGE-M3 模型"""

    def encode(self, texts, **kwargs):
        return np.zeros((len(texts), 4))

    def embed_query(self, query, **kwargs):
        return [0.0] * 4


class TestBug3ParentText(unittest.TestCase):

    def _make_store(self, mock_client):
        """构造 MilvusVectorStore，client 用 MagicMock 代替，不连接真实 Milvus"""
        if mock_client is None:
            mock_client = mock.MagicMock(name='MilvusClient')
        with mock.patch('rag.repo.milvus_repo.MilvusClient', return_value=mock_client):
            from rag.repo.milvus_repo import MilvusVectorStore
            store = MilvusVectorStore(_FakeEmbedder(), collection_name='test_bug_fixes')
        return store, mock_client

    def test_schema_contains_parent_text_field(self):
        """Collection schema 中必须包含 parent_text 字段"""
        from rag.repo.milvus_repo import MILVUS_HOST, MILVUS_PORT
        client = mock.MagicMock()
        client.has_collection.return_value = False
        client.create_schema.return_value = mock.MagicMock()

        store, client = self._make_store(client)
        store.create_collection(drop_if_exists=True)

        fields = set()
        for call in client.create_schema.return_value.add_field.call_args_list:
            if call.args:
                fields.add(call.args[0])
            elif 'field_name' in call.kwargs:
                fields.add(call.kwargs['field_name'])
        self.assertIn('parent_text', fields)

    def test_insert_stores_parent_text(self):
        """insert() 时必须把父块文本写入 parent_text 字段"""
        parent_docs = [
            Document(page_content='父块完整上下文内容 A'),
            Document(page_content='父块完整上下文内容 B'),
        ]
        child_docs = [
            Document(page_content='子块内容', metadata={
                'parent_index': 1, 'file_name': 'f.pdf',
                'file_type': 'pdf', 'category': 'annual_reports', 'chunk_index': 0,
            }),
        ]
        client = mock.MagicMock()
        store, client = self._make_store(client)
        store.insert(parent_docs, child_docs)

        data = client.insert.call_args.kwargs['data']
        self.assertEqual(data[0]['parent_text'], '父块完整上下文内容 B')

    def test_search_reads_parent_text_from_milvus(self):
        """search() 应从 Milvus entity 直接读取 parent_text（不依赖内存缓存）"""
        client = mock.MagicMock()
        client.search.return_value = [[{
            'id': 'chunk_0',
            'distance': 0.92,
            'entity': {
                'text': '子块内容',
                'parent_index': 1,
                'parent_text': '数据库返回的父块内容',
                'file_name': 'f.pdf', 'file_type': 'pdf', 'category': 'annual_reports',
            },
        }]]
        store, client = self._make_store(client)

        results = store.search('测试查询', top_k=1)

        self.assertEqual(results[0]['parent_text'], '数据库返回的父块内容')
        self.assertEqual(results[0]['text'], '子块内容')

    def test_knowledge_context_falls_back_to_child_text(self):
        """rag_service 组装 context 时，parent_text 为空应回落到子块 text"""
        from rag.rag_service import RAGService

        svc = RAGService.__new__(RAGService)          # 绕过 __init__（不加载模型）
        svc.llm = mock.MagicMock()
        svc.llm.generate.return_value = '生成的回答'
        svc.reranker = mock.MagicMock()
        svc.reranker.available = False                # 跳过 rerank，直接取前 n 条

        svc.store = mock.MagicMock()
        svc.store.search.return_value = [
            {'parent_text': '', 'text': '子块文本A', 'file_name': 'f.pdf', 'score': 0.9},
            {'parent_text': '', 'text': '子块文本B', 'file_name': 'f.pdf', 'score': 0.8},
        ]

        result = svc._handle_knowledge(
            '测试问题', top_k=10, rerank_top=3, enable_rerank=True, enable_rewrite=False,
        )

        self.assertEqual(result['contexts'], ['子块文本A', '子块文本B'])
        # 生成的 prompt 应包含子块文本（context 非空）
        prompt = svc.llm.generate.call_args.args[0][0]['content']
        self.assertIn('子块文本A', prompt)


if __name__ == '__main__':
    unittest.main()