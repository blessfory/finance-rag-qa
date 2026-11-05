# ============================================================================
# 模块：rag/repo/milvus_repo.py
# Milvus 向量数据库操作（pymilvus v2.5+，使用 MilvusClient 新 API）
# ============================================================================
# 【Milvus 是什么？】
# 专为向量检索设计的分布式数据库，支撑 RAG 的语义检索。
#
# 【核心概念（类比 MySQL 理解）】
#   MySQL          →  Milvus
#   ─────────         ─────────
#   Database       →  Database       数据库
#   Table          →  Collection     集合（表）
#   Row            →  Entity         实体（行）
#   Column（id）   →  Field（id）    主键字段
#   Column（text） →  Field（text）  标量字段
#   Column（vec）  →  Field（vec）   向量字段（MySQL 没有这个）
#   Index（B+树）  →  Index（IVF/HNSW） 索引（算法不同）
#
# 【PyMilvus 新旧 API 对比】
#   旧 API（ORM 模式，已废弃）         新 API（MilvusClient，推荐）
#   ────────────────────────────       ────────────────────────────
#   connections.connect(host, port)     MilvusClient(uri="http://host:port")
#   Collection(name, schema)           client.create_collection(name, schema=schema)
#   FieldSchema / CollectionSchema     client.create_schema() + schema.add_field()
#   collection.create_index(...)       client.prepare_index_params() + 一步创建
#   collection.insert(data)            client.insert(collection_name, data)
#   collection.search(...)             client.search(collection_name, ...)
#   collection.load()                  create_collection 时自动加载
#   utility.has_collection(name)       client.has_collection(name)
#   utility.drop_collection(name)      client.drop_collection(name)
#
# 【为什么用 Milvus 而不是 Faiss？】
#   Faiss 是本地库，数据存内存，重启就没了，不支持分布式。
#   Milvus 是服务端数据库，数据持久化，支持分布式部署，适合生产环境。
#
# 【Milvus 索引类型详解】
#   ┌──────────────┬────────────────────────────────────────────────────┐
#   │ 索引类型      │ 说明                                               │
#   ├──────────────┼────────────────────────────────────────────────────┤
#   │ IVF_FLAT     │ 倒排文件索引。先对向量做 K-Means 聚类，搜索时只搜索  │
#   │              │ 最近的几个聚类中心（nprobe 个），速度快，精度高。    │
#   │              │ 参数：nlist（聚类中心数），nprobe（搜索时探测的聚类数）│
#   │              │ 推荐：数据量 10万~1000万 级别首选                    │
#   ├──────────────┼────────────────────────────────────────────────────┤
#   │ HNSW         │ 分层可导航小世界图。构建多层图结构，搜索效率最高，    │
#   │              │ 但内存占用大（需存储图结构）。                       │
#   │              │ 参数：M（每层连接数），efConstruction（构建时搜索宽度）│
#   │              │ 推荐：对延迟要求极高、内存充足的场景                  │
#   ├──────────────┼────────────────────────────────────────────────────┤
#   │ FLAT         │ 暴力搜索。逐个比较所有向量，100% 精度，速度最慢。    │
#   │              │ 推荐：数据量 < 1万 时可用，或作为精度基准测试        │
#   └──────────────┴────────────────────────────────────────────────────┘
#
# 【相似度度量类型（metric_type）】
#   ┌──────────────┬────────────────────────────────────────────────────┐
#   │ 度量类型      │ 说明                                               │
#   ├──────────────┼────────────────────────────────────────────────────┤
#   │ IP           │ 内积（Inner Product）。值越大越相似。               │
#   │              │ 要求向量已做 L2 归一化（模长为 1），此时等价于余弦   │
#   │              │ BGE 系列模型推荐使用 IP + normalize_embeddings=True │
#   ├──────────────┼────────────────────────────────────────────────────┤
#   │ COSINE       │ 余弦相似度。值域 [-1, 1]，值越大越相似。            │
#   │              │ Milvus 内部会自动归一化，无需手动处理                │
#   ├──────────────┼────────────────────────────────────────────────────┤
#   │ L2           │ 欧几里得距离。值越小越相似。                        │
#   │              │ 注意：L2 的排序逻辑是"越小越好"，与 IP/COSINE 相反  │
#   └──────────────┴────────────────────────────────────────────────────┘
#
# 【检索流程】
#   1. 用户提问 → EmbeddingService 转向量
#   2. Milvus 搜索最相似的 Top-K 个子块（search）
#   3. 根据子块的 parent_index 找到对应的父块
#   4. 父块内容喂给 LLM 生成回答
# ============================================================================

import json
from typing import List, Dict, Any

from langchain_core.documents import Document
from pymilvus import DataType, MilvusClient

from base.config import Config
from base.logger import setup_logger

cfg = Config()
logger = setup_logger('RAG.vectordb', log_file=cfg.log_file)

MILVUS_HOST = cfg.milvus_host
MILVUS_PORT = cfg.milvus_port
COLLECTION_NAME = cfg.milvus_collection_name or 'rag_documents'
EMBEDDING_DIM = 1024
RETRIEVAL_K = cfg.retrieval_k


class MilvusVectorStore:
    """
    Milvus 向量数据库操作类（只负责数据库 CRUD + 检索，不负责模型加载）

    【依赖注入】
        embedding_service: EmbeddingService — 向量化服务，由外部传入。
        这样换模型或换向量库都不需要改这个类。

    【使用方式】
        from rag.core.embedding import EmbeddingService
        embedder = EmbeddingService()
        store = MilvusVectorStore(embedder)
        store.create_collection(drop_if_exists=True)
        store.insert(parent_chunks, child_chunks)
        results = store.search("用户问题")

    【内部数据结构 — Collection 字段设计】
        ┌──────────────┬────────────────┬──────────────────────────────┐
        │ 字段名        │ 类型            │ 说明                         │
        ├──────────────┼────────────────┼──────────────────────────────┤
        │ id           │ VARCHAR(256)   │ 主键，chunk 的唯一标识         │
        │ text         │ VARCHAR(65535) │ 子块文本内容                   │
        │ embedding    │ FLOAT_VECTOR   │ 子块的 1024 维向量（BGE-M3）   │
        │ parent_index │ INT64          │ 指向父块的索引，用于返回完整上下文│
        │ file_name    │ VARCHAR(512)   │ 源文件名                       │
        │ file_type    │ VARCHAR(32)    │ 文件类型（pdf/md/txt/csv 等）  │
        │ category     │ VARCHAR(64)    │ 类别标签（如 'ai'、'java'）    │
        │ chunk_index  │ INT64          │ chunk 在原始文档中的序号        │
        │ parent_text  │ VARCHAR(65535) │ 父块完整文本（避免跨进程丢失）    │
        │ metadata_json│ VARCHAR(65535) │ 完整元数据 JSON（兜底字段）     │
        └──────────────┴────────────────┴──────────────────────────────┘
    """

    def __init__(
        self,
        embedding_service,
        host: str = MILVUS_HOST,
        port: int = MILVUS_PORT,
        collection_name: str = COLLECTION_NAME,
    ):
        """
        【参数】
            embedding_service: EmbeddingService — 向量化服务（依赖注入）
            host: str — Milvus 服务地址，默认 localhost
            port: int — Milvus gRPC 端口，默认 19530
            collection_name: str — Collection 名称，类似 MySQL 的表名
        """
        self.collection_name = collection_name
        self.embedding_service = embedding_service

        uri = f"http://{host}:{port}"
        self.client = MilvusClient(uri=uri)
        logger.info("已连接 Milvus：%s", uri)

        self._parent_chunks: List[Document] = []

    # ========================================================================
    # Collection 管理
    # ========================================================================

    def create_collection(self, drop_if_exists: bool = False):
        """
        创建 Collection（一步创建 schema + 索引，自动加载）

        【参数】
            drop_if_exists: bool — 已存在时是否先删后建。
                                  True：先删后建（适合开发阶段反复测试）
                                  False：已存在则直接复用（适合生产环境）

        【MilvusClient.create_schema() 参数】
            auto_id: bool — 是否自动生成主键 ID。False 表示手动指定 ID
            enable_dynamic_field: bool — 是否允许动态字段。
                                       False 表示严格 Schema 模式，未定义的字段不能插入

        【DataType 常用类型】
            DataType.VARCHAR       → 可变长字符串，需指定 max_length
            DataType.INT64         → 64 位整数
            DataType.FLOAT_VECTOR  → 浮点向量，需指定 dim（维度）
            DataType.BOOL          → 布尔值
            DataType.FLOAT         → 浮点数
            DataType.JSON          → JSON 字符串

        【prepare_index_params() 返回的 IndexParams 常用方法】
            add_index(field_name, index_type, metric_type, params):
                field_name: str — 要建索引的字段名，通常是 embedding 字段
                index_type: str — 索引类型：'IVF_FLAT', 'HNSW', 'FLAT'
                metric_type: str — 相似度度量：'IP', 'COSINE', 'L2'
                params: dict — 索引参数，如 {'nlist': 128}（聚类中心数）
        """
        if self.client.has_collection(self.collection_name):
            if drop_if_exists:
                self.client.drop_collection(self.collection_name)
                logger.info("已删除旧 Collection：%s", self.collection_name)
            else:
                logger.info("Collection 已存在：%s，直接使用", self.collection_name)
                return

        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)

        schema.add_field(field_name='id', datatype=DataType.VARCHAR, is_primary=True, max_length=256)
        schema.add_field(field_name='text', datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name='embedding', datatype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
        schema.add_field(field_name='parent_index', datatype=DataType.INT64)
        schema.add_field(field_name='file_name', datatype=DataType.VARCHAR, max_length=512)
        schema.add_field(field_name='file_type', datatype=DataType.VARCHAR, max_length=32)
        schema.add_field(field_name='category', datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name='chunk_index', datatype=DataType.INT64)
        schema.add_field(field_name='parent_text', datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name='metadata_json', datatype=DataType.VARCHAR, max_length=65535)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name='embedding', index_type='IVF_FLAT',
            metric_type='IP', params={'nlist': 128},
        )

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema, index_params=index_params,
        )
        logger.info("已创建 Collection：%s（已自动加载到内存）", self.collection_name)

    # ========================================================================
    # 数据插入
    # ========================================================================

    def insert(self, parent_chunks: List[Document], child_chunks: List[Document]):
        """
        将父子块插入 Milvus

        【参数】
            parent_chunks: List[Document] — 父块列表，缓存到内存中供检索时取完整上下文
            child_chunks:  List[Document] — 子块列表，存入 Milvus 做向量检索

        【内部流程】
            1. 缓存父块到 self._parent_chunks（内存），检索时根据 parent_index 获取
            2. 提取子块文本 → EmbeddingService 向量化
            3. 构造插入数据（id/text/embedding/metadata）
            4. 调用 MilvusClient.insert() 批量写入

        【MilvusClient.insert() 参数】
            collection_name: str — Collection 名称
            data: List[Dict] — 插入数据，每个 Dict 的 key 对应 schema 中的 field_name

        【返回值】
            Dict，包含：
            - insert_count: int — 成功插入的条数
            - ids: List[str] — 插入的主键 ID 列表
        """
        self._parent_chunks = parent_chunks
        texts = [chunk.page_content for chunk in child_chunks]

        logger.info("正在将 %d 个子块转为向量...", len(texts))
        embeddings = self.embedding_service.encode(texts, normalize=True, show_progress_bar=True)

        insert_data = []
        for i, (chunk, emb) in enumerate(zip(child_chunks, embeddings)):
            parent_idx = chunk.metadata.get('parent_index', -1)
            parent_text = ''
            if 0 <= parent_idx < len(parent_chunks):
                parent_text = parent_chunks[parent_idx].page_content[:65000]
            insert_data.append({
                'id': f"chunk_{i}",
                'text': chunk.page_content[:65000],
                'embedding': emb.tolist(),
                'parent_index': parent_idx,
                'parent_text': parent_text,
                'file_name': chunk.metadata.get('file_name', ''),
                'file_type': chunk.metadata.get('file_type', ''),
                'category': chunk.metadata.get('category', ''),
                'chunk_index': chunk.metadata.get('chunk_index', i),
                'metadata_json': json.dumps(chunk.metadata, ensure_ascii=False),
            })

        result = self.client.insert(collection_name=self.collection_name, data=insert_data)
        logger.info("已插入 %d 条数据", result['insert_count'])

        # 插入后必须 flush，否则数据在缓冲区中，搜索时查不到
        self.client.flush(self.collection_name)
        logger.info("数据已刷盘，可供搜索")

    # ========================================================================
    # 向量检索
    # ========================================================================

    def search(self, query: str, top_k: int = RETRIEVAL_K) -> List[Dict[str, Any]]:
        """
        语义检索：将用户问题转为向量，在 Milvus 中搜索最相似的子块

        【参数】
            query: str — 用户提问的原始文本
            top_k: int — 返回的最相似结果数量，默认从配置读取（RETRIEVAL_K）

        【MilvusClient.search() 参数详解】
            collection_name: str — 要搜索的 Collection 名称
            data: List[List[float]] — 查询向量列表，每个元素是一个向量
            anns_field: str — 向量字段名，即 schema 中 FLOAT_VECTOR 类型的字段名
            search_params: dict — 搜索参数：
                - metric_type: str — 相似度度量，与建索引时一致
                - params: dict — 索引参数，如 {'nprobe': 16}
                  nprobe 越大精度越高速度越慢，推荐 [1, nlist] 之间
            limit: int — 返回多少条结果（Top-K）
            output_fields: List[str] — 返回哪些标量字段，不包含向量字段

        【返回值】
            List[Dict[str, Any]]，每个 Dict 包含：
            - id: str — chunk 的主键 ID
            - text: str — 子块文本内容
            - score: float — 相似度分数（IP 度量下越大越相似）
            - parent_index: int — 对应父块的索引
            - parent_text: str — 父块完整文本（已从 Milvus 标量字段直接返回）
            - file_name: str — 源文件名
            - file_type: str — 文件类型
            - category: str — 类别标签

        【nprobe 参数调优】
            nprobe 控制搜索时探测的聚类中心数：
            - nprobe=1：只搜索最近的一个聚类，速度最快，但可能漏掉结果
            - nprobe=nlist：搜索所有聚类，相当于暴力搜索，100% 精度但最慢
            - 推荐值：nlist=128 时 nprobe=16，精度和速度的平衡点
        """
        query_embedding = self.embedding_service.embed_query(query)

        logger.info("查询向量维度：%d，前5个值：%s", len(query_embedding), query_embedding[:5])
        logger.info("Collection：%s，top_k：%d", self.collection_name, top_k)

        # 确保 Collection 已加载到内存（不加载无法搜索）
        self.client.load_collection(self.collection_name)

        search_params = {'metric_type': 'IP', 'params': {'nprobe': 16}}
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            anns_field='embedding',
            search_params=search_params,
            limit=top_k,
            output_fields=[
                'text', 'parent_index', 'parent_text', 'file_name',
                'file_type', 'category', 'metadata_json',
            ],
        )

        logger.info("Milvus 原始返回：外层列表长度=%d", len(results))
        if results:
            logger.info("  第1次查询命中数：%d", len(results[0]))

        output = []
        for hits in results:
            for hit in hits:
                entity = hit.get('entity', {})
                parent_idx = entity.get('parent_index', -1)

                result = {
                    'id': hit.get('id', ''),
                    'text': entity.get('text', ''),
                    'score': hit.get('distance', 0.0),
                    'parent_index': parent_idx,
                    'parent_text': entity.get('parent_text', ''),
                    'file_name': entity.get('file_name', ''),
                    'file_type': entity.get('file_type', ''),
                    'category': entity.get('category', ''),
                    'metadata_json': entity.get('metadata_json', ''),
                }

                output.append(result)

        return output

    # ========================================================================
    # 资源管理
    # ========================================================================

    def drop_collection(self):
        """
        删除 Collection（不可逆，生产环境慎用）

        【MilvusClient.drop_collection()】
            删除整个 Collection，包括所有数据和索引。
            删除后不可恢复，等同于 MySQL 的 DROP TABLE。
        """
        if self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)
            logger.warning("已删除 Collection：%s", self.collection_name)

    def close(self):
        """
        释放 Milvus 连接资源

        【MilvusClient.close()】
            关闭与 Milvus 服务端的 gRPC 连接，释放网络资源。
            不会删除数据，只是断开连接。
            建议在程序退出或不再使用时调用。
        """
        self.client.close()
        logger.info("已断开 Milvus 连接")