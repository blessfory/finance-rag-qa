# ============================================================================
# 模块：rag/pipeline.py
# 编排层（Orchestration Layer）：一键构建向量库
# ============================================================================
# 【模块定位】
# 这是 RAG 向量库构建的"一键启动"入口。
# 把"加载文档 → 父子分块 → 向量化 → 存入 Milvus"串联成一个函数调用。
#
# 【设计模式：编排模式（Orchestration）】
# 编排层不包含业务逻辑，只负责按顺序调用各模块，协调它们之间的数据传递。
# 好处：如果某个步骤需要替换（如换向量库），只需要改编排层，不影响各模块。
#
# 【处理流程】
#   build_vector_store(directory)
#   │
#   ├── Step 1：DocumentLoader.load_directory()
#   │   └── 遍历目录 → 加载 PDF/MD/TXT/CSV → List[Document]（原始文档，不做分块）
#   │
#   ├── Step 2：ParentChildChunker.split()
#   │   └── 原始文档 → 父子分块 → (parent_chunks, child_chunks)
#   │
#   └── Step 3：MilvusVectorStore
#       ├── create_collection() → 创建/复用 Collection
#       ├── EmbeddingService.encode() → 子块向量化（1024维）
#       └── insert() → 存入 Milvus
#
# 【使用方式】
#   # 方式一：作为模块导入
#   from rag.pipeline import build_vector_store
#   store = build_vector_store("data/finance/", drop_if_exists=True)
#
#   # 方式二：直接运行
#   python -m rag.pipeline
#
# 【数据目录结构建议】
#   data/
#   └── finance/               ← 根目录
#       ├── annual_reports/    ← 自动识别为 category="annual_reports"
#       ├── products/          ← 自动识别为 category="products"
#       └── industry_knowledge/← 自动识别为 category="industry_knowledge"
# ============================================================================

import os

from base.config import Config
from base.logger import setup_logger
from rag.core.embedding import EmbeddingService
from rag.repo.milvus_repo import MilvusVectorStore, COLLECTION_NAME

cfg = Config()
logger = setup_logger('RAG.pipeline', log_file=cfg.log_file)


def build_vector_store(
    directory: str,
    add_category: str = None,
    collection_name: str = None,
    drop_if_exists: bool = False,
) -> MilvusVectorStore:
    """
    一键构建向量库：加载文档 → 父子分块 → 存入 Milvus

    【参数】
        directory: str — 文档目录路径，会递归加载目录下所有支持的文件
        add_category: str | None — 类别标签，不传则自动从子目录名提取。
                              检索时可通过 category 过滤结果
        collection_name: str | None — Milvus Collection 名称。
                                      默认使用 config.ini 中的配置，或 'rag_documents'
        drop_if_exists: bool — 是否删除已存在的 Collection 重新创建。
                              True：重建（开发测试用，数据会丢失）
                              False：复用已有 Collection（生产环境用）

    【返回值】
        MilvusVectorStore — 构建完成的向量库实例，可直接用于 search() 检索

    【各步骤说明】
        Step 1 — 加载文档：遍历目录，加载所有支持的文档格式，返回原始 Document 列表
        Step 2 — 父子分块：将原始文档拆分为父块（大块，喂给 LLM）和子块（小块，存入向量库）
        Step 3 — 存入 Milvus：子块向量化后存入 Milvus，父块缓存到内存供检索时取完整上下文

    【使用示例】
        # 首次构建：删除旧数据，重新建库（自动按子目录分类）
        store = build_vector_store("data/finance/", drop_if_exists=True)

        # 增量添加：不删除已有数据，追加新文档
        store = build_vector_store("data/finance/")

        # 检索测试
        results = store.search("兴业银行2025年营收多少？")
        for r in results:
            print(f"相似度: {r['score']:.4f}, 来源: {r['file_name']}")
    """
    from rag.document_loader import DocumentLoader
    from rag.chunker import ParentChildChunker

    logger.info("=" * 60)
    logger.info("Step 1/3：加载文档")
    logger.info("=" * 60)
    dl = DocumentLoader()
    docs = dl.load_directory(directory, add_category=add_category)
    logger.info("加载完成，共 %d 个原始文档", len(docs))

    logger.info("Step 2/3：父子分块")
    chunker = ParentChildChunker()
    parent_chunks, child_chunks = chunker.split(docs)
    logger.info("父块：%d 个，子块：%d 个", len(parent_chunks), len(child_chunks))

    logger.info("Step 3/3：存入 Milvus")
    embedder = EmbeddingService()
    store = MilvusVectorStore(
        embedding_service=embedder,
        collection_name=collection_name or COLLECTION_NAME,
    )
    store.create_collection(drop_if_exists=drop_if_exists)
    store.insert(parent_chunks, child_chunks)

    logger.info("向量库构建完成！")
    return store


if __name__ == '__main__':
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'finance')
    store = build_vector_store(data_dir, drop_if_exists=True)

    logger.info("=" * 60)
    logger.info("测试检索")
    logger.info("=" * 60)
    results = store.search("兴业银行2025年营收多少？")

    logger.info("检索到 %d 条结果", len(results))

    if not results:
        logger.warning("未检索到任何结果！可能原因：")
        logger.warning("  1. 文档目录下没有文件")
        logger.warning("  2. 文档内容为空或无法解析")
        logger.warning("  3. Milvus 服务未启动")
        print("⚠️ 未检索到任何结果，请检查文档目录和 Milvus 服务状态")

    for i, r in enumerate(results):
        logger.info("--- 结果 %d（相似度：%.4f）---", i + 1, r['score'])
        logger.info("文件：%s", r['file_name'])
        logger.info("子块预览：%s...", r['text'][:100])
        logger.info("父块预览：%s...", r['parent_text'][:100])

    store.close()