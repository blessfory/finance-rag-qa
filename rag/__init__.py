# ============================================================================
# rag — RAG 核心模块
# ============================================================================
# 【架构】
#   rag/
#   ├── rag_service.py     — RAG 服务编排层（ask 入口）
#   ├── pipeline.py        — 向量存储构建流水线
#   ├── chunker.py         — 父子分块策略
#   ├── prompts.py         — 提示词管理
#   ├── document_loader.py — 文档加载门面
#   ├── core/              — 核心服务（Embedding/LLM/Reranker）
#   ├── repo/              — 数据访问层（Milvus Repository）
#   ├── loaders/           — 各格式文档加载器
#   └── evaluation/        — 评估模块
#
# 【使用方式】
#   from rag.rag_service import RAGService
#   rag = RAGService()
#   answer = rag.ask("兴业银行2023年净利润是多少？")
#   rag.close()
#   ---
#   from rag.pipeline import build_vector_store
#   store = build_vector_store("data/")
# ============================================================================