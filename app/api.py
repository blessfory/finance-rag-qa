# ============================================================================
# 模块：app/api.py
# FastAPI 后端服务：为 RAG 系统提供 RESTful API 接口
# ============================================================================
# 【FastAPI 是什么？】
# FastAPI 是一个现代、高性能的 Python Web 框架，专为构建 API 而设计。
# 特点：
#   - 快：基于 Starlette + Pydantic，性能对标 Node.js/Go
#   - 自动文档：访问 /docs 即可看到 Swagger UI 交互式文档
#   - 类型安全：基于 Python 类型提示，自动做参数校验和序列化
#   - 异步支持：原生 async/await，高并发场景下性能优秀
#
# 【启动方式】
#   uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
#
#   --host 0.0.0.0  → 允许外部访问（局域网/公网）
#   --port 8000     → 监听端口
#   --reload        → 开发模式：代码修改后自动重启（生产环境不要用）
#
# 【API 端点一览】
#   GET  /                       → 服务信息
#   GET  /health                 → 健康检查（K8s/Docker 探活用）
#   POST /api/v1/ask             → 问答接口（核心）
#   POST /api/v1/build_index     → 构建向量索引
#   GET  /api/v1/collection_info → 查询 Collection 信息
#   GET  /docs                   → Swagger 交互式文档（FastAPI 自动生成）
#   GET  /redoc                  → ReDoc 文档（FastAPI 自动生成）
# ============================================================================

import os
import sys
import time
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from base.config import Config
from base.logger import setup_logger

cfg = Config()
logger = setup_logger('RAG.api', log_file=cfg.log_file)

# ============================================================================
# FastAPI 应用实例化
# ============================================================================
app = FastAPI(
    title="RAG 智能问答系统 API",
    description="两级检索（FAQ关键词匹配 + RAG向量检索）的智能问答系统",
    version="1.0.0",
    docs_url="/docs",       # Swagger UI 文档地址
    redoc_url="/redoc",     # ReDoc 文档地址
)

# ============================================================================
# CORS 中间件（跨域资源共享）
# 为什么需要？前端（Streamlit/React/Vue）和后端（FastAPI）不在同一端口时，
# 浏览器会拦截跨域请求。CORS 中间件允许前端跨域调用后端 API。
# 生产环境应限制 origins 为具体域名，不要用 ["*"]。
# ============================================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # 允许的源，生产环境改为具体域名
    allow_credentials=True,         # 允许携带 Cookie
    allow_methods=["*"],            # 允许的 HTTP 方法
    allow_headers=["*"],            # 允许的请求头
)

# ============================================================================
# Pydantic 数据模型（请求/响应结构定义）
# Pydantic 是 FastAPI 的数据校验库，基于 Python 类型提示自动做：
#   1. 请求参数校验（类型、必填、范围）
#   2. 请求/响应序列化（dict → JSON）
#   3. 自动生成 API 文档（Swagger 中展示字段说明）
# ============================================================================


class AskRequest(BaseModel):
    """
    问答请求体

    【字段说明】
        question: str — 用户问题，必填
        enable_rerank: bool — 是否启用 Reranker 重排序，默认 True。
                            对于简单问题可以关掉以减少延迟。
        top_k: int — 粗排召回数量，默认 10。召回越多 Reranker 可选范围越大，
                   但也会增加延迟和 Token 消耗。
        rerank_top: int — 精排后保留数量，默认 3。最终喂给 LLM 的上下文条数。
    """
    question: str = Field(..., description="用户问题", min_length=1, max_length=2000)
    enable_rerank: bool = Field(True, description="是否启用Reranker重排序")
    top_k: int = Field(10, description="粗排召回数量", ge=1, le=50)
    rerank_top: int = Field(3, description="精排保留数量", ge=1, le=10)


class AskResponse(BaseModel):
    """
    问答响应体

    【字段说明】
        success: bool — 是否成功
        answer: str — LLM 生成的最终回答
        intent: str — 用户意图类型（chitchat/faq/knowledge/complaint/other）
        sources: list — 引用来源列表，每个元素包含文件名、相似度分数、文本预览
        elapsed: float — 总耗时（秒）
    """
    success: bool = Field(True, description="是否成功")
    answer: str = Field("", description="回答内容")
    intent: str = Field("knowledge", description="意图类型")
    sources: List[Dict[str, Any]] = Field([], description="引用来源")
    elapsed: float = Field(0.0, description="耗时（秒）")


class BuildIndexRequest(BaseModel):
    """
    构建索引请求体

    【字段说明】
        directory: str — 文档目录路径，必填。会递归加载目录下所有支持的文件。
        add_category: str — 类别标签，可选。如 'annual_reports'、'products'，不传自动从子目录名提取。
        drop_if_exists: bool — 是否删除旧 Collection 后重建，默认 False。
                              True 适合开发阶段反复测试。
    """
    directory: str = Field(..., description="文档目录路径", min_length=1)
    add_category: Optional[str] = Field(None, description="类别标签")
    drop_if_exists: bool = Field(False, description="是否删除旧Collection重建")


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = Field("ok", description="服务状态")
    version: str = Field("1.0.0", description="API版本")


# ============================================================================
# 懒加载：服务模块（首次调用时才初始化，避免启动时加载模型耗时过长）
# 面试点：为什么用懒加载？
#   答：生产环境的服务启动要求快（K8s 探活有时间限制），模型加载可能
#       需要几十秒。用懒加载让服务先启动，第一个请求时再加载模型。
# ============================================================================
_rag_service = None
_initialized = False


def _get_rag_service():
    """获取 RAG 服务实例（懒加载，线程安全）"""
    global _rag_service, _initialized
    if not _initialized:
        from rag.rag_service import RAGService
        _rag_service = RAGService()
        _initialized = True
        logger.info("RAG 服务已初始化")
    return _rag_service


# ============================================================================
# API 端点
# ============================================================================


@app.get("/", response_model=dict)
async def root():
    """
    服务信息

    【返回值】
        { "service": "RAG 智能问答系统", "version": "1.0.0", "docs": "/docs" }
    """
    return {
        "service": "RAG 智能问答系统",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "health": "/health",
            "ask": "POST /api/v1/ask",
            "build_index": "POST /api/v1/build_index",
        },
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    健康检查

    【用途】
    - K8s/Docker 探活：定期调用此接口判断服务是否存活
    - 负载均衡器：根据健康状态决定是否将流量转发到此实例
    - 监控告警：服务不可用时触发告警

    【返回值】
        { "status": "ok", "version": "1.0.0" }
    """
    return HealthResponse(status="ok", version="1.0.0")


@app.post("/api/v1/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    """
    智能问答接口（核心 API）

    【请求示例】
        POST /api/v1/ask
        {
            "question": "兴业银行2024年净利润和营收分别是多少？",
            "enable_rerank": true,
            "top_k": 10,
            "rerank_top": 3
        }

    【响应示例】
        {
            "success": true,
            "answer": "根据兴业银行2024年年度报告，该行实现营业收入...",
            "intent": "knowledge",
            "sources": [
                {"file_name": "兴业银行_2024年年度报告.pdf", "score": 0.92, "text": "..."},
                ...
            ],
            "elapsed": 1.23
        }

    【内部流程】
        1. 意图识别 → 判断是闲聊/FAQ/知识问答
        2. 闲聊 → 直接回复（不检索，不调 Reranker）
        3. FAQ  → 查 MySQL 关键词匹配
        4. 知识问答 → 向量检索 → Reranker 重排序 → LLM 生成回答
    """
    start = time.time()

    try:
        rag = _get_rag_service()
        result = rag.ask(
            question=req.question,
            top_k=req.top_k,
            rerank_top=req.rerank_top,
            enable_rerank=req.enable_rerank,
        )
        elapsed = round(time.time() - start, 3)

        return AskResponse(
            success=True,
            answer=result['answer'],
            intent=result.get('intent', 'knowledge'),
            sources=result.get('sources', []),
            elapsed=elapsed,
        )
    except Exception as e:
        logger.error("问答失败：%s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/build_index")
async def build_index(req: BuildIndexRequest):
    """
    构建向量索引

    【请求示例】
        POST /api/v1/build_index
        {
            "directory": "data/finance/",
            "add_category": "annual_reports",
            "drop_if_exists": true
        }

    【注意事项】
    - 此操作可能耗时较长（取决于文档数量），建议异步执行
    - drop_if_exists=true 会删除旧数据，不可逆
    """
    try:
        from rag.pipeline import build_vector_store

        if not os.path.exists(req.directory):
            raise HTTPException(status_code=400, detail=f"目录不存在：{req.directory}")

        store = build_vector_store(
            directory=req.directory,
            add_category=req.add_category,
            drop_if_exists=req.drop_if_exists,
        )
        store.close()
        return {"success": True, "message": "索引构建完成"}
    except Exception as e:
        logger.error("构建索引失败：%s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


class EvalRequest(BaseModel):
    """
    评估请求体

    【字段说明】
        test_data: List[Dict] — 可选，自定义测试数据，不传则使用默认数据集
        top_k: int — 检索返回数量，默认 5
        enable_generation_eval: bool — 是否启用生成评估（LLM-as-Judge），默认 True
    """
    test_data: Optional[List[Dict[str, Any]]] = Field(None, description="自定义测试数据")
    top_k: int = Field(5, description="检索返回数量", ge=1, le=20)
    enable_generation_eval: bool = Field(True, description="是否启用生成评估")


class EvalResponse(BaseModel):
    """评估响应体"""
    success: bool = Field(True, description="是否成功")
    report: str = Field("", description="评估报告")
    retrieval_metrics: Dict[str, Any] = Field({}, description="检索指标汇总")
    generation_metrics: Dict[str, Any] = Field({}, description="生成指标汇总")


@app.post("/api/v1/evaluate", response_model=EvalResponse)
async def evaluate(req: EvalRequest):
    """
    RAG 系统评估接口

    【请求示例】
        POST /api/v1/evaluate
        {
            "top_k": 5,
            "enable_generation_eval": true
        }

    【响应示例】
        {
            "success": true,
            "report": "评估报告文本...",
            "retrieval_metrics": {"hit_rate_5": 0.8, "mrr": 0.65, ...},
            "generation_metrics": {"faithfulness": 4.2, ...}
        }

    【注意事项】
    - enable_generation_eval=true 时需调用 LLM，耗时较长（约 30-60 秒）
    - 仅评估检索（enable_generation_eval=false）速度很快（约 1-3 秒）
    """
    try:
        from rag.evaluation import RAGEvaluator

        evaluator = RAGEvaluator()
        test_data = req.test_data if req.test_data else None

        # 检索评估
        retrieval_result = evaluator.evaluate_retrieval_only(test_data, req.top_k)

        gen_metrics = {}
        if req.enable_generation_eval:
            gen_result = evaluator.evaluate_generation(test_data or RAGEvaluator.generate_default_test_data())
            gen_metrics = {
                "faithfulness": gen_result.get("faithfulness", 0),
                "answer_relevance": gen_result.get("answer_relevance", 0),
                "context_relevance": gen_result.get("context_relevance", 0),
            }

        retrieval_metrics = {
            "hit_rate_3": retrieval_result["hit_rate_3"],
            "hit_rate_5": retrieval_result["hit_rate_5"],
            "recall_3": retrieval_result["recall_3"],
            "recall_5": retrieval_result["recall_5"],
            "mrr": retrieval_result["mrr"],
            "precision_3": retrieval_result["precision_3"],
            "precision_5": retrieval_result["precision_5"],
        }

        # 生成简化报告
        report_lines = [
            "=" * 60,
            "RAG 系统评估报告",
            "=" * 60,
            f"测试用例数：{len(test_data or RAGEvaluator.generate_default_test_data())}",
            "",
            "【检索质量】",
            f"  HitRate@3: {retrieval_metrics['hit_rate_3']:.2%}",
            f"  HitRate@5: {retrieval_metrics['hit_rate_5']:.2%}",
            f"  Recall@3:  {retrieval_metrics['recall_3']:.2%}",
            f"  Recall@5:  {retrieval_metrics['recall_5']:.2%}",
            f"  MRR:       {retrieval_metrics['mrr']:.2%}",
            f"  Precision@3: {retrieval_metrics['precision_3']:.2%}",
            f"  Precision@5: {retrieval_metrics['precision_5']:.2%}",
        ]

        if gen_metrics:
            report_lines += [
                "",
                "【生成质量】（LLM-as-Judge，1-5分制）",
                f"  忠实度:       {gen_metrics['faithfulness']:.1f}",
                f"  回答相关性:   {gen_metrics['answer_relevance']:.1f}",
                f"  上下文相关性: {gen_metrics['context_relevance']:.1f}",
            ]

        evaluator.close()

        return EvalResponse(
            success=True,
            report="\n".join(report_lines),
            retrieval_metrics=retrieval_metrics,
            generation_metrics=gen_metrics,
        )
    except Exception as e:
        logger.error("评估失败：%s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# 主入口：直接运行 python app/api.py 启动服务
# ============================================================================
if __name__ == '__main__':
    import uvicorn

    logger.info("启动 FastAPI 服务...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )