# ============================================================================
# 模块：app/ui.py
# Streamlit 可视化界面
# ============================================================================
# 【Streamlit 是什么？】
# Streamlit 是一个纯 Python 的 Web 应用框架，用于快速构建数据/ML 应用。
# 特点：
#   - 纯 Python：不需要写 HTML/CSS/JS，全部用 Python 代码
#   - 实时刷新：代码修改后自动刷新页面
#   - 组件丰富：文本框、按钮、图表、文件上传等开箱即用
#   - 适合场景：快速原型、内部工具、机器学习 Demo
#
# 缺点：不适合做复杂的前端交互（多层嵌套、自定义动画），
#       生产级前端建议用 React/Vue + FastAPI 后端。
#
# 【启动方式】
#   streamlit run app/ui.py
#
# 【Streamlit 常用 API 速查】
#   st.title("标题")                — 页面标题
#   st.markdown("**粗体**")         — Markdown 渲染
#   st.text_input("标签")           — 单行文本输入
#   st.chat_input("提示")           — 聊天输入框（ChatGPT 风格）
#   st.chat_message("user")         — 聊天消息气泡
#   st.button("按钮")               — 按钮
#   st.sidebar.xxx()                — 侧边栏组件
#   st.spinner("加载中...")         — 加载动画
#   st.expander("展开")             — 可折叠区域
#   st.columns(n)                   — 多列布局
#   st.session_state                — 会话状态（跨 rerun 保持）
# ============================================================================

import os
import sys
import time
from typing import List, Dict

import streamlit as st
import pandas as pd

from base.config import Config
from base.logger import setup_logger

cfg = Config()
logger = setup_logger('RAG.ui', log_file=cfg.log_file)

# ============================================================================
# 页面配置
# ============================================================================
st.set_page_config(
    page_title="RAG 智能问答系统",
    page_icon="🤖",
    layout="wide",  # 宽屏布局
    initial_sidebar_state="expanded",  # 侧边栏默认展开
)

# ============================================================================
# 懒加载 RAG 服务
# ============================================================================
@st.cache_resource
def load_rag_service():
    """
    加载 RAG 服务（只加载一次，后续使用缓存）

    @st.cache_resource 是 Streamlit 的缓存装饰器，用于缓存不可序列化的对象
    （如模型实例、数据库连接）。与 @st.cache_data 的区别：
    - @st.cache_data：缓存数据（dict/list/DataFrame），可序列化
    - @st.cache_resource：缓存资源（模型、连接、文件句柄），不可序列化
    """
    from rag.rag_service import RAGService
    logger.info("正在加载 RAG 服务...")
    return RAGService()


# ============================================================================
# 检索可视化渲染函数
# ============================================================================
def _render_retrieval_viz(viz_data: dict):
    """
    渲染检索过程可视化：相似度柱状图 + 召回来源分布 + 流程说明

    面试场景下，面试官可以直观看到 RAG 检索的全过程：
    - 哪些文档被召回
    - 相似度分数分布
    - 召回来源分布
    """
    sources = viz_data.get("sources", [])
    intent = viz_data.get("intent", "knowledge")
    rewritten = viz_data.get("rewritten_query")

    if not sources:
        st.info("未检索到相关文档")
        return

    # 流程说明
    if rewritten:
        st.caption(f"🔄 Query改写：{rewritten}")
    st.caption(f"🎯 意图识别：{intent}")

    # 相似度柱状图
    st.markdown("**📈 文档相似度分布**")
    df = pd.DataFrame([
        {
            "文档": f"{s['file_name'][:20]}...",
            "相似度": s.get('score', 0),
        }
        for s in sources
    ])
    st.bar_chart(df.set_index("文档"), use_container_width=True, height=200)

    # 召回来源分布
    st.markdown("**📂 召回来源分布**")
    source_counts = {}
    for s in sources:
        name = s.get('file_name', '未知')
        source_counts[name] = source_counts.get(name, 0) + 1

    col1, col2 = st.columns(2)
    with col1:
        st.metric("召回文档数", len(sources))
    with col2:
        st.metric("来源文件数", len(source_counts))

    # 相似度评级
    scores = [s.get('score', 0) for s in sources]
    if scores:
        avg_score = sum(scores) / len(scores)
        if avg_score > 0.8:
            st.success(f"✅ 平均相似度：{avg_score:.4f} — 检索质量优秀")
        elif avg_score > 0.6:
            st.warning(f"⚠️ 平均相似度：{avg_score:.4f} — 检索质量良好")
        else:
            st.error(f"❌ 平均相似度：{avg_score:.4f} — 检索质量需改进")


# ============================================================================
# 侧边栏
# ============================================================================
with st.sidebar:
    st.title("⚙️ 设置")
    st.markdown("---")

    # 检索参数
    st.subheader("🔍 检索参数")
    top_k = st.slider(
        "粗排召回数量",
        min_value=1, max_value=50, value=10, step=1,
        help="向量检索返回的候选数量。值越大召回越全，但延迟越高。"
    )
    rerank_top = st.slider(
        "精排保留数量",
        min_value=1, max_value=10, value=3, step=1,
        help="Reranker 重排序后保留的条数，最终喂给 LLM 的上下文数量。"
    )
    enable_rerank = st.checkbox(
        "启用 Reranker 重排序",
        value=True,
        help="勾选后会对检索结果做精排，提高准确性，但会增加约 0.5-1 秒延迟。"
    )

    st.markdown("---")
    st.subheader("📊 系统信息")
    st.markdown(f"- **粗排量**: {top_k}")
    st.markdown(f"- **精排量**: {rerank_top}")
    st.markdown(f"- **Reranker**: {'已启用' if enable_rerank else '已关闭'}")
    st.markdown(f"- **模型**: {cfg.llm_model}")

    st.markdown("---")

    # 可视化开关
    st.subheader("🎨 可视化")
    show_retrieval_viz = st.checkbox("检索过程可视化", value=True,
        help="用图表展示检索结果的相似度分布和召回来源")
    show_flow = st.checkbox("展示处理流程", value=False,
        help="在回答中标注每一步处理耗时")

    st.markdown("---")
    st.caption("Powered by LangChain + Milvus + BGE-M3")

# ============================================================================
# 主页面
# ============================================================================
st.title("🤖 RAG 智能问答系统")
st.markdown("*两级检索：FAQ 关键词匹配 → RAG 向量检索 → LLM 生成回答*")

# 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = []

# 渲染历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 引用来源"):
                for src in msg["sources"]:
                    score = src.get('score', 0)
                    color = '🟢' if score > 0.8 else ('🟡' if score > 0.6 else '🟠')
                    st.caption(f"{color} **{src['file_name']}** (相似度: {score:.4f})")
                    st.text(src.get('text', '')[:300])
                    st.markdown("---")
        if msg.get("retrieval_viz"):
            with st.expander("📊 检索过程可视化"):
                _render_retrieval_viz(msg["retrieval_viz"])

# 输入框
if prompt := st.chat_input("请输入您的问题..."):
    # 显示用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 生成回答
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            try:
                rag = load_rag_service()
                start = time.time()

                # 各阶段计时
                t0 = time.time()
                result = rag.ask(
                    question=prompt,
                    top_k=top_k,
                    rerank_top=rerank_top,
                    enable_rerank=enable_rerank,
                )
                t1 = time.time()
                elapsed = t1 - start

                answer = result.get('answer', '抱歉，处理出错。')
                sources = result.get('sources', [])
                intent = result.get('intent', 'knowledge')
                contexts = result.get('contexts', [])
                rewritten = result.get('rewritten_query')

                # 检索可视化数据
                retrieval_viz = None
                if show_retrieval_viz and sources:
                    retrieval_viz = {
                        "sources": sources,
                        "intent": intent,
                        "rewritten_query": rewritten,
                        "contexts": contexts,
                    }

                # 显示回答
                st.markdown(answer)

                # 流程耗时展示
                if show_flow:
                    st.caption(
                        f"⏱️ 总耗时 {elapsed:.2f}s | "
                        f"意图识别 → {intent} | "
                        f"引用 {len(sources)} 条"
                    )
                else:
                    st.caption(
                        f"⏱️ 耗时 {elapsed:.2f}s | "
                        f"意图: {intent} | "
                        f"引用 {len(sources)} 条"
                    )

                # 检索过程可视化
                if retrieval_viz:
                    with st.expander("📊 检索过程可视化", expanded=True):
                        _render_retrieval_viz(retrieval_viz)

                # 显示引用来源
                if sources:
                    with st.expander("📚 引用来源详情"):
                        for i, src in enumerate(sources):
                            score = src.get('score', 0)
                            color = '🟢' if score > 0.8 else ('🟡' if score > 0.6 else '🟠')
                            st.markdown(f"{color} **{i+1}. {src['file_name']}** (相似度: {score:.4f})")
                            st.text(src.get('text', '')[:300])
                            st.markdown("---")

                # 保存到历史
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "retrieval_viz": retrieval_viz,
                })

            except Exception as e:
                st.error(f"出错了：{e}")
                logger.error("UI 问答失败：%s", e, exc_info=True)

# 清空按钮
if st.sidebar.button("🗑️ 清空对话"):
    st.session_state.messages = []
    st.rerun()