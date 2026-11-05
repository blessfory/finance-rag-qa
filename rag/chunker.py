# ============================================================================
# 模块：rag/chunker.py
# 父子分块（Parent-Child Chunking）
# ============================================================================
# 【什么是父子分块？】
# 文档分块时同时生成"大块"（父块）和"小块"（子块）：
# - 子块：较短，向量化后存入 Milvus，用于语义检索（精度高）
# - 父块：较长，不存向量库，仅在检索到对应子块时提供完整上下文（喂给 LLM）
#
# 【为什么需要父子分块？】
# 传统单一尺寸分块的问题：
#   - 块太小 → 检索更精准，但 LLM 看不到完整上下文，回答断章取义
#   - 块太大 → LLM 有完整上下文，但检索精度低（噪声多，不相关的内容也被召回）
# 父子分块同时解决了这两个问题：小粒度检索 + 大粒度回答。
#
# 【父子分块可视化】
#   原始文档（2000字）
#   ┌──────────────────────────────────────────────────────────────────┐
#   │ 第一章 人工智能概述                                                │
#   │ 人工智能是...机器学习是...深度学习是...                              │
#   │ 神经网络是...Transformer是...GPT是...                              │
#   │ ...                                                              │
#   └──────────────────────────────────────────────────────────────────┘
#                    │
#                    ▼ ParentChildChunker.split()
#   ┌──────────────────────────────────────────────────────────────────┐
#   │ 父块（喂给 LLM）                   子块（存入 Milvus 检索）         │
#   │ ┌────────────────────────────┐    ┌────────────────┐             │
#   │ │ 父块0（800字）              │    │ 子块0（200字）   │───→ 向量    │
#   │ │ 人工智能...神经网络...       │◄───│ 人工智能...      │             │
#   │ │ Transformer...GPT...       │    ├────────────────┤             │
#   │ └────────────────────────────┘    │ 子块1（200字）   │───→ 向量    │
#   │                                    │ 机器学习...      │             │
#   │ ┌────────────────────────────┐    ├────────────────┤             │
#   │ │ 父块1（800字）              │    │ 子块2（200字）   │───→ 向量    │
#   │ │ ...                        │◄───│ 深度学习...      │             │
#   │ └────────────────────────────┘    ├────────────────┤             │
#   │                                    │ 子块3（200字）   │───→ 向量    │
#   │ ┌────────────────────────────┐    │ 神经网络...      │             │
#   │ │ 父块2（400字）              │    └────────────────┘             │
#   │ │ ...                        │◄─── 最多 4 个子块                  │
#   │ └────────────────────────────┘                                    │
#   └──────────────────────────────────────────────────────────────────┘
#
#   检索时：用户问题 → 匹配子块2 → 根据 parent_index 找到父块0 → 父块0喂LLM
#
# 【LangChain 的 RecursiveCharacterTextSplitter】
# 按优先级递归拆分文本，直到满足 chunk_size：
#   1. 先按 "\n\n"（段落）拆分
#   2. 再按 "\n"（换行）拆分
#   3. 再按 "。"、"！"、"？"（中文句子）拆分
#   4. 再按 "；"（分号）拆分
#   5. 再按 "，"（逗号）拆分
#   6. 再按 " "（空格）拆分
#   7. 最后按字符拆分
# 这样能尽量保持语义完整性，避免在句子中间截断。
#
# 【RecursiveCharacterTextSplitter 常用参数】
#   chunk_size: int — 每个 chunk 的最大字符数（近似值，以分隔符实际位置为准）
#   chunk_overlap: int — 相邻 chunk 重叠的字符数，避免信息在边界丢失
#   separators: List[str] — 分隔符优先级列表，按顺序尝试拆分
#   length_function: Callable — 计算文本长度的函数，默认 len
#   is_separator_regex: bool — 分隔符是否为正则表达式，默认 False
#
#   【常用方法】
#   split_documents(docs) → List[Document]    拆分 Document 列表
#   split_text(text) → List[str]              拆分单个字符串
# ============================================================================

from typing import List, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from base.config import Config
from base.logger import setup_logger

cfg = Config()
logger = setup_logger('RAG.chunker', log_file=cfg.log_file)

PARENT_SIZE = cfg.parent_chunk_size
CHILD_SIZE = cfg.child_chunk_size
OVERLAP = cfg.chunk_overlap


class ParentChildChunker:
    """
    父子分块器：大块喂 LLM，小块做检索

    【核心设计】
        - 父块（大块）：parent_size 大小，提供完整上下文给 LLM
        - 子块（小块）：child_size 大小，存入向量库做语义检索
        - 每个子块通过 metadata['parent_index'] 关联到对应的父块

    【使用方式】
        chunker = ParentChildChunker(parent_size=800, child_size=200, overlap=50)
        parent_chunks, child_chunks = chunker.split(documents)

        # parent_chunks[i] 对应 child_chunks 中所有 parent_index == i 的子块
        # 检索到子块后，通过 parent_index 获取父块内容喂给 LLM

    【参数选择建议】
        parent_size: 800-2000 字符。太小 → 上下文不完整；太大 → 超过 LLM 上下文窗口
        child_size:  200-500 字符。太小 → 语义不完整；太大 → 检索精度下降
        overlap:     50-150 字符。保证边界信息不丢失
    """

    def __init__(
        self,
        parent_size: int = PARENT_SIZE,
        child_size: int = CHILD_SIZE,
        overlap: int = OVERLAP,
    ):
        """
        【参数】
            parent_size: int — 父块大小（字符数），默认从 config.ini 读取（parent_chunk_size）
            child_size: int — 子块大小（字符数），默认从 config.ini 读取（child_chunk_size）
            overlap: int — 相邻 chunk 重叠字符数，默认从 config.ini 读取（chunk_overlap）
        """
        self.parent_size = parent_size
        self.child_size = child_size
        self.overlap = overlap

        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_size, chunk_overlap=overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        )
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_size, chunk_overlap=overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        )

    def split(self, docs: List[Document]) -> Tuple[List[Document], List[Document]]:
        """
        将文档列表拆分为父子块

        【参数】
            docs: List[Document] — 要拆分的文档列表（来自 DocumentLoader 加载的结果）

        【返回值】
            Tuple[List[Document], List[Document]]：
            - 第一个元素：父块列表（大块，喂给 LLM）
            - 第二个元素：子块列表（小块，存入向量库）

        【内部流程】
            1. 用 parent_splitter 将文档拆分成父块（大块）
            2. 为每个父块设置 parent_index 元数据
            3. 用 child_splitter 将每个父块再拆分成子块（小块）
            4. 每个子块继承父块的元数据，并标记 parent_index 关联关系

        【metadata 字段说明】
            每个子块 Document 的 metadata 包含：
            - parent_index: int — 指向父块的索引（0-based）
            - child_index: int — 子块在父块内的序号（0-based）
            - chunk_size: int — 子块大小配置
            - file_name: str — 源文件名（继承自父块）
            - file_type: str — 文件类型（继承自父块）
            - category: str — 类别标签（继承自父块）
            - ... 以及其他父块传入的元数据
        """
        parent_chunks = self.parent_splitter.split_documents(docs)

        child_chunks = []
        for parent_idx, parent_chunk in enumerate(parent_chunks):
            parent_chunk.metadata['parent_index'] = parent_idx
            child_texts = self.child_splitter.split_text(parent_chunk.page_content)
            for child_idx, child_text in enumerate(child_texts):
                child_chunk = Document(
                    page_content=child_text,
                    metadata={
                        **parent_chunk.metadata,
                        'parent_index': parent_idx,
                        'child_index': child_idx,
                        'chunk_size': self.child_size,
                    },
                )
                child_chunks.append(child_chunk)

        return parent_chunks, child_chunks

if __name__ == '__main__':
    import os
    from rag.document_loader import DocumentLoader

    dl = DocumentLoader()
    file_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'finance', 'annual_reports', '兴业银行_2025年年度报告.pdf')
    docs = dl.load(file_path, add_category='annual_reports')

    chunker = ParentChildChunker()
    parent_chunks, child_chunks = chunker.split(docs)

    logger.info("父块数量：%d", len(parent_chunks))
    logger.info("子块数量：%d", len(child_chunks))
    logger.info("第一个父块预览：%s...", parent_chunks[0].page_content[:100])
    logger.info("第一个子块预览：%s...", child_chunks[0].page_content[:100])
    logger.info("该子块属于父块：%s", child_chunks[0].metadata['parent_index'])