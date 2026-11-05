# ============================================================================
# 模块：rag/loaders/markdown_loader.py
# Markdown 文档加载器（基于 UnstructuredMarkdownLoader）
# ============================================================================
# 【包的作用】
# UnstructuredMarkdownLoader 是 LangChain 社区提供的 Markdown 加载器，
# 底层使用 unstructured 库解析 Markdown 文件，能识别标题层级、代码块、列表等结构。
#
# 【UnstructuredMarkdownLoader 核心 API】
#   UnstructuredMarkdownLoader(file_path) → 创建加载器
#   loader.load() → 加载 MD 文件，返回 List[Document]
#
# 【与 TextLoader 的区别】
#   TextLoader：将整个文件当作纯文本，不做结构化解析
#   UnstructuredMarkdownLoader：识别 Markdown 语法（# ## ``` 等），
#                               按标题/段落拆分，保留文档结构
#
# 【使用示例】
#   loader = MarkdownLoader("data/README.md")
#   docs = loader.load()
# ============================================================================

from typing import List

from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain_core.documents import Document

from rag.loaders.base import BaseDocumentLoader


class MarkdownLoader(BaseDocumentLoader):
    """
    Markdown 文档加载器

    【SUPPORTED_EXTENSIONS】
    同时支持 .md 和 .markdown 两种扩展名。
    """

    SUPPORTED_EXTENSIONS = {'.md', '.markdown'}

    def load(self) -> List[Document]:
        """
        加载 Markdown 文件，返回 Document 列表

        【返回值】
            List[Document] — 按标题/段落拆分的 Document 列表
                             通过 _add_metadata 自动添加 file_type='md' 和 file_name
        """
        loader = UnstructuredMarkdownLoader(self.file_path)
        docs = loader.load()
        return self._add_metadata(docs, 'md')