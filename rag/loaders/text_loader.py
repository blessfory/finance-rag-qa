# ============================================================================
# 模块：rag/loaders/text_loader.py
# 纯文本加载器（基于 TextLoader）
# ============================================================================
# 【包的作用】
# TextLoader 是 LangChain 提供的纯文本文件加载器，将整个文件内容读取为一个 Document。
#
# 【TextLoader 核心 API】
#   TextLoader(file_path, encoding='utf-8') → 创建加载器
#   loader.load() → 加载文本文件，返回 List[Document]（通常只有一个元素）
#
# 【TextLoader 工作原理】
#   1. 以指定编码打开文件
#   2. 读取全部内容 → Document.page_content
#   3. Document.metadata = {'source': 文件路径}
#
# 【支持的扩展名】
#   .txt, .text, .log, .py, .java, .xml, .json, .yaml, .yml
#   基本覆盖了所有纯文本格式。
#
# 【编码处理策略】
#   首先尝试 UTF-8 编码，失败后降级到 GBK（中文 Windows 常用编码）。
#   这是实际项目中非常常见的处理方式，因为中文文件可能用 GBK 保存。
#
# 【使用示例】
#   loader = PlainTextLoader("data/笔记.txt")
#   docs = loader.load()
# ============================================================================

from typing import List

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

from rag.loaders.base import BaseDocumentLoader


class PlainTextLoader(BaseDocumentLoader):
    """
    纯文本加载器

    【SUPPORTED_EXTENSIONS】
    支持常见的纯文本格式：.txt, .text, .log, .py, .java, .xml, .json, .yaml, .yml
    """

    SUPPORTED_EXTENSIONS = {'.txt', '.text', '.log', '.py', '.java', '.xml', '.json', '.yaml', '.yml'}

    def __init__(self, file_path: str, encoding: str = 'utf-8'):
        """
        【参数】
            file_path: str — 文件路径
            encoding: str — 文件编码，默认 'utf-8'
                          中文 Windows 系统下可能需要 'gbk'
        """
        super().__init__(file_path)
        self.encoding = encoding

    def load(self) -> List[Document]:
        """
        加载文本文件，返回 Document 列表

        【返回值】
            List[Document] — 通常只有一个元素，包含整个文件内容

        【编码降级策略】
        1. 先尝试 UTF-8 编码读取
        2. 如果遇到 UnicodeDecodeError（通常是 GBK 编码的中文文件），降级到 GBK
        3. 这种"先 UTF-8 后 GBK"的策略覆盖了 90% 的中文文本场景
        """
        try:
            loader = TextLoader(self.file_path, encoding=self.encoding)
            docs = loader.load()
        except UnicodeDecodeError:
            loader = TextLoader(self.file_path, encoding='gbk')
            docs = loader.load()
        return self._add_metadata(docs, 'text')