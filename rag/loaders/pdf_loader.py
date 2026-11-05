# ============================================================================
# 模块：rag/loaders/pdf_loader.py
# PDF 文档加载器（基于 PyMuPDFLoader）
# ============================================================================
# 【包的作用】
# PyMuPDFLoader 是 LangChain 社区提供的 PDF 加载器，底层使用 PyMuPDF（fitz）解析 PDF。
# 相比 PyPDF2，PyMuPDF 速度快 4 倍，中文排版更好，表格保留行列结构。
#
# 【PyMuPDFLoader 核心 API】
#   PyMuPDFLoader(file_path)        → 创建加载器，传入 PDF 文件路径
#   loader.load()                   → 加载 PDF，返回 List[Document]
#   loader.load_and_split(splitter) → 加载并直接分块（本项目不直接用，用父子分块替代）
#
# 【PyMuPDFLoader 工作原理】
#   1. 打开 PDF 文件，逐页提取文本
#   2. 每一页生成一个 Document 对象
#   3. Document.page_content = 该页的文本内容
#   4. Document.metadata = {'source': 文件路径, 'page': 页码}
#
# 【PyMuPDFLoader 的优势】
#   - 速度比 PyPDF2 快 4 倍
#   - 中文排版保留更完整，不会乱序
#   - 表格保留行列结构，不会被拆成散乱行
#   - 可提取图片，配合 OCR 处理扫描件
#   - 局限性：仍不能识别图片中的文字，扫描版需 OCR
#
# 【使用示例】
#   loader = PDFLoader("data/finance/annual_reports/兴业银行_2025年年度报告.pdf")
#   docs = loader.load()  # → List[Document]，每个元素是一页的内容
# ============================================================================

from typing import List

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document

from rag.loaders.base import BaseDocumentLoader


class PDFLoader(BaseDocumentLoader):
    """
    PDF 文档加载器（基于 PyMuPDF）

    【SUPPORTED_EXTENSIONS】
    类属性，定义支持的扩展名集合。DocumentLoader 门面类通过检查
    文件扩展名是否在此集合中来决定使用哪个 Loader。

    【为什么选 PyMuPDF 而不是 PyPDF2】
    PyMuPDF 底层是 MuPDF 引擎，C 语言实现，比 PyPDF2 快 4 倍。
    对中文排版、表格结构的保留效果更好，是金融文档 RAG 的首选方案。
    """

    SUPPORTED_EXTENSIONS = {'.pdf'}

    def load(self) -> List[Document]:
        """
        加载 PDF 文件，返回 Document 列表

        【返回值】
            List[Document] — 每页一个 Document，包含 page_content 和 metadata
                             通过 _add_metadata 自动添加 file_type='pdf' 和 file_name

        【PyMuPDFLoader.load() 返回值】
            List[Document]，每个 Document 的 metadata 包含：
            - source: str — PDF 文件路径
            - page: int — 页码（0-based）
        """
        loader = PyMuPDFLoader(self.file_path)
        docs = loader.load()
        return self._add_metadata(docs, 'pdf')