# ============================================================================
# rag/loaders — 文档加载器集合
# ============================================================================
# 【作用】
#   提供统一的文档加载接口，支持多种文件格式：
#   - PDFLoader：PDF 文档加载（含 OCR 识别文字层为空时的回退）
#   - MarkdownLoader：Markdown 文件加载
#   - PlainTextLoader：纯文本文件加载
#   - CSVQuestionLoader：CSV 格式的 FAQ 问答对加载
#   - ImageLoader：图片文件加载（OCR 文字提取）
#
# 【使用方式】
#   from rag.loaders import PDFLoader
#   documents = PDFLoader.load("path/to/file.pdf")
# ============================================================================

from rag.loaders.base import BaseDocumentLoader
from rag.loaders.pdf_loader import PDFLoader
from rag.loaders.markdown_loader import MarkdownLoader
from rag.loaders.text_loader import PlainTextLoader
from rag.loaders.csv_loader import CSVQuestionLoader
from rag.loaders.image_loader import ImageLoader