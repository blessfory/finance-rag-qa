# ============================================================================
# 模块：rag/loaders/csv_loader.py
# CSV 问答对加载器（基于 CSVLoader）
# ============================================================================
# 【包的作用】
# CSVLoader 是 LangChain 提供的 CSV 文件加载器，将 CSV 的每一行作为一个 Document。
#
# 【CSVLoader 核心 API】
#   CSVLoader(file_path, csv_args, encoding) → 创建加载器
#     csv_args: dict — 传递给 Python csv 模块的参数
#       delimiter: str — 分隔符，默认 ','（逗号）
#       quotechar: str — 引号字符，默认 '"'（双引号）
#     encoding: str — 文件编码，CSV 通常用 'utf-8-sig'（带 BOM 的 UTF-8）
#   loader.load() → 返回 List[Document]
#
# 【CSVLoader 工作原理】
#   1. 读取 CSV 文件
#   2. 每一行生成一个 Document，page_content 为整行内容的拼接
#   3. 每一列的列名作为 metadata 的 key
#
# 【为什么 CSV 用 'utf-8-sig' 编码？】
#   Windows 的 Excel 保存 CSV 时会在文件开头加上 BOM（Byte Order Mark，字节序标记）。
#   'utf-8-sig' 会自动识别并跳过 BOM，而 'utf-8' 会把 BOM 当作内容的一部分，
#   导致第一列列名前面出现不可见字符（如 \ufeff）。
#
# 【使用示例】
#   loader = CSVQuestionLoader("data/FAQ.csv")
#   docs = loader.load()
#   # 假设 CSV 有两列：question, answer
#   # doc.page_content = "question: Python是什么？\nanswer: Python是一种语言"
#   # doc.metadata = {'question': 'Python是什么？', 'answer': 'Python是一种语言', ...}
# ============================================================================

from typing import List

from langchain_community.document_loaders import CSVLoader
from langchain_core.documents import Document

from rag.loaders.base import BaseDocumentLoader


class CSVQuestionLoader(BaseDocumentLoader):
    """
    CSV 问答对加载器

    【命名说明】
    叫 CSVQuestionLoader 而不是 CSVLoader，是因为本项目专门用它加载"问答对"格式的 CSV，
    如 FAQ 知识库。通用 CSV 数据加载也可以用这个类。

    【SUPPORTED_EXTENSIONS】
    只支持 .csv 扩展名。
    """

    SUPPORTED_EXTENSIONS = {'.csv'}

    def __init__(self, file_path: str, encoding: str = 'utf-8-sig'):
        """
        【参数】
            file_path: str — CSV 文件路径
            encoding: str — 文件编码，默认 'utf-8-sig'（兼容 Excel 导出的 CSV）
        """
        super().__init__(file_path)
        self.encoding = encoding

    def load(self) -> List[Document]:
        """
        加载 CSV 文件，返回 Document 列表

        【返回值】
            List[Document] — 每行一个 Document，page_content 为整行拼接，
                             metadata 包含每列的列名和值
                             通过 _add_metadata 自动添加 file_type='csv' 和 file_name

        【csv_args 参数说明】
            delimiter: ',' — 逗号分隔（标准 CSV）
            quotechar: '"' — 字段包含逗号时用双引号包裹，如 "Python, Java, C++"
        """
        loader = CSVLoader(
            self.file_path,
            csv_args={'delimiter': ',', 'quotechar': '"'},
            encoding=self.encoding,
        )
        docs = loader.load()
        return self._add_metadata(docs, 'csv')