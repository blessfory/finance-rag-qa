# ============================================================================
# 模块：rag/loaders/base.py
# 文档加载器抽象基类（ABC — Abstract Base Class）
# ============================================================================
# 【什么是 ABC（抽象基类）？】
# Python 的 abc 模块提供了定义"接口规范"的能力。
# 抽象基类不能直接实例化，只能被继承，子类必须实现所有 @abstractmethod 方法。
# 作用：强制所有子类遵循统一的接口，保证多态替换。
#
# 【类比理解】
# 就像"动物"是一个抽象概念，你不能直接 new 一个"动物"，但可以 new "猫"和"狗"。
# 猫和狗都必须实现"叫"这个方法，但各自的叫法不同。
#
#   BaseDocumentLoader（抽象，不能实例化）
#   ├── PDFLoader       → load() 用 PyPDFLoader 读取 PDF
#   ├── MarkdownLoader  → load() 用 UnstructuredMarkdownLoader 读取 MD
#   ├── PlainTextLoader → load() 用 TextLoader 读取 TXT
#   ├── CSVQuestionLoader → load() 用 CSVLoader 读取 CSV
#   └── ImageLoader     → load() 用 PaddleOCR 识别图片文字
#
# 【abc 模块核心 API】
#   ABC               → 所有抽象基类的父类，继承它才能用 @abstractmethod
#   @abstractmethod   → 装饰器，标记必须被子类实现的方法
#
# 【os 模块核心 API】
#   os.path.basename(path) → 取文件名，如 "a/b/c.pdf" → "c.pdf"
#   os.path.abspath(path)  → 转绝对路径
#   os.path.exists(path)   → 判断文件是否存在
#   os.path.splitext(path) → 分离扩展名，如 ("a/b/c", ".pdf")
#   os.path.dirname(path)  → 取目录名，如 "a/b/c.pdf" → "a/b"
# ============================================================================

import os
from abc import ABC, abstractmethod
from typing import List

from langchain_core.documents import Document


class BaseDocumentLoader(ABC):
    """
    文档加载器抽象基类：所有 Loader 的父类，定义统一接口规范

    【设计模式：模板方法（Template Method）】
    - 公共方法 _add_metadata() 在基类实现，所有子类复用
    - 抽象方法 load() 由各子类自己实现，基类只定义接口

    【Document 对象结构】
    Document 是 LangChain 的核心数据类型，表示一个文档片段：
    - page_content: str — 文档的文本内容
    - metadata: dict — 元数据，如：
        - file_name: 源文件名
        - file_type: 文件类型（pdf/md/txt/csv/image）
        - category: 类别标签（如 'ai'、'java'）
        - page: 页码（PDF 加载器自动添加）
        - source: 文件路径（LangChain 加载器自动添加）
    """

    def __init__(self, file_path: str):
        """
        【参数】
            file_path: str — 文件的绝对路径或相对路径
        """
        self.file_path = file_path
        self.file_name = os.path.basename(file_path)

    @abstractmethod
    def load(self) -> List[Document]:
        """
        加载文档的抽象方法（子类必须实现）

        【返回值】
            List[Document] — 加载后的 Document 列表

        【子类实现示例】
            class PDFLoader(BaseDocumentLoader):
                def load(self) -> List[Document]:
                    loader = PyPDFLoader(self.file_path)
                    return loader.load()
        """
        pass

    def _add_metadata(self, docs: List[Document], file_type: str) -> List[Document]:
        """
        为 Document 列表统一添加元数据（公共方法，所有子类复用）

        【参数】
            docs: List[Document] — 加载后的 Document 列表
            file_type: str — 文件类型标识，如 'pdf'、'md'、'txt'

        【返回值】
            List[Document] — 添加了元数据的 Document 列表（原地修改 + 返回）

        【添加的元数据字段】
            - file_type: 文件类型，用于后续分块策略选择
            - file_name: 文件名，用于检索结果展示来源
        """
        for doc in docs:
            doc.metadata['file_type'] = file_type
            doc.metadata['file_name'] = self.file_name
        return docs