# ============================================================================
# 模块：rag/document_loader.py
# 文档加载门面（Facade Pattern）：根据扩展名，创建对应的 Loader 实例，调用 load()
# ============================================================================
# 【设计模式：门面模式（Facade Pattern）】
# 门面模式提供一个统一的入口，隐藏内部复杂的子系统。
# 调用者不需要知道 PDF、Markdown、CSV 各用哪个 Loader，只需要调用一个方法。
#
# 【与 chunker.py 的关系】
# 本模块负责：
#   1. 根据文件扩展名选择合适的 Loader（PDF/MD/TXT/CSV/Image）
#   2. 加载文档 → 得到原始 Document 列表
# chunker.py 负责：
#   3. 对本模块输出的 Document 做父子分块（大块喂 LLM + 小块检索）
#
# 分块逻辑全部交给 chunker.py，本模块只做纯粹的文档读取，各司其职。
#
# 【金融数据分类机制】
# data/finance/ 下的子目录名自动成为 category：
#   data/finance/annual_reports/   → category = "annual_reports"
#   data/finance/products/         → category = "products"
#   data/finance/industry_knowledge/ → category = "industry_knowledge"
# 检索时可通过 category 过滤，实现按数据源精准检索。
# ============================================================================

import os
from typing import List

from langchain_core.documents import Document

from base.config import Config
from base.logger import setup_logger
from rag.loaders import (
    PDFLoader,
    MarkdownLoader,
    PlainTextLoader,
    CSVQuestionLoader,
    ImageLoader,
)

cfg = Config()
logger = setup_logger('RAG.loader', log_file=cfg.log_file)


class DocumentLoader:
    """
    文档加载门面：根据文件扩展名，创建对应的 Loader 实例

    【使用方式】
        # 加载单个文件
        dl = DocumentLoader()
        file_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'finance', 'annual_reports', '兴业银行_2025年年度报告.pdf')
        docs = dl.load(file_path, add_category='annual_reports')

        # 加载整个目录（自动按子目录分类）
        docs = dl.load_directory("data/finance/")

    【add_category 的作用】
    为所有 Document 的 metadata 添加 'category' 字段，如 'annual_reports'、'products'。
    检索时可以通过 category 过滤，实现按类别检索（如只搜年报相关的文档）。
    """

    def load(self, file_path: str, add_category: str = None) -> List[Document]:
        """
        加载单个文件，返回原始 Document 列表（不做分块）

        【参数】
            file_path: str — 文件路径（相对路径会自动转绝对路径）
            add_category: str | None — 类别标签，如 'annual_reports'、'products'，None 表示不添加

        【返回值】
            List[Document] — 加载后的原始 Document 列表（分块交给 chunker.py）

        【内部流程】
            1. 路径处理：转绝对路径、检查文件是否存在
            2. 根据扩展名选择 Loader（PDF/MD/TXT/CSV/Image）
            3. 调用 Loader.load() 加载文档
            4. 添加 category 元数据

        【异常处理】
            - 文件不存在 → FileNotFoundError
            - 不支持的文件格式 → ValueError
        """
        if not os.path.isabs(file_path):
            file_path = os.path.abspath(file_path)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在：{file_path}")

        ext = os.path.splitext(file_path)[1].lower()

        if ext in PDFLoader.SUPPORTED_EXTENSIONS:
            docs = PDFLoader(file_path).load()
        elif ext in MarkdownLoader.SUPPORTED_EXTENSIONS:
            docs = MarkdownLoader(file_path).load()
        elif ext in PlainTextLoader.SUPPORTED_EXTENSIONS:
            docs = PlainTextLoader(file_path).load()
        elif ext in CSVQuestionLoader.SUPPORTED_EXTENSIONS:
            docs = CSVQuestionLoader(file_path).load()
        elif ext in ImageLoader.SUPPORTED_EXTENSIONS:
            docs = ImageLoader(file_path).load()
        else:
            raise ValueError(f"不支持的文件格式：{ext}")

        if add_category:
            for doc in docs:
                doc.metadata['category'] = add_category
        return docs

    def load_directory(self, dir_path: str, add_category: str = None) -> List[Document]:
        """
        加载整个目录下的所有支持文件，自动按子目录分类

        【参数】
            dir_path: str — 根目录路径（如 data/finance/）
            add_category: str | None — 全局类别标签，不传则自动从子目录名提取

        【分类逻辑】
            不传 add_category 时，自动将子目录名作为 category：
              data/finance/annual_reports/兴业银行_2025年报.pdf  → category="annual_reports"
              data/finance/products/产品手册.md                  → category="products"
            传入 add_category 时，所有文件统一使用该标签（兼容旧用法）。

        【内部流程】
            1. 收集所有支持的扩展名（取所有 Loader 的 SUPPORTED_EXTENSIONS 的并集）
            2. 用 os.walk() 递归遍历目录
            3. 对每个文件，自动检测其相对于根目录的子目录名作为 category
            4. 加载失败的文件会记录警告日志并跳过，不会中断整个流程
        """
        if not os.path.isabs(dir_path):
            dir_path = os.path.abspath(dir_path)

        all_docs = []
        all_exts = (
            PDFLoader.SUPPORTED_EXTENSIONS
            | MarkdownLoader.SUPPORTED_EXTENSIONS
            | PlainTextLoader.SUPPORTED_EXTENSIONS
            | CSVQuestionLoader.SUPPORTED_EXTENSIONS
            | ImageLoader.SUPPORTED_EXTENSIONS
        )

        for root, dirs, files in os.walk(dir_path):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in all_exts:
                    file_path = os.path.join(root, file)
                    try:
                        if add_category is not None:
                            category = add_category
                        else:
                            rel_dir = os.path.relpath(root, dir_path)
                            category = rel_dir.replace(os.sep, '_') if rel_dir != '.' else ''
                        docs = self.load(file_path, add_category=category)
                        all_docs.extend(docs)
                    except Exception as e:
                        logger.warning("跳过文件（加载失败）：%s，错误：%s", file_path, e)

        return all_docs


def load_document(file_path: str, **kwargs) -> List[Document]:
    """
    便捷函数：加载单个文件（无需手动创建 DocumentLoader 实例）

    【参数】
        file_path: str — 文件路径
        **kwargs — 传递给 DocumentLoader.load() 的其他参数，如 add_category='annual_reports'

    【返回值】
        List[Document] — 原始 Document 列表（分块交给 chunker.py）
    """
    return DocumentLoader().load(file_path, **kwargs)


def load_directory(dir_path: str, **kwargs) -> List[Document]:
    """
    便捷函数：加载整个目录（无需手动创建 DocumentLoader 实例）

    【参数】
        dir_path: str — 目录路径
        **kwargs — 传递给 DocumentLoader.load_directory() 的其他参数

    【返回值】
        List[Document] — 所有文件的原始 Document 合并后的列表
    """
    return DocumentLoader().load_directory(dir_path, **kwargs)


if __name__ == '__main__':
    dl = DocumentLoader()
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'finance')
    docs = dl.load_directory(data_dir)

    logger.info("总计加载 %d 个文档", len(docs))
    for i, doc in enumerate(docs[:3]):
        logger.info("--- 文档 %d ---", i)
        logger.info("来源：%s", doc.metadata.get('file_name'))
        logger.info("类型：%s", doc.metadata.get('file_type'))
        logger.info("分类：%s", doc.metadata.get('category'))
        logger.info("内容预览：%s...", doc.page_content[:100])