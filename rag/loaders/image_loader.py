# ============================================================================
# 模块：rag/loaders/image_loader.py
# 图片文字识别加载器（基于 PaddleOCR）
# ============================================================================
# 【包的作用】
# PaddleOCR 是百度开源的 OCR（光学字符识别）工具，支持中英文识别。
# 本项目用它从图片中提取文字，使图片也能被 RAG 系统检索。
#
# 【PaddleOCR 核心 API】
#   PaddleOCR(lang='ch', show_log=False) → 创建 OCR 实例
#     lang: str — 语言，'ch'=中文，'en'=英文
#     show_log: bool — 是否显示 PaddleOCR 的内部日志
#     use_angle_cls: bool — 是否启用文字方向分类（默认 True）
#     det_db_thresh: float — 文字检测阈值（0.3），越低越敏感
#   ocr.ocr(file_path, cls=True) → 返回识别结果
#     cls: bool — 是否进行文字方向分类
#     返回值格式：[[[bbox, (text, confidence)], ...], ...]
#       bbox: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] — 文字框四个顶点坐标
#       text: str — 识别出的文字
#       confidence: float — 置信度（0~1）
#
# 【PaddleOCR 的安装与使用注意】
#   - 安装：pip install paddlepaddle paddleocr
#   - 首次运行会自动下载模型文件（约 100MB），需要联网
#   - GPU 版本需要 CUDA 环境，CPU 版本也能用但较慢
#   - 本项目延迟导入（在 load() 方法内 import），避免不使用图片时也加载模型
#
# 【适用场景】
#   - 扫描版 PDF 转图片后的文字提取
#   - 金融报表截图、合同照片的文字识别
#   - 带文字的图表、流程图
#
# 【使用示例】
#   loader = ImageLoader("data/财务报表截图.png")
#   docs = loader.load()  # → 一个 Document，page_content 为识别出的文字
# ============================================================================

from typing import List

from langchain_core.documents import Document

from rag.loaders.base import BaseDocumentLoader


class ImageLoader(BaseDocumentLoader):
    """
    图片 OCR 加载器：使用 PaddleOCR 识别图片中的文字

    【SUPPORTED_EXTENSIONS】
    支持常见图片格式：.jpg, .jpeg, .png, .bmp, .gif, .webp

    【为什么延迟导入 PaddleOCR？】
    PaddleOCR 是一个重型依赖（约 100MB 模型文件），不是所有场景都需要。
    在 load() 方法内部 import，只有真正加载图片时才触发导入，
    避免 import 本模块时就加载模型，节省内存和启动时间。
    """

    SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}

    def load(self) -> List[Document]:
        """
        使用 PaddleOCR 识别图片中的文字

        【返回值】
            List[Document] — 单元素列表，Document.page_content 为识别出的文字

        【PaddleOCR 返回结果解析】
        ocr.ocr() 返回格式：
        [
            [  # 第一页（图片只有一页）
                [[[x1,y1],[x2,y2],[x3,y3],[x4,y4]], ('文字内容', 0.98)],
                [[...], ('更多文字', 0.95)],
                ...
            ]
        ]
        我们只取每行的文字内容（line[1][0]），用换行符拼接。
        """
        from paddleocr import PaddleOCR

        ocr = PaddleOCR(lang='ch', show_log=False)
        result = ocr.ocr(self.file_path, cls=True)

        text = ""
        if result and result[0]:
            lines = [line[1][0] for line in result[0]]
            text = '\n'.join(lines)

        doc = Document(page_content=text, metadata={})
        return self._add_metadata([doc], 'image')