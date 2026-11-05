# ============================================================================
# 模块：rag/core/embedding.py
# 向量化服务：加载模型 + 文本转向量，与具体数据库解耦
# ============================================================================
# 【什么是 Embedding（向量化）？】
# 将文本（如"人工智能是什么"）转换成一串固定长度的浮点数（向量），
# 语义相近的文本，向量在空间中距离也近。
#
# 例如：
#   "人工智能"  →  [0.12, -0.34, 0.56, ..., 0.78]  （1024个数字）
#   "机器学习"  →  [0.11, -0.33, 0.55, ..., 0.77]  （与上面的向量很接近）
#   "今天天气"  →  [-0.45, 0.67, -0.23, ..., -0.11] （与上面的向量距离很远）
#
# 【为什么需要向量化？】
# 用户输入问题后，RAG 系统需要从海量文档中找到最相关的内容。
# 但计算机无法直接理解"语义相似"，只能做数学运算。
# 向量化就是把"语义相似"变成"向量距离近"这个数学问题。
#
# 【BGE-M3 模型】
# 本项目使用 BAAI 开源的 bge-m3 模型：
# - 支持中英文等多语言
# - 输出 1024 维向量
# - 本地运行，不需要联网调用 API
# - 模型文件存放在 项目根目录/models/bge-m3/
#
# 【与 chunker.py 的关系】
# chunker.py 负责：文档 → 文本块（父子分块）
# 本模块负责：  文本块 → 向量（存入 Milvus 供检索）
#
# 【与 Milvus 的关系】
# 本模块只负责"文本→向量"，不关心向量存到哪里。
# 存入 Milvus 的逻辑在 rag/repo/milvus_repo.py 中，实现了关注点分离。
# ============================================================================

import os  # 操作系统接口，用于路径拼接
from typing import List  # 类型注解，声明 List[str] 等类型

import numpy as np  # 数值计算库，用于处理向量数组

# ============================================================================
# 【核心依赖：sentence_transformers 库 + BGE-M3 模型】
# ============================================================================
# 本文件的核心工作"文本→向量"，靠的是下面这两个东西配合：
#
# 1. sentence_transformers 库（Python 包）
#    - HuggingFace 公司开发的 Python 库，相当于一个"模型运行器"
#    - 它本身不包含任何 AI 模型，只是一个加载和调用模型的工具
#    - 类比：它就像 VLC 播放器，用来播放视频，但视频文件（.mp4）是另外的
#
# 2. BGE-M3 模型（模型文件，存放在 ../models/bge-m3/ 目录下）
#    - BAAI（北京智源研究院）开源的多语言 Embedding 模型
#    - 这才是真正做"文本→向量"转换的 AI 大脑
#    - 类比：它就是 .mp4 视频文件，包含实际内容
#
# 【两者的关系】
#   SentenceTransformer  ←加载→  BGE-M3 模型文件
#       （工具）                    （大脑）
#
#   代码示例：
#     model = SentenceTransformer("models/bge-m3")   # 用工具加载模型
#     vec = model.encode(["你好"])                    # 调用模型把文本变成向量
#
# 【为什么用 BGE-M3？】
#   - 中英文效果都好（很多模型只擅长英文）
#   - 完全本地运行，不需要联网，不花 API 费用
#   - 输出 1024 维向量，精度和速度的平衡点
#   - 开源免费，可以商用
# ============================================================================
from sentence_transformers import SentenceTransformer

from base.config import Config  # 项目配置类，读取 config.ini
from base.logger import setup_logger  # 日志工具，统一日志输出

# 初始化配置和日志
cfg = Config()  # 读取配置文件
logger = setup_logger('RAG.embedding', log_file=cfg.log_file)  # 创建本模块专用日志器


class EmbeddingService:
    """
    向量化服务：封装 SentenceTransformer 模型的加载和 encode 调用

    【核心职责】
        1. 加载本地 BGE-M3 模型
        2. 提供文本→向量的转换接口
        3. 不关心向量存到哪个数据库（解耦设计）

    【使用方式】
        service = EmbeddingService()                          # 加载模型（默认路径）
        vectors = service.encode(["文本1", "文本2"])           # 批量编码 → numpy 数组
        query_vec = service.embed_query("用户问题")            # 单条查询 → Python 列表

    【设计要点】
        - 模型路径默认从项目根目录/models/bge-m3/ 加载，不依赖工作目录
        - 默认开启 L2 归一化，向量内积 = 余弦相似度，与 Milvus 的 IP 度量匹配
        - device 参数支持 CPU/GPU 灵活切换
    """

    def __init__(self, model_path: str = None, device: str = None):
        """
        初始化：加载 BGE-M3 模型

        【参数】
            model_path: str | None
                - 不传 → 自动从 项目根目录/models/bge-m3/ 加载
                - 传入 → 使用指定路径（如 HuggingFace 模型名 "BAAI/bge-m3"）
            device: str | None
                - 不传 → SentenceTransformer 自动选择（优先 GPU，没有则用 CPU）
                - "cpu" → 强制使用 CPU
                - "cuda:0" → 使用第 0 号 GPU

        【内部流程】
            1. 确定模型路径（默认路径用 os.path.dirname(__file__) 定位）
            2. 加载 SentenceTransformer 模型
            3. 获取向量维度（1024）并记录日志
        """
        # 如果没传模型路径，使用默认路径
        # os.path.dirname(__file__)：当前文件所在目录 → rag/core/
        # '..'：上一级 → rag/
        # '..'：再上一级 → 项目根目录/
        # 最终拼接：项目根目录/models/bge-m3/
        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(__file__), '..', '..', 'models', 'bge-m3'
            )
        # 加载模型，device 决定用 CPU 还是 GPU 推理
        self.model = SentenceTransformer(model_path, device=device)
        # 获取向量维度，bge-m3 输出 1024 维
        self.dim = self.model.get_embedding_dimension()
        logger.info("Embedding 模型已加载：%s（维度=%d）", model_path, self.dim)

    def encode(
        self,
        texts: List[str],  # 要编码的文本列表，如 ["文本1", "文本2", ...]
        normalize: bool = True,  # 是否 L2 归一化，默认 True
        show_progress_bar: bool = False,  # 是否显示进度条，默认 False
        batch_size: int = 32,  # 每批处理多少条文本，默认 32
    ) -> np.ndarray:
        """
        批量编码：将多条文本一次性转为向量数组

        【参数】
            texts: List[str] — 要编码的文本列表
            normalize: bool — 是否归一化：
                - True（默认）：向量模长变为 1，内积 = 余弦相似度
                - False：保留原始向量，用于需要原始值计算的场景
            show_progress_bar: bool — 是否显示 tqdm 进度条，大量文本时建议开启
            batch_size: int — 每次喂给模型的文本数量，影响显存占用和速度

        【返回值】
            np.ndarray — shape 为 (len(texts), dim) 的 numpy 数组
            例如：传入 3 条文本 → 返回 (3, 1024) 的数组

        【为什么 normalize=True 是默认值？】
            归一化后，向量 A 和 B 的内积（点积）就等于它们的余弦相似度。
            Milvus 用 IP（Inner Product，内积）度量时，直接对应语义相似度，
            不需要额外计算余弦距离，检索效率更高。

        【batch_size 说明】
            - 太小 → 推理次数多，GPU 利用率低，慢
            - 太大 → 显存溢出（OOM）
            - 32 是经验值，CPU 推理时可以设更大
        """
        # 调用 SentenceTransformer 的 encode 方法
        return self.model.encode(
            texts,  # 输入文本列表
            normalize_embeddings=normalize,  # 归一化开关
            show_progress_bar=show_progress_bar,  # 进度条开关
            batch_size=batch_size,  # 批次大小
            convert_to_numpy=True,  # 返回 numpy 数组（而非 torch tensor）
        )

    def embed_query(self, query: str, normalize: bool = True) -> List[float]:
        """
        单条查询编码：encode 的便捷包装

        【参数】
            query: str — 用户输入的问题，如 "兴业银行2025年净利润是多少？"
            normalize: bool — 是否归一化，默认 True

        【返回值】
            List[float] — 长度为 dim 的 Python 列表，可直接传给 Milvus 检索
            例如：[0.12, -0.34, 0.56, ..., 0.78]  （共 1024 个浮点数）

        【为什么单独封装这个方法？】
            - encode 返回的是 numpy 数组，取 [0] 再 tolist() 比较繁琐
            - 这个方法直接返回 Python 列表，调用方更简洁
            - 语义上明确表示"这是用户查询"，与批量文档编码区分开
        """
        # 调用 encode（内部会自动包装成列表），取第 0 条结果，转为 Python 列表
        return self.encode([query], normalize=normalize)[0].tolist()