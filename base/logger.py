# ============================================================================
# 模块：base/logger.py
# ============================================================================
# 【包的作用】
# - logging：Python 标准库的日志模块，提供灵活的多级别、多输出目标的日志记录
# - os：用于创建日志文件目录
# - datetime：时间处理（备用）
#
# 【logging 核心概念】
# 日志级别（从低到高）：
#   DEBUG     → 10   调试信息，开发时使用
#   INFO      → 20   一般信息，记录系统正常运行状态
#   WARNING   → 30   警告信息，可能有问题但不影响运行
#   ERROR     → 40   错误信息，某些功能无法正常运行
#   CRITICAL  → 50   严重错误，系统可能崩溃
#
# 日志组件：
#   Logger   → 日志记录器，提供应用程序调用的接口
#   Handler  → 处理器，决定日志输出到哪（控制台/文件/网络等）
#   Formatter → 格式化器，决定日志的输出格式
#   Filter   → 过滤器，决定哪些日志可以被输出
#
# 【logging 常用 API 速查】
# ┌──────────────────────────────────────────────────────────────────┐
# │ API                                   │ 说明                      │
# ├──────────────────────────────────────────────────────────────────┤
# │ logging.getLogger(name)               │ 获取/创建指定名称的 Logger  │
# │ logger.setLevel(logging.INFO)         │ 设置日志级别               │
# │ logger.addHandler(handler)            │ 添加处理器                 │
# │ logger.hasHandlers()                  │ 判断是否已有处理器          │
# │ logger.handlers.clear()               │ 清空所有处理器（防重复）     │
# │ logging.StreamHandler()               │ 输出到控制台               │
# │ logging.FileHandler(path, encoding)   │ 输出到文件                 │
# │ logging.Formatter(fmt)                │ 设置日志格式               │
# └──────────────────────────────────────────────────────────────────┘
#
# 【日志格式占位符】
# %(asctime)s   → 时间戳，如 2024-01-15 14:30:00,123
# %(name)s      → Logger 名称，如 'RAG'
# %(levelname)s → 日志级别，如 'INFO'、'ERROR'
# %(filename)s  → 输出日志的源文件名，如 'main.py'
# %(lineno)d    → 输出日志的代码行号
# %(message)s   → 日志消息内容
# ============================================================================

import logging
import os
from datetime import datetime


def setup_logger(name='RAG', log_file=None, level=logging.INFO):
    """
    初始化并配置日志记录器

    【为什么需要这个函数？】
    直接使用 logging.basicConfig() 只能配置一次，且全局生效。
    在模块化的项目中，每个模块可能需要独立的 logger，通过名称区分。
    本函数封装了完整的 logger 配置流程，并防止重复添加 handler。

    【参数】
        name: str — Logger 名称，默认 'RAG'
                    通过名称可以分层管理：'RAG.vectordb' 是 'RAG' 的子 logger
                    getLogger('RAG.vectordb') 会自动继承 'RAG' 的配置
        log_file: str | None — 日志文件路径，默认 None
                    - None 或不传：只输出到控制台
                    - 传入文件路径：同时输出到控制台和文件
                    - 如果目录不存在，会自动创建
        level: int — 日志级别，默认 logging.INFO (20)
                    常用值：logging.DEBUG(10), INFO(20), WARNING(30), ERROR(40)

    【返回值】
        logging.Logger — 配置好的日志记录器实例

    【使用示例】
        # 基础用法：只输出到控制台
        logger = setup_logger('MyApp')

        # 同时输出到文件
        logger = setup_logger('MyApp', log_file='logs/app.logs')

        # 调试模式
        logger = setup_logger('MyApp', level=logging.DEBUG)

        logger.info('系统启动')          # → 2024-01-15 14:30:00 - MyApp - INFO - main.py:10 - 系统启动
        logger.error('数据库连接失败')    # → 2024-01-15 14:30:01 - MyApp - ERROR - main.py:11 - 数据库连接失败
    """
    # getLogger(name)：如果 name 对应的 Logger 已存在则返回它，否则创建新的
    # 这意味着同一个 name 多次调用 getLogger 返回的是同一个对象（单例）
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 防止重复添加 Handler
    # 如果 logger 已经被配置过（比如在模块重载时），先清空旧的 handler
    # 否则每次调用都会添加新的 handler，导致日志重复输出
    if logger.hasHandlers():
        logger.handlers.clear()

    # Formatter：定义日志的输出格式
    # 格式说明：
    #   2024-01-15 14:30:00,123 - RAG - INFO - main.py:42 - 这是一条日志
    #   |____________________|   |__|  |__|   |_______|  |__________|
    #       时间戳+毫秒          名称   级别   文件名:行号    消息内容
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )

    # 控制台 Handler：将日志输出到终端（标准输出流）
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件 Handler：将日志同时写入文件
    # 为什么需要文件日志：控制台日志会丢失，文件日志可以持久保存，方便事后排查
    if log_file:
        # 自动创建日志文件所在目录（如果不存在）
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        # encoding='utf-8' 确保中文日志不乱码
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name='RAG'):
    """
    获取已有的 Logger 实例（不做配置）

    【与 setup_logger 的区别】
    - setup_logger()：创建/配置 Logger（适合在程序入口调用一次）
    - get_logger()：  只获取 Logger（适合在各模块中调用，继承已有配置）

    【参数】
        name: str — Logger 名称，默认 'RAG'

    【返回值】
        logging.Logger — 已存在的 Logger 实例，如果不存在则创建一个未配置的

    【使用示例】
        # 在子模块中获取 logger（继承入口处 setup_logger 的配置）
        from base.logger import get_logger
        logger = get_logger('RAG.vectordb')  # 自动继承 'RAG' 的 handler 和 level
        logger.info('Milvus 连接成功')
    """
    return logging.getLogger(name)