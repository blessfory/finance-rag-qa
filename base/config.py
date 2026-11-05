# ============================================================================
# 模块：base/config.py
# ============================================================================
# 【包的作用】
# - configparser：Python 标准库，用于解析 .ini 格式的配置文件
# - os：操作系统路径操作，用于拼接文件路径等
#
# 【configparser 核心 API 速查】
# ┌─────────────────────────────────────────────────────────────────┐
# │ API                             │ 说明                          │
# ├─────────────────────────────────────────────────────────────────┤
# │ ConfigParser()                  │ 创建配置解析器实例               │
# │ read(file)                      │ 读取 .ini 文件，返回读取到的文件列表│
# │ get(section, key)               │ 获取字符串值，key 不存在抛异常   │
# │ get(section, key, fallback=x)   │ 获取值，不存在时返回 x          │
# │ getint(section, key)            │ 获取整数值                      │
# │ getfloat(section, key)          │ 获取浮点值                      │
# │ getboolean(section, key)        │ 获取布尔值（'yes'/'no' 等）      │
# │ sections()                      │ 返回所有 section 名称列表        │
# │ options(section)                │ 返回指定 section 的所有 key 列表 │
# │ items(section)                  │ 返回指定 section 的全部 (key,value)│
# │ has_section(section)            │ 检查 section 是否存在            │
# │ has_option(section, key)        │ 检查 key 是否存在               │
# └─────────────────────────────────────────────────────────────────┘
#
# 【设计模式：单例模式】
# 配置类通常在整个项目中只需实例化一次，避免重复读取文件。
# 推荐用法：在项目入口创建全局实例，其他模块通过 import 引用。
# 例如：from base.config import config → cfg = config() → cfg.milvus_host
# ============================================================================

import os
import configparser


class Config:
    """
    全局配置管理类

    加载 config.ini 中的所有配置项，将每个配置节（section）中的键值对
    读取为实例属性，方便项目内各模块统一使用。

    【使用方式】
        cfg = Config('config.ini')   # 指定配置文件路径
        cfg = Config()               # 默认读取项目根目录下的 config.ini
        print(cfg.mysql_host)        # 访问配置项
        print(cfg.retrieval_k)       # 访问检索参数

    【参数说明】
        config_file: str — 配置文件路径，不传则自动定位到项目根目录的 config.ini
                       支持相对路径和绝对路径

    【configparser.get() 参数详解】
        section: str  — 配置节名称，如 'mysql'、'llm'
        option:  str  — 配置项名称，如 'host'、'port'
        fallback: Any — 默认值，当 section 或 option 不存在时返回此值
                        如果不用 fallback 且 key 不存在，会抛出 NoOptionError
    """

    def __init__(self, config_file=None):
        if config_file is None:
            config_file = os.path.join(os.path.dirname(__file__), '..', 'config.ini')
        # ConfigParser 实例化，用于解析 .ini 文件
        self.config = configparser.ConfigParser()
        # read() 读取配置文件，返回成功读取的文件路径列表
        # 如果文件不存在，不会报错，只是返回空列表
        # encoding='utf-8'：Windows 下默认用 GBK 编码，显式指定 UTF-8 避免中文乱码
        self.config.read(config_file, encoding='utf-8')
        logger_name = 'RAG.config'
        try:
            # 延迟导入，避免循环依赖
            from base.logger import setup_logger
            _cfg_logger = setup_logger(logger_name, log_file=self.config.get('logger', 'log_file', fallback='rag.logs'))
            _cfg_logger.info("配置文件已加载：%s", config_file)
        except Exception:
            pass

        # =====================================================================
        # MySQL 配置 — 关系型数据库
        # 用途：存储结构化业务数据，如问答记录、知识库元数据、用户信息
        # 面试点：为什么 RAG 系统还需要 MySQL？
        #   答：向量库负责语义检索，MySQL 负责精确查询和事务管理。
        #       例如"用户 A 最近 3 天的提问历史"这类精确查询，MySQL 更合适。
        # =====================================================================
        self.mysql_host = self.config.get('mysql', 'host', fallback='localhost')
        self.mysql_user = self.config.get('mysql', 'user', fallback='root')
        self.mysql_password = self.config.get('mysql', 'password', fallback='root')
        self.mysql_db = self.config.get('mysql', 'database', fallback='bank_rag')
        # int() 转换：configparser 读取的值都是字符串，端口需要转为整数
        self.mysql_port = int(self.config.get('mysql', 'port', fallback='3306'))

        # =====================================================================
        # Redis 配置 — 内存缓存数据库
        # 用途：缓存 LLM 问答结果、会话管理、限流计数
        # 面试点：Redis 在 RAG 中的作用？
        #   答：1) 缓存相同问题的回答，避免重复调用 LLM（省钱+提速）
        #       2) 存储多轮对话上下文，实现会话记忆
        #       3) 实现 API 限流（Rate Limiting），防止滥用
        # =====================================================================
        self.redis_host = self.config.get('redis', 'host', fallback='localhost')
        self.redis_port = int(self.config.get('redis', 'port', fallback='16379'))
        self.redis_password = self.config.get('redis', 'password', fallback='1234')
        self.redis_db = int(self.config.get('redis', 'db', fallback='0'))

        # =====================================================================
        # Milvus 配置 — 向量数据库（RAG 核心组件）
        # 用途：存储文档的 Embedding 向量，支撑语义相似度检索
        # 面试点：Milvus 和传统数据库的区别？
        #   答：传统库是"精确匹配"（SELECT * WHERE id=5），
        #       Milvus 是"近似匹配"（找与查询向量最相似的 Top-K 个向量）。
        #       它基于 ANN（近似最近邻）算法，如 IVF、HNSW，牺牲微小精度换巨大速度。
        # =====================================================================
        self.milvus_host = self.config.get('milvus', 'host', fallback='localhost')
        self.milvus_port = int(self.config.get('milvus', 'port', fallback='19530'))
        self.milvus_database_name = self.config.get('milvus', 'database_name', fallback='')
        self.milvus_collection_name = self.config.get('milvus', 'collection_name', fallback='')

        # =====================================================================
        # LLM 配置 — 大语言模型
        # 用途：1) 调用 Embedding 模型将文本转为向量
        #       2) 调用 Chat 模型根据检索结果生成最终回答
        # 面试点：Embedding 模型和 Chat 模型的区别？
        #   答：Embedding 模型输出固定维度的向量（如 1024 维），用于文本相似度计算；
        #       Chat 模型输出自然语言文本，用于对话生成。
        #       在 RAG 中，同一个服务商通常同时提供两种模型。
        # =====================================================================
        self.llm_model = self.config.get('llm', 'model', fallback='qwen3.7-max-2026-05-17')
        self.dashscope_api_key = self.config.get('llm', 'dashscope_api_key', fallback='') or os.getenv('DASHSCOPE_API_KEY') or ''
        self.dashscope_base_url = self.config.get(
            'llm', 'dashscope_base_url',
            fallback='https://dashscope.aliyuncs.com/compatible-mode/v1'
        )

        # =====================================================================
        # 检索参数 — RAG 系统核心调参区
        # 面试点：父子切片（Parent-Child Chunk）策略是什么？
        #   答：是一种"小块检索 + 大块喂给 LLM"的策略。
        #       - 子块（child_chunk_size=400）：小粒度，用于向量检索，匹配精度高
        #       - 父块（parent_chunk_size=1500）：大粒度，是子块所在的完整上下文段落
        #       检索时：用子块做向量匹配 → 找到对应的父块 → 把父块喂给 LLM
        #       好处：检索精度高 + 上下文完整，兼顾准确率和召回率
        # =====================================================================
        self.parent_chunk_size = int(self.config.get('retrieval', 'parent_chunk_size', fallback='1500'))
        self.child_chunk_size = int(self.config.get('retrieval', 'child_chunk_size', fallback='400'))
        self.chunk_overlap = int(self.config.get('retrieval', 'chunk_overlap', fallback='80'))
        self.retrieval_k = int(self.config.get('retrieval', 'retrieval_k', fallback='3'))
        self.candidate_m = int(self.config.get('retrieval', 'candidate_m', fallback='2'))

        # =====================================================================
        # 日志配置
        # =====================================================================
        self.log_file = self.config.get('logger', 'log_file', fallback='rag.logs')

        # =====================================================================
        # 应用配置 — 业务层面配置
        # =====================================================================
        valid_sources_str = self.config.get('app', 'valid_sources', fallback='[]')
        self.valid_sources = self._parse_list(valid_sources_str)
        self.customer_service_phone = self.config.get('app', 'customer_service_phone', fallback='')

    @staticmethod
    def _parse_list(raw_str):
        """
        解析 config.ini 中的列表格式字符串

        【为什么需要这个函数？】
        configparser 读取的值都是字符串，例如 '["ai", "java"]' 是一个字符串，
        需要手动解析为 Python 列表 ['ai', 'java']。

        【参数】
            raw_str: str — 形如 '["ai", "java", "test"]' 的字符串

        【返回值】
            list — 解析后的 Python 列表，如 ['ai', 'java', 'test']

        【解析步骤】
            1. strip() 去除首尾空格
            2. strip('[]') 去除首尾的方括号
            3. 按逗号分割
            4. 每个元素去除引号和空格
        """
        raw_str = raw_str.strip().strip('[]')
        if not raw_str:
            return []
        return [item.strip().strip('"').strip("'") for item in raw_str.split(',')]