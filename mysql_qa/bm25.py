# ============================================================================
# 模块：mysql_qa/bm25.py
# ============================================================================
# 【包的作用】
# - rank_bm25：BM25 算法的 Python 实现，用于关键词级别的文本检索
#   - BM25Okapi：标准 BM25 实现，参数 k1=1.5, b=0.75（Okapi 默认）
#   - BM25L：BM25 的变体，对长文档的惩罚更温和
#   - BM25Plus：BM25 的变体，加入了一个小的 δ 项避免下界问题
#   - 安装：pip install rank-bm25
#   - 官网：https://github.com/dorianbrown/rank_bm25
#
# - numpy：Python 数值计算核心库
#   - 这里用于 softmax 归一化计算（exp、max、sum 等向量化操作）
#   - 安装：pip install numpy
#
# - preprocess_text：本项目自定义的中文分词函数，基于 jieba
#   - 输入：原始文本字符串
#   - 输出：分词后的 token 列表，如 ['python', '是', '什么']
#
# 【BM25 算法简介】
# BM25（Best Matching 25）是搜索引擎中经典的排序算法，用于计算查询和文档的相关性分数。
# 核心思想：考虑词频（TF）和逆文档频率（IDF），同时对长文档做了长度归一化。
#
# 公式（简化理解）：
#   score(D, Q) = Σ IDF(qi) × (TF(qi, D) × (k1 + 1)) / (TF(qi, D) + k1 × (1 - b + b × |D|/avgdl))
#   其中：k1 控制词频饱和度（默认1.5），b 控制文档长度归一化程度（默认0.75）
#
# 【BM25 vs 向量检索（Embedding）】
# ┌──────────────────┬──────────────────────────────┬──────────────────────────────┐
# │ 维度              │ BM25（关键词匹配）             │ 向量检索（语义匹配）             │
# ├──────────────────┼──────────────────────────────┼──────────────────────────────┤
# │ 原理              │ 基于词频和逆文档频率统计         │ 基于深度学习 Embedding 向量     │
# │ 优势              │ 精确匹配强、可解释、无需GPU      │ 语义理解、同义词、跨语言          │
# │ 劣势              │ 无法理解同义词、语义            │ 需要模型、计算成本高             │
# │ 适用场景           │ 关键词明确的问题               │ 语义模糊、需要推理的问题          │
# │ 本项目定位         │ 第一轮粗筛（快速过滤）           │ 第二轮精排（Rerank）            │
# └──────────────────┴──────────────────────────────┴──────────────────────────────┘
#
# 【rank_bm25.BM25Okapi 核心 API】
# ┌────────────────────────────────────────────┬──────────────────────────────────┐
# │ API                                        │ 说明                              │
# ├────────────────────────────────────────────┼──────────────────────────────────┤
# │ BM25Okapi(corpus)                          │ 初始化，传入分词后的语料库          │
# │   corpus: list[list[str]]                  │ 每个文档是一个 token 列表           │
# │   k1: float = 1.5                          │ 词频饱和度参数，越大词频影响越大    │
# │   b: float = 0.75                          │ 长度归一化参数，0=不归一化，1=完全归一化│
# │   epsilon: float = 0.25                    │ 防止 IDF 为负的小常数              │
# │ get_scores(query_tokens)                   │ 计算查询与所有文档的 BM25 分数     │
# │   返回: numpy.ndarray                       │ 每个文档一个分数，越高越相关        │
# │ get_top_n(query_tokens, corpus, n=5)       │ 返回分数最高的 n 个文档             │
# │ get_batch_scores(query_tokens_list)        │ 批量计算多个查询的分数              │
# └────────────────────────────────────────────┴──────────────────────────────────┘
#
# 【numpy 核心 API 速查】
# ┌────────────────────────────────────────────┬──────────────────────────────────┐
# │ API                                        │ 说明                              │
# ├────────────────────────────────────────────┼──────────────────────────────────┤
# │ np.array(list)                             │ 创建数组                          │
# │ np.exp(x)                                  │ e^x，逐元素计算                    │
# │ np.max(x) / np.min(x)                      │ 最大值 / 最小值                    │
# │ np.sum(x) / np.mean(x)                     │ 求和 / 平均值                      │
# │ np.argmax(x) / np.argmin(x)                │ 最大值 / 最小值 的索引               │
# │ np.argsort(x)                              │ 返回排序后的索引（升序）            │
# │ np.top_k(x, k)                             │ （需自行实现）返回最大的 k 个值      │
# └────────────────────────────────────────────┴──────────────────────────────────┘
# ============================================================================

from rank_bm25 import BM25Okapi
import numpy as np
from .preprocess import preprocess_text
from base import setup_logger
from mysql_qa.db.mysql_client import MySQLClient
from .redis_client import RedisClient

# 模块级单例：Logger 在整个项目中只实例化一次
logger = setup_logger()


class BM25Search:
    """
    基于 BM25 算法的关键词检索器

    【工作流程】
    1. 初始化时从 Redis/MySQL 加载所有问答对的问题
    2. 对每个问题做中文分词，构建 BM25 索引
    3. 用户查询时：分词 → BM25 打分 → Softmax 归一化 → 阈值过滤 → 返回答案

    【为什么用 BM25 而不是直接用向量检索？】
    BM25 作为第一轮粗筛，速度快（毫秒级），不需要 GPU，适合精确关键词匹配。
    本项目后续还会用 Embedding 向量做 Rerank 精排，两者互补。

    【使用方式】
        bm25 = BM25Search()
        answer, need_rerank = bm25.search("Python是什么？")
        if need_rerank:
            # BM25 没找到可靠答案，交给向量检索处理
            answer = vector_search("Python是什么？")
    """

    # Redis 缓存 key 常量（类级别定义，统一管理）
    _REDIS_KEY_ORIGINAL = "qa_original_questions"   # 原始问题列表
    _REDIS_KEY_TOKENIZED = "qa_tokenized_questions"  # 分词后的问题列表

    def __init__(self):
        """
        初始化 BM25 检索器

        【初始化步骤】
        1. 连接 Redis 和 MySQL
        2. 调用 _load_data() 加载问题数据并构建 BM25 索引
        """
        self.logger = logger
        self.redis_client = RedisClient()
        self.mysql_client = MySQLClient()
        self.bm25 = None               # BM25Okapi 模型实例
        self.questions = None          # 分词后的问题列表，用于 BM25 打分
        self.original_questions = None # 原始问题文本列表，用于返回结果
        self._load_data()

    # ========================================================================
    # 数据加载
    # ========================================================================

    def _load_data(self):
        """
        加载问答数据并构建 BM25 索引

        【加载策略：Redis 缓存 → MySQL 兜底】
        1. 优先从 Redis 读取（速度快，减少 MySQL 压力）
        2. Redis 未命中 → 从 MySQL 加载 → 分词 → 回写 Redis 缓存
        3. 最后统一构建 BM25 模型

        【数据格式统一】
        MySQL 的 fetch_questions() 返回 [(q1,), (q2,), ...] 格式的元组列表，
        需要统一转为 ['q1', 'q2', ...] 的字符串列表，与 Redis 格式保持一致。
        """
        original_key = self._REDIS_KEY_ORIGINAL
        tokenized_key = self._REDIS_KEY_TOKENIZED

        # ======================================================================
        # 第一步：尝试从 Redis 缓存加载
        # get_data() 返回 Python 对象（list/dict/str）或 None
        # ======================================================================
        self.original_questions = self.redis_client.get_data(original_key)
        tokenized_questions = self.redis_client.get_data(tokenized_key)

        # ======================================================================
        # 第二步：Redis 未命中，从 MySQL 加载并预处理
        # ======================================================================
        if not self.original_questions or not tokenized_questions:
            # fetch_questions() 返回 [(q1,), (q2,), ...] 格式
            raw_questions = self.mysql_client.fetch_questions()
            if not raw_questions:
                self.logger.warning("未加载到问题，BM25 模型未构建")
                return

            # 统一格式：将 MySQL 的元组列表转为字符串列表
            # 这样无论数据来源，self.original_questions 都是 ['q1', 'q2', ...]
            self.original_questions = [q[0] for q in raw_questions]

            # 分词：对每个问题调用 jieba 分词
            # preprocess_text("Python是什么？") → ['python', '是', '什么', '？']
            tokenized_questions = [
                preprocess_text(q) for q in self.original_questions
            ]

            # 回写 Redis 缓存，下次启动直接命中
            self.redis_client.set_data(original_key, self.original_questions)
            self.redis_client.set_data(tokenized_key, tokenized_questions)
            self.logger.info(
                f"从 MySQL 加载 {len(self.original_questions)} 条问题，已缓存到 Redis"
            )

        # ======================================================================
        # 第三步：构建 BM25 模型
        # BM25Okapi 的 corpus 参数要求：list[list[str]]，每个文档是一个 token 列表
        # ======================================================================
        self.questions = tokenized_questions
        self.bm25 = BM25Okapi(self.questions)
        self.logger.info(f"BM25 模型构建完成，共 {len(self.questions)} 条问题")

    # ========================================================================
    # 资源释放
    # ========================================================================

    def close(self):
        """
        释放 BM25 检索器持有的数据库和缓存连接

        【关闭顺序】
        先关闭 MySQL 连接，再关闭 Redis 连接（与初始化顺序相反）。
        """
        try:
            self.mysql_client.close()
        except Exception as e:
            self.logger.error(f"BM25 关闭 MySQL 连接时出错: {e}")

        try:
            self.redis_client.close()
        except Exception as e:
            self.logger.error(f"BM25 关闭 Redis 连接时出错: {e}")

    # ========================================================================
    # 分数归一化
    # ========================================================================

    def _softmax(self, scores):
        """
        将 BM25 原始分数归一化为 0~1 之间的概率分布

        【参数】
        scores: numpy.ndarray — BM25 原始分数数组，shape=(n_docs,)

        【返回值】
        numpy.ndarray — 归一化后的概率分布，所有元素之和为 1

        【Softmax 公式】
        softmax(xi) = exp(xi) / Σ exp(xj)

        【为什么要减去 max(scores)？】
        这是数值稳定性技巧（numerical stability）：
        - exp(1000) 会导致浮点数溢出（inf）
        - exp(1000 - 1000) = exp(0) = 1，安全可控
        - 减去最大值不改变 softmax 的数学结果，因为：
          softmax(xi - max) = exp(xi - max) / Σ exp(xj - max)
                           = exp(xi)/exp(max) / Σ(exp(xj)/exp(max))
                           = exp(xi) / Σ exp(xj) = softmax(xi)

        【为什么用 Softmax 而不是直接比较 BM25 分数？】
        BM25 原始分数没有上界，不同查询的分数范围差异很大：
        - 查询"Python"可能最高分 50
        - 查询"机器学习"可能最高分 8
        用 Softmax 归一化后，分数都在 0~1 之间，可以用统一的阈值（如 0.85）判断。
        """
        # 减去最大值防止 exp 溢出
        exp_scores = np.exp(scores - np.max(scores))
        # 归一化：各项除以总和
        return exp_scores / exp_scores.sum()

    # ========================================================================
    # 检索主方法
    # ========================================================================

    def search(self, query, threshold=0.85):
        """
        执行 BM25 关键词检索，返回最佳匹配的答案

        【参数】
        query: str — 用户输入的查询问题
        threshold: float — 置信度阈值（0~1），默认 0.85
                  只有 Softmax 归一化后最高分 ≥ threshold 的才返回答案

        【返回值】
        tuple[answer, need_rerank]
        - answer: str | None — 匹配到的答案文本，未找到时为 None
        - need_rerank: bool — True 表示 BM25 没找到可靠答案，需要交给向量检索做语义匹配

        【返回值设计意图】
        返回 (answer, need_rerank) 让调用方可以：
        1. 如果 need_rerank=False → 直接使用 answer
        2. 如果 need_rerank=True → 将 query 交给 Embedding + Milvus 向量检索
        这是典型的"BM25 粗筛 + 向量精排"两级检索策略。

        【检索流程】
        1. 参数校验：检查 query 是否合法
        2. Redis 缓存检查：相同问题之前查过直接返回
        3. BM25 打分：query 分词 → get_scores → 得到每个文档的相关性分数
        4. Softmax 归一化：原始分数映射到 0~1
        5. 阈值过滤：最高分 ≥ threshold 才采纳
        6. 查 MySQL 获取答案并缓存

        【阈值选择经验】
        - 0.95+：只有非常确定的匹配才返回，精确率高但可能漏掉一些
        - 0.85：推荐值，平衡精确率和召回率
        - 0.70：放宽条件，召回率高但可能返回不太相关的答案
        """
        # ======================================================================
        # 第一步：参数校验
        # ======================================================================
        if not query or not isinstance(query, str):
            self.logger.error(f"无效查询: {query}")
            return None, False

        # ======================================================================
        # 第二步：检查 Redis 缓存（相同问题之前回答过）
        # 缓存 key 格式：answer:{query}
        # ======================================================================
        cached_answer = self.redis_client.get_data(f"answer:{query}")
        if cached_answer:
            self.logger.info(f"命中缓存: {query[:30]}...")
            return cached_answer, False

        try:
            # ==================================================================
            # 第三步：对查询分词
            # preprocess_text("Python是什么？") → ['python', '是', '什么', '？']
            # ==================================================================
            query_tokens = preprocess_text(query)

            # ==================================================================
            # 第四步：BM25 打分
            # get_scores() 返回 numpy.ndarray，shape=(n_docs,)
            # 每个元素是 query 与对应文档的 BM25 分数，分数越高越相关
            # ==================================================================
            scores = self.bm25.get_scores(query_tokens)

            # ==================================================================
            # 第五步：Softmax 归一化
            # 将原始分数映射到 0~1，便于用统一阈值判断
            # ==================================================================
            softmax_scores = self._softmax(scores)

            # ==================================================================
            # 第六步：取最高分
            # argmax() 返回分数最高的文档索引（不是分数值本身）
            # ==================================================================
            best_idx = softmax_scores.argmax()
            best_score = softmax_scores[best_idx]

            self.logger.info(
                f"查询: {query[:30]}... | 最佳匹配: {self.original_questions[best_idx][:30]}... "
                f"| Softmax 分数: {best_score:.3f}"
            )

            # ==================================================================
            # 第七步：阈值判断
            # ==================================================================
            if best_score >= threshold:
                # 置信度足够高，从 MySQL 获取匹配问题的答案
                original_question = self.original_questions[best_idx]
                answer = self.mysql_client.fetch_answer(original_question)

                if answer:
                    # 将答案缓存到 Redis，下次同样问题直接返回
                    self.redis_client.set_data(f"answer:{query}", answer, ex=3600)
                    self.logger.info(
                        f"BM25 检索成功 | Softmax: {best_score:.3f}"
                    )
                    return answer, False

            # 置信度不够，标记需要 Rerank（向量检索）
            self.logger.info(
                f"BM25 置信度不足 | 最高 Softmax: {best_score:.3f} | 阈值: {threshold}"
            )
            return None, True

        except Exception as e:
            self.logger.error(f"BM25 检索异常: {e}")
            return None, True


# ============================================================================
# 模块入口：直接运行此脚本时，测试 BM25 检索功能
# 用法：python -m mysql_qa.bm25
# 或：  cd mysql_qa && python bm25.py
# ============================================================================
if __name__ == '__main__':
    bm25 = BM25Search()
    # 测试几个查询
    test_queries = [
        "什么是LPR？LPR多久调整一次？",
        "普惠贷款利率是多少？",
    ]
    for q in test_queries:
        answer, need_rerank = bm25.search(q)
        print(f"查询: {q}")
        print(f"答案: {answer}")
        print(f"需要Rerank: {need_rerank}")
        print("-" * 50)