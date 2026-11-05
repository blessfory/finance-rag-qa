# ============================================================================
# 模块：mysql_qa/sql_main.py
# ============================================================================
# 【模块定位】
# MySQL 知识问答系统（FAQ 问答）的主入口。
# 本模块是 RAG 系统两级检索策略中的第一级——BM25 关键词精确匹配。
# 负责接收用户查询、调用 BM25 检索、输出结果，以及串联整个 mysql_qa 子系统的各组件。
#
# 【两级检索策略】
# ┌──────────────────────────────────────────────────────────────────────┐
# │ 第一级：BM25 关键词匹配（本模块）→ 快速、低成本、精确关键词匹配        │
# │ 第二级：RAG 向量检索（外部模块）   → 语义理解、同义词、需要推理        │
# └──────────────────────────────────────────────────────────────────────┘
#
# 【工作流程】
# 1. 用户输入问题
# 2. MySQLQASystem.query() 调用 BM25Search.search() 进行关键词匹配
# 3. BM25Search 内部：Redis 缓存检查 → 分词 → BM25 打分 → Softmax 归一化 → 阈值过滤
# 4. 命中 → 返回答案；未命中 → 标记 need_rerank，提示需要调用向量检索（RAG）
#
# 【使用方式】
#     python sql_main.py          # 交互式命令行问答
#     python -m mysql_qa.sql_main # 作为模块运行
#
# 【依赖关系】
#     sql_main.py
#     └── MySQLQASystem
#         └── BM25Search     → BM25 关键词检索 + Softmax 归一化 + 阈值过滤
#             ├── MySQLClient    → 加载原始问题、获取答案
#             ├── RedisClient    → 缓存问题列表、分词结果、问答结果
#             └── preprocess_text → jieba 中文分词
# ============================================================================

# ============================================================================
# 导入依赖
# ============================================================================
# 标准库
import sys
from typing import Optional

# 项目内部模块
from mysql_qa.bm25 import BM25Search
from base import setup_logger


class MySQLQASystem:
    """
    MySQL 知统问答系统

    封装了 FAQ 问答的完整流程：初始化组件 → 接收查询 → BM25 检索 → 返回结果。
    所有数据库和缓存操作均由 BM25Search 内部管理，本类只负责流程编排。

    【属性】
    - logger: logging.Logger — 日志记录器，用于记录系统运行日志
    - bm25_search: BM25Search — BM25 关键词检索器（核心检索引擎，内部持有 MySQL 和 Redis 连接）

    【使用示例】
        system = MySQLQASystem()
        answer = system.query("Python是什么？")
        print(answer)

        # 使用上下文管理器（推荐）
        with MySQLQASystem() as system:
            answer = system.query("如何学习Java？")
    """

    def __init__(self) -> None:
        """
        初始化 MySQL 知统问答系统

        【初始化步骤】
        1. 初始化日志记录器
        2. 初始化 BM25 检索器（内部自动连接 MySQL 和 Redis、加载数据、构建索引）
        """
        self.logger = setup_logger()
        self.bm25_search = BM25Search()

    def __enter__(self) -> 'MySQLQASystem':
        """
        上下文管理器入口，支持 with 语句

        【使用示例】
            with MySQLQASystem() as system:
                answer = system.query("Python是什么？")
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """
        上下文管理器出口，退出 with 块时自动释放资源

        【参数】
        exc_type  — 异常类型（无异常时为 None）
        exc_val   — 异常值
        exc_tb    — traceback 对象

        【返回值】
        False — 不抑制异常，让异常继续向上传播
        """
        self.close()
        return False

    def query(self, query: str, threshold: float = 0.85) -> str:
        """
        执行 FAQ 问答查询

        【参数】
        query: str — 用户输入的问题
        threshold: float — BM25 置信度阈值（0~1），默认 0.85
                  只有 Softmax 归一化后最高分 ≥ threshold 的才返回答案
                  调高 → 更精确但可能漏掉；调低 → 更宽松但可能不准确

        【返回值】
        str — 匹配到的答案，或提示信息

        【检索流程】
        1. 参数校验：检查 query 是否为有效的非空字符串
        2. BM25 检索：调用 BM25Search.search() 进行关键词匹配
           - 内部先查 Redis 缓存（相同问题直接返回）
           - 分词 → BM25 打分 → Softmax 归一化 → 阈值过滤
        3. 结果处理：
           - 命中（need_rerank=False）→ 返回答案
           - 未命中（need_rerank=True）→ 提示需要调用 RAG 向量检索

        【使用示例】
            system = MySQLQASystem()
            result = system.query("Python是什么？")
            # → "Python 是一种解释型、面向对象的高级编程语言..."
            result = system.query("Python是什么？", threshold=0.95)
            # → 使用更严格的阈值
        """
        # ======================================================================
        # 第一步：参数校验
        # 检查 query 是否为有效的非空字符串
        # ======================================================================
        if not query or not isinstance(query, str):
            self.logger.error(f"无效查询参数: {query}")
            return "请输入有效的问题"

        # 去除首尾空白字符，避免空字符串或纯空格输入
        query = query.strip()
        if not query:
            self.logger.error("查询参数为空字符串")
            return "请输入有效的问题"

        # ======================================================================
        # 第二步：记录查询信息
        # ======================================================================
        self.logger.info(f"处理查询: '{query}'")

        # ======================================================================
        # 第三步：BM25 关键词检索
        # search() 返回 (answer, need_rerank)：
        # - answer: 匹配到的答案（str）或 None
        # - need_rerank: True 表示 BM25 置信度不足，需要 RAG 向量检索
        # ======================================================================
        answer, need_rerank = self.bm25_search.search(query, threshold=threshold)

        # ======================================================================
        # 第四步：结果处理
        # ======================================================================
        if need_rerank:
            # BM25 未找到可靠答案，置信度低于阈值
            self.logger.info(
                f"SQL中未找到可靠答案（阈值: {threshold}），建议调用 RAG 向量检索系统"
            )
            return None
        elif answer:
            # BM25 命中，返回答案
            self.logger.info(f"FAQ 答案: {answer[:100]}...")
            return answer
        else:
            # 理论上不会走到这里（search 返回 answer=None 时 need_rerank 应为 True）
            self.logger.warning("BM25 检索返回了未预期的结果")
            return "未找到相关答案"

    def close(self) -> None:
        """
        释放系统资源

        关闭 BM25Search 内部持有的 MySQL 和 Redis 连接。

        【为什么需要手动关闭？】
        - MySQL 有最大连接数限制（max_connections，默认 151）
        - Redis 连接也会占用服务器资源
        - 不关闭的连接最终会导致资源耗尽
        - Python 垃圾回收不保证立即释放，手动关闭是最佳实践
        """
        try:
            self.bm25_search.close()
        except Exception as e:
            self.logger.error(f"关闭 BM25 检索器资源时出错: {e}")


def main() -> None:
    """
    MySQL 知统问答系统的主入口函数

    【功能】
    1. 初始化 MySQLQASystem（连接数据库、Redis、构建 BM25 索引）
    2. 进入交互式命令行循环，等待用户输入问题
    3. 调用 system.query() 获取答案并输出
    4. 支持 'exit' 命令退出
    5. 支持 Ctrl+C 优雅退出

    【使用示例】
        python sql_main.py
        # 欢迎使用 MySQL 知统问答系统
        # 请输入您的问题: Python是什么？
        # Python 是一种解释型、面向对象的高级编程语言...
    """
    mysql_system: Optional[MySQLQASystem] = None

    try:
        # ======================================================================
        # 初始化系统
        # ======================================================================
        mysql_system = MySQLQASystem()
        mysql_system.logger.info("=" * 60)
        mysql_system.logger.info("MySQL 知统问答系统启动")
        mysql_system.logger.info("=" * 60)

        print("=" * 60)
        print("欢迎使用 MySQL 知统问答系统（FAQ 关键词匹配）")
        print("输入 'exit' 或按 Ctrl+C 退出")
        print("=" * 60)

        # ======================================================================
        # 交互式命令行循环
        # ======================================================================
        while True:
            try:
                query = input("\n请输入您的问题: ").strip()

                # 检查退出命令
                if query.lower() == "exit":
                    break

                # 跳过空输入
                if not query:
                    continue

                # 执行查询
                answer = mysql_system.query(query)
                print(f"\n答案: {answer if answer else '未命中，建议向量检索'}")

            except KeyboardInterrupt:
                # Ctrl+C → 退出循环
                print("\n")
                break
            except EOFError:
                # Ctrl+D（Windows 上的 Ctrl+Z）→ 退出循环
                print("\n")
                break

        print("\n再见！")

    except KeyboardInterrupt:
        # 初始化阶段按 Ctrl+C → 静默退出
        print("\n系统已取消启动")
        sys.exit(0)

    except Exception as e:
        # 未预期的系统错误
        if mysql_system and mysql_system.logger:
            mysql_system.logger.error(f"系统错误: {e}", exc_info=True)
        print(f"发生错误: {e}")

    finally:
        # ======================================================================
        # 资源清理：确保无论如何都关闭连接
        # ======================================================================
        if mysql_system:
            mysql_system.close()
            # 注意：logging.Logger 没有 close() 方法，不需要手动关闭
            # 如需关闭日志文件 handler，可在 setup_logger 中配置


# ============================================================================
# 模块入口
# 用法：python sql_main.py 或 python -m mysql_qa.sql_main
# ============================================================================
if __name__ == '__main__':
    main()