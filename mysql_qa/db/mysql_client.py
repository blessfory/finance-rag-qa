# ============================================================================
# 模块：mysql_qa/db/mysql_client.py
# ============================================================================
# 【包的作用】
# - pymysql：纯 Python 实现的 MySQL 客户端，不需要安装 MySQL 原生驱动
# - 相比 mysql-connector-python：pymysql 更轻量，社区更活跃
# - 相比 SQLAlchemy：直接写 SQL 更灵活，适合本项目这种表结构简单的场景
# - pandas：Python 数据分析核心库，这里用于读取 CSV 和批量处理数据
#
# 【pymysql 核心 API 速查】
# ┌──────────────────────────────────────────────────────────────────┐
# │ API                                          │ 说明              │
# ├──────────────────────────────────────────────────────────────────┤
# │ pymysql.connect(**kwargs)                    │ 创建数据库连接      │
# │ conn.cursor()                                │ 创建游标对象        │
# │ cursor.execute(sql, params)                  │ 执行 SQL（参数化）  │
# │ cursor.executemany(sql, params_list)         │ 批量执行 SQL       │
# │ cursor.fetchone()                            │ 获取一行结果        │
# │ cursor.fetchall()                            │ 获取全部结果        │
# │ cursor.fetchmany(n)                          │ 获取 n 行结果      │
# │ cursor.rowcount                              │ 受影响的行数        │
# │ cursor.lastrowid                             │ 最后插入的自增 ID   │
# │ conn.commit()                                │ 提交事务           │
# │ conn.rollback()                              │ 回滚事务           │
# │ conn.close()                                 │ 关闭连接           │
# └──────────────────────────────────────────────────────────────────┘
#
# 【pymysql.connect() 参数详解】
# - host:       数据库主机地址
# - port:       端口号（MySQL 默认 3306）
# - user:       用户名
# - password:   密码
# - database:   数据库名
# - charset:    字符集，utf8mb4 支持 emoji 和生僻字
# - cursorclass: 游标类型，DictCursor 让查询结果返回字典而非元组
# - autocommit: 是否自动提交，默认 False（需要手动 commit）
# ============================================================================

import pymysql
import pandas as pd
from base import Config, setup_logger

# 模块级单例：Config 和 Logger 在整个项目中各只需实例化一次
# 其他模块通过 from base import config, logger 引用同一实例
config = Config()
logger = setup_logger()


class MySQLClient:
    """
    MySQL 客户端封装类

    封装了 MySQL 的常见操作：连接管理、自动建库、建表、CRUD、批量导入。

    【pymysql 事务机制】
    1. pymysql 默认 autocommit=False，即不自动提交
    2. SELECT 查询不需要 commit()，只读操作不产生事务
    3. 事务管理：INSERT/UPDATE/DELETE 需要手动 commit()，否则数据不会持久化
    4. 异常时调用 rollback() 回滚，保证数据一致性

    【使用方式】
        # 方式一：普通使用
        client = MySQLClient()
        client.create_table()
        client.insert_data('data.csv')
        questions = client.fetch_questions()
        client.close()

        # 方式二：上下文管理器（推荐，自动关闭连接）
        with MySQLClient() as client:
            client.create_table()
            questions = client.fetch_questions()
    """

    def __init__(self):
        """
        初始化 MySQL 连接

        【连接流程（三步走）】
        第一步：无数据库连接 → 先连上 MySQL 服务器（不指定 database）
        第二步：自动建库       → CREATE DATABASE IF NOT EXISTS（已存在则跳过）
        第三步：切换数据库     → 关闭临时连接，正式连接到目标数据库

        【为什么不是直接连接？】
        pymysql.connect(database='xxx') 要求数据库已存在，否则报错
        OperationalError: (1049, "Unknown database 'xxx'")
        所以在正式连接前，先用无数据库的连接去创建数据库。
        """
        self.logger = logger
        self.connection = None
        self.cursor = None

        try:
            # ==================================================================
            # 第一步：无数据库临时连接，用于创建数据库
            # 注意：这里不传 database 参数，只连到 MySQL 服务器本身
            # ==================================================================
            temp_conn = pymysql.connect(
                host=config.mysql_host,
                user=config.mysql_user,
                password=config.mysql_password,
                port=config.mysql_port,
                charset='utf8mb4'
            )
            temp_cursor = temp_conn.cursor()

            # ==================================================================
            # 第二步：创建数据库（如果不存在）
            # CREATE DATABASE IF NOT EXISTS：幂等操作，重复执行不会报错
            # utf8mb4：支持完整 Unicode（含 emoji），比 utf8（只支持基本多语言平面）更全
            # utf8mb4_unicode_ci：不区分大小写的排序规则，适合中文场景
            # ==================================================================
            temp_cursor.execute(
                f"CREATE DATABASE IF NOT EXISTS `{config.mysql_db}` "
                f"DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
            temp_cursor.close()
            temp_conn.close()
            self.logger.info(f"数据库 {config.mysql_db} 已就绪")

            # ==================================================================
            # 第三步：正式连接到目标数据库
            # 此时数据库一定存在，可以安全地指定 database 参数
            # ==================================================================
            self.connection = pymysql.connect(
                host=config.mysql_host,
                user=config.mysql_user,
                password=config.mysql_password,
                database=config.mysql_db,
                port=config.mysql_port,
                charset='utf8mb4'
            )
            self.cursor = self.connection.cursor()
            self.logger.info("MySQL 连接成功")
        except pymysql.MySQLError as e:
            self.logger.error(f"MySQL 连接失败: {e}")
            raise

    def __enter__(self):
        """
        上下文管理器入口：支持 with 语句

        【使用示例】
            with MySQLClient() as client:
                client.create_table()

        进入 with 块时调用此方法，返回自身供 as 后的变量接收。
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        上下文管理器出口：退出 with 块时自动关闭连接

        【参数说明】
        exc_type  — 异常类型（无异常时为 None）
        exc_val   — 异常值
        exc_tb    — traceback 对象

        【返回值】
        False — 不抑制异常，让异常继续向上传播
        """
        self.close()
        return False  # 不抑制异常

    def close(self):
        """
        关闭数据库连接，释放资源

        【为什么需要手动关闭？】
        - MySQL 有最大连接数限制（max_connections，默认 151）
        - 不关闭的连接会占用服务器资源，最终导致 "Too many connections" 错误
        - Python 垃圾回收不保证立即释放，手动关闭是最佳实践

        【关闭顺序】
        先关游标（cursor），再关连接（connection），和打开顺序相反。

        【幂等设计】
        关闭后将 self.connection 和 self.cursor 置为 None，
        确保重复调用 close()（包括 __del__ 中的调用）不会报 "Already closed" 错误。
        """
        if self.cursor:
            try:
                self.cursor.close()
            except Exception:
                pass
            self.cursor = None
            self.logger.info("游标已关闭")
        if self.connection:
            try:
                self.connection.close()
            except Exception:
                pass
            self.connection = None
            self.logger.info("数据库连接已关闭")

    def __del__(self):
        """
        析构函数：对象被垃圾回收时自动关闭连接

        【注意】
        不要完全依赖 __del__ 来关闭连接，因为：
        1. Python 不保证 __del__ 何时被调用（由 GC 决定）
        2. 循环引用可能导致 __del__ 永远不会被调用
        推荐使用 with 语句或手动调用 close()。

        【安全设计】
        close() 内部已做幂等处理，关闭后会将 connection 置为 None，
        重复调用不会报错。这里额外包裹 try-except 防止 __del__ 中任何异常泄露。
        """
        try:
            self.close()
        except Exception:
            pass

    # ========================================================================
    # 建表
    # ========================================================================

    def create_table(self):
        """
        创建 bank_faq 表 — 银行金融知识问答表

        【表结构说明】
        ┌────────────────┬──────────────┬──────────────────────────────────┐
        │ 字段            │ 类型         │ 说明                             │
        ├────────────────┼──────────────┼──────────────────────────────────┤
        │ id             │ INT AUTO_INCREMENT │ 主键，自增唯一标识            │
        │ category       │ VARCHAR(100) │ 知识分类，如"贷款知识"、"理财知识"  │
        │ question       │ TEXT         │ 问题内容，可能很长故用 TEXT        │
        │ answer         │ TEXT         │ 答案内容，可能很长故用 TEXT        │
        │ created_at     │ TIMESTAMP    │ 记录创建时间，默认当前时间          │
        └────────────────┴──────────────┴──────────────────────────────────┘

        【MySQL 数据类型速查】
        ┌───────────────┬──────────────────────────────────────────────────┐
        │ 类型          │ 说明                                             │
        ├───────────────┼──────────────────────────────────────────────────┤
        │ INT           │ 整数（-2,147,483,648 ~ 2,147,483,647）           │
        │ BIGINT        │ 大整数，适合大数据量主键                          │
        │ VARCHAR(N)    │ 可变长字符串，最大65535字节（取决于字符集和行大小） │
        │ TEXT          │ 长文本，最大65535字符（utf8mb4约21845个中文）      │
        │ MEDIUMTEXT    │ 更长的文本，约16MB                               │
        │ LONGTEXT      │ 超长文本，约4GB                                  │
        │ TIMESTAMP     │ 时间戳，范围1970-2038，受时区影响                  │
        │ DATETIME      │ 日期时间，范围1000-9999，不受时区影响              │
        └───────────────┴──────────────────────────────────────────────────┘

        【CREATE TABLE 关键子句】
        - IF NOT EXISTS      → 仅在表不存在时创建，避免重复执行报错
        - AUTO_INCREMENT     → 自增列，插入时无需手动指定值
        - PRIMARY KEY        → 主键约束，唯一且非空，默认建立聚簇索引
        - NOT NULL           → 非空约束，该列不允许 NULL 值
        - DEFAULT            → 默认值，插入时未指定该列则使用默认值
        - ENGINE=InnoDB      → 存储引擎，支持事务/行级锁/外键（MySQL 5.5+默认）
        - DEFAULT CHARSET    → 字符集，utf8mb4 支持 emoji 和全部 Unicode 字符
        - COMMENT            → 表/列注释，便于维护和理解
        - INDEX              │ 普通索引，加速查询（为 category 建索引，按分类筛选更快）
        """
        create_table_query = """
            CREATE TABLE IF NOT EXISTS bank_faq (
                id INT AUTO_INCREMENT PRIMARY KEY COMMENT '主键ID',
                category VARCHAR(100) NOT NULL COMMENT '知识分类',
                question TEXT NOT NULL COMMENT '问题内容',
                answer TEXT NOT NULL COMMENT '答案内容',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
                INDEX idx_category (category)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='银行金融知识问答表';
        """

        try:
            self.cursor.execute(create_table_query)
            self.connection.commit()
            self.logger.info("bank_faq 表创建成功")
        except pymysql.MySQLError as e:
            self.logger.error(f"bank_faq 表创建失败: {e}")
            self.connection.rollback()
            raise

    # ========================================================================
    # 数据写入
    # ========================================================================

    def insert_data(self, csv_path):
        """
        从 CSV 文件批量导入数据到 bank_faq 表

        【参数】
        csv_path: str — CSV 文件路径，支持相对路径和绝对路径

        【CSV 文件格式要求】
        必须包含以下列（列名必须完全匹配）：
        - 分类：知识分类，如"贷款知识"、"理财知识"（可选，缺失时默认"银行知识"）
        - 问题：问答对中的问题文本
        - 答案：问答对中的答案文本

        【为什么用 executemany 而不是循环 execute？】
        - executemany 将多条 SQL 打包成一次网络请求发送，减少网络往返次数
        - 对于大批量数据，性能差距可达数十倍
        - 但仍然在一个事务中执行，失败时可整体回滚

        【参数化查询（%s 占位符）的作用】
        1. 防止 SQL 注入：pymysql 会自动转义特殊字符
        2. 类型安全：不需要手动拼接字符串和加引号
        3. 性能优化：MySQL 可以缓存执行计划，重复使用
        注意：pymysql 的占位符是 %s，不是 ? 也不是 :param_name
        """
        try:
            data = pd.read_csv(csv_path)
            self.logger.info(f"成功读取 CSV 文件，共 {len(data)} 行数据")

            # 兼容有无"分类"列两种情况：有则用，无则默认"银行知识"
            if '分类' in data.columns:
                params_list = [
                    (row['分类'], row['问题'], row['答案'])
                    for _, row in data.iterrows()
                ]
            else:
                params_list = [
                    ('银行知识', row['问题'], row['答案'])
                    for _, row in data.iterrows()
                ]

            insert_query = "INSERT INTO bank_faq (category, question, answer) VALUES (%s, %s, %s)"

            # 幂等导入：先取出已有问题，重复的直接跳过，避免多次执行导致 FAQ 重复
            self.cursor.execute("SELECT question FROM bank_faq")
            existing = {row[0] for row in self.cursor.fetchall()}
            params_list = [p for p in params_list if p[1] not in existing]

            if not params_list:
                self.logger.info("所有问题已存在，跳过导入")
                return

            self.cursor.executemany(insert_query, params_list)

            self.connection.commit()
            self.logger.info(f"成功插入 {len(params_list)} 条数据（跳过重复 {len(existing)} 条）")
        except Exception as e:
            self.logger.error(f"数据插入失败: {e}")
            self.connection.rollback()
            raise

    # ========================================================================
    # 数据查询
    # ========================================================================

    def fetch_questions(self):
        """
        获取 bank_faq 表中所有的问题列表

        【返回值】
        list[tuple] — 问题列表，每个元素是单元素元组 ('问题内容',)
                      例如：[('什么是LPR？',), ('企业贷款有哪些担保方式？',)]

        【为什么返回元组而不是字典？】
        默认游标（Cursor）返回元组，内存占用小，适合简单查询。
        如果需要字典格式，可以在 connect 时指定 cursorclass=pymysql.cursors.DictCursor。

        【为什么 SELECT 不需要 commit？】
        SELECT 是只读操作，不修改数据，不产生事务。
        只有 INSERT、UPDATE、DELETE 等写操作才需要 commit/rollback。

        【异常处理策略：返回空列表而非抛出异常】
        查询失败时返回空列表 [ ] 而不是抛异常，调用方无需 try/except，
        直接用 for q in questions 遍历即可，空列表不会报错。
        """
        try:
            self.cursor.execute("SELECT question FROM bank_faq")
            results = self.cursor.fetchall()
            self.logger.info(f"成功获取 {len(results)} 条问题")
            return results
        except pymysql.MySQLError as e:
            self.logger.error(f"查询问题失败: {e}")
            return []

    def fetch_answer(self, question):
        """
        根据问题文本查询对应的答案

        【参数】
        question: str — 要查询的问题文本（精确匹配）

        【返回值】
        str 或 None — 匹配到的答案文本，未找到时返回 None

        【参数化查询详解】
        execute(sql, (question,)) 中的 (question,) 是单元素元组：
        - 逗号不能省略：没有逗号的 (question) 就是普通括号，不是元组
        - 作用：pymysql 会自动对 question 进行转义，防止 SQL 注入

        【fetchone 返回格式】
        - 查到数据：返回单行元组，如 ('Python是...',)
        - 没查到：返回 None
        所以用 result[0] 取第一个（也是唯一一个）字段的值。

        【为什么用精确匹配（=）而不是模糊匹配（LIKE）？】
        这里是从知识库中按问题查答案，问题是预先定义好的，应该精确匹配。
        如果未来需要模糊搜索，可以用 LIKE %keyword% 或全文索引（FULLTEXT INDEX）。
        """
        try:
            self.cursor.execute(
                'SELECT answer FROM bank_faq WHERE question = %s',
                (question,)
            )
            result = self.cursor.fetchone()
            if result:
                self.logger.info(f"成功获取答案：{question[:30]}...")
                return result[0]  # 取出元组中的第一个字段
            else:
                self.logger.warning(f"未找到问题对应的答案：{question[:30]}...")
                return None
        except pymysql.MySQLError as e:
            self.logger.error(f"查询答案失败: {e}")
            return None

    def fetch_all_qa(self):
        """
        获取所有问答对（问题 + 答案）

        【返回值】
        list[tuple] — 问答对列表，每个元素是 (问题, 答案) 二元组
                      例如：[('什么是LPR？', 'LPR是...'), ...]

        【使用场景】
        - 导出全部知识库数据
        - 构建本地缓存
        - 数据分析与统计
        """
        try:
            self.cursor.execute("SELECT question, answer FROM bank_faq")
            results = self.cursor.fetchall()
            self.logger.info(f"成功获取 {len(results)} 条问答对")
            return results
        except pymysql.MySQLError as e:
            self.logger.error(f"查询问答对失败: {e}")
            return []

    def count_qa(self):
        """
        统计 bank_faq 表中的总记录数

        【返回值】
        int — 记录总数

        【COUNT(*) vs COUNT(column)】
        - COUNT(*)：统计所有行数（包括 NULL），推荐
        - COUNT(column)：统计该列非 NULL 的行数，忽略 NULL
        """
        try:
            self.cursor.execute("SELECT COUNT(*) FROM bank_faq")
            result = self.cursor.fetchone()
            count = result[0]
            self.logger.info(f"bank_faq 表共有 {count} 条记录")
            return count
        except pymysql.MySQLError as e:
            self.logger.error(f"统计记录数失败: {e}")
            return 0


# ============================================================================
# 模块入口：直接运行此脚本时，自动建库建表并导入 CSV 数据
# 用法：python -m mysql_qa.db.mysql_client
# 或：  cd mysql_qa/db && python mysql_client.py
# ============================================================================
if __name__ == '__main__':
    # 使用上下文管理器，确保退出时自动关闭连接
    from pathlib import Path

    csv_path = Path(__file__).resolve().parents[1] / 'data' / '银行金融知识问答.csv'
    with MySQLClient() as client:
        # 第一步：建表（IF NOT EXISTS，重复执行不会报错）
        client.create_table()
        # 第二步：从 CSV 导入数据（路径相对本文件解析，不受工作目录影响）
        client.insert_data(str(csv_path))
        # 第三步：验证数据
        count = client.count_qa()
        print(f"导入完成，共 {count} 条记录")