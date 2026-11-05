# ============================================================================
# 模块：redis_client.py
# ============================================================================
# 【包的作用】
# - redis（redis-py）：Python 操作 Redis 的官方推荐客户端库
#   - 官网：https://github.com/redis/redis-py
#   - 安装：pip install redis
#   - 核心类：StrictRedis（命令严格遵循 Redis 官方语法）、Redis（兼容旧版）
#   - 从 redis-py 3.0+ 起，StrictRedis 和 Redis 已合并为同一个类，
#     两者可以互换使用，推荐使用 redis.Redis
# - json：Python 标准库，用于 JSON 数据的序列化与反序列化
#   - 核心 API：json.dumps()（Python对象→JSON字符串）、json.loads()（JSON字符串→Python对象）
# ============================================================================

import redis
import json
from base import Config, setup_logger

# ============================================================================
# 全局配置与日志实例
# 模块级单例，所有 RedisClient 实例共享同一个 config 和 logger
# ============================================================================
config = Config()
logger = setup_logger()


# ============================================================================
# 类：RedisClient — Redis 缓存操作封装
# ============================================================================
class RedisClient:
    """
    Redis 客户端封装类

    封装了 Redis 的常见操作：连接管理、JSON 数据存取、问答缓存存取。

    【redis.StrictRedis / redis.Redis 连接参数详解】
    ┌────────────────────┬──────────┬──────────────────────────────────────┐
    │ 参数               │ 默认值    │ 说明                                  │
    ├────────────────────┼──────────┼──────────────────────────────────────┤
    │ host               │ localhost│ Redis 服务器地址                       │
    │ port               │ 6379     │ Redis 端口号                           │
    │ db                 │ 0        │ 数据库编号（0~15，Redis 默认16个库）      │
    │ password           │ None     │ 认证密码（生产环境必须设置）              │
    │ decode_responses   │ False    │ True=自动将bytes解码为str（推荐开启）     │
    │ socket_timeout     │ None     │ 套接字超时时间（秒），None=永不超时       │
    │ socket_connect_timeout │ None │ 连接超时时间（秒）                      │
    │ max_connections    │ None     │ 连接池最大连接数（需配合ConnectionPool）   │
    │ retry_on_timeout   │ False    │ 超时后是否重试                          │
    │ ssl                │ False    │ 是否启用 SSL/TLS 加密                   │
    └────────────────────┴──────────┴──────────────────────────────────────┘

    【redis.set() 方法参数详解】
    ┌────────────────────┬──────────┬──────────────────────────────────────┐
    │ 参数               │ 说明                                           │
    ├────────────────────┼──────────┼──────────────────────────────────────┤
    │ name               │ 键名（str）                                    │
    │ value              │ 值（str/bytes/int/float），非字符串会自动转换     │
    │ ex                 │ 过期时间（秒），如 ex=60 表示60秒后过期           │
    │ px                 │ 过期时间（毫秒），如 px=5000 表示5秒后过期        │
    │ nx                 │ True=仅当key不存在时才设置（SET NX，分布式锁常用）  │
    │ xx                 │ True=仅当key已存在时才设置（SET XX，更新场景）     │
    │ keepttl           │ True=保留key原有的TTL（Redis 6.0+）              │
    │ get                │ True=返回旧值（Redis 6.2+）                    │
    └────────────────────┴──────────┴──────────────────────────────────────┘

    【redis.get() 方法】
    - 参数：name（键名）
    - 返回值：键存在时返回对应值（str/None取决于decode_responses），不存在返回 None
    - 注意：返回 None 时需区分"键不存在"和"值为空字符串"

    【json.dumps() 参数详解】
    ┌────────────────────┬──────────┬──────────────────────────────────────┐
    │ 参数               │ 默认值    │ 说明                                  │
    ├────────────────────┼──────────┼──────────────────────────────────────┤
    │ obj                │ (必填)    │ 要序列化的 Python 对象                  │
    │ ensure_ascii       │ True     │ False=允许输出非ASCII字符（如中文）      │
    │ indent             │ None     │ 缩进空格数，None=紧凑输出               │
    │ separators         │ (', ',': ') │ 分隔符，可设为(',', ':')减少体积     │
    │ default            │ None     │ 自定义不可序列化对象的处理函数           │
    │ sort_keys          │ False    │ True=按key排序输出                     │
    └────────────────────┴──────────┴──────────────────────────────────────┘

    【json.loads() 参数详解】
    ┌────────────────────┬──────────┬──────────────────────────────────────┐
    │ 参数               │ 默认值    │ 说明                                  │
    ├────────────────────┼──────────┼──────────────────────────────────────┤
    │ s                  │ (必填)    │ 要解析的 JSON 字符串                   │
    │ object_hook        │ None     │ 自定义dict→对象的转换函数（如转OrderedDict）│
    │ parse_float        │ float    │ 自定义浮点数解析函数（如Decimal）        │
    │ parse_int          │ int      │ 自定义整数解析函数                      │
    └────────────────────┴──────────┴──────────────────────────────────────┘
    """

    def __init__(self):
        """
        初始化 Redis 连接

        【连接流程】
        1. 使用 config 中的 Redis 配置创建连接
        2. decode_responses=True 确保存取中文等非ASCII字符时自动编解码
        3. 连接失败时记录错误日志并抛出异常，让调用方感知
        """
        self.logger = logger

        try:
            self.client = redis.Redis(
                host=config.redis_host,
                port=config.redis_port,
                db=config.redis_db,
                password=config.redis_password,
                decode_responses=True
            )
            self.client.ping()
            self.logger.info('Redis 连接成功')
        except redis.RedisError as e:
            self.logger.error(f"Redis 连接失败: {e}")
            raise

    def set_data(self, key, value, ex=None):
        """
        存储 JSON 数据到 Redis

        【参数】
            key: str — 存储的键名
            value: Any — 要存储的 Python 对象（dict/list/str等），会自动转为 JSON
            ex: int | None — 过期时间（秒），None 表示永不过期

        【使用示例】
            client.set_data('user:1001', {'name': '张三', 'age': 25}, ex=3600)
            # 1小时后过期

        【返回值】
            bool — True 表示存储成功，False 表示失败
        """
        try:
            self.client.set(key, json.dumps(value, ensure_ascii=False), ex=ex)
            self.logger.info(f"存储数据到Redis: {key}")
            return True
        except redis.RedisError as e:
            self.logger.error(f"Redis存储失败: {e}")
            return False

    def get_data(self, key):
        """
        从 Redis 获取并解析 JSON 数据

        【参数】
            key: str — 要获取的键名

        【返回值】
            Any | None — 解析后的 Python 对象，key 不存在或出错时返回 None

        【使用示例】
            data = client.get_data('user:1001')
            # data = {'name': '张三', 'age': 25}
        """
        try:
            data = self.client.get(key)
            if data:
                data = json.loads(data)
                self.logger.info(f"从Redis获取数据: {key}")
                return data
            else:
                self.logger.warning(f"Redis中键不存在: {key}")
                return None
        except redis.RedisError as e:
            self.logger.error(f"Redis获取失败: {e}")
            return None

    def delete_data(self, key):
        """
        删除指定键

        【参数】
            key: str — 要删除的键名

        【返回值】
            int — 实际删除的键数量（0=键不存在，1=删除成功）

        【redis.delete() 说明】
            - 支持批量删除：delete('key1', 'key2', 'key3')
            - 返回值为实际删除的键数量（因为有些键可能不存在）
        """
        try:
            result = self.client.delete(key)
            self.logger.info(f"删除Redis键: {key}, 结果: {result}")
            return result
        except redis.RedisError as e:
            self.logger.error(f"Redis删除失败: {e}")
            return 0

    def exists(self, key):
        """
        检查键是否存在

        【参数】
            key: str — 要检查的键名

        【返回值】
            bool — True 表示键存在，False 表示不存在

        【redis.exists() 说明】
            - 支持批量检查：exists('key1', 'key2') 返回存在的键数量
        """
        try:
            return bool(self.client.exists(key))
        except redis.RedisError as e:
            self.logger.error(f"Redis exists 查询失败: {e}")
            return False

    def close(self):
        """
        关闭 Redis 连接

        【说明】
            程序退出前应调用此方法释放连接资源。
            对于长时间运行的服务，建议使用连接池（ConnectionPool）管理连接。
        """
        try:
            self.client.close()
            self.logger.info("Redis 连接已关闭")
        except redis.RedisError as e:
            self.logger.error(f"Redis 关闭连接失败: {e}")


# ============================================================================
# 测试入口
# ============================================================================
def main():
    redis_client = RedisClient()

    # 测试 JSON 数据存取
    redis_client.set_data('test_key', {'test_value': 'Hello Redis!'})
    data = redis_client.get_data('test_key')
    if data:
        print(f"JSON数据获取成功: {data}")
    else:
        print('未获取到数据')

    # 测试问答缓存存取（直接用 set_data/get_data，键名加 answer: 前缀）
    redis_client.set_data('answer:你好', '你好！我是RAG金融知识助手，有什么可以帮你的？', ex=3600)
    answer = redis_client.get_data('answer:你好')
    if answer:
        print(f"缓存答案获取成功: {answer}")
    else:
        print('未获取到缓存答案')

    # 测试键是否存在
    if redis_client.exists('test_key'):
        print("test_key 存在")

    # 测试删除
    redis_client.delete_data('test_key')
    if not redis_client.exists('test_key'):
        print("test_key 已删除")

    # 关闭连接
    redis_client.close()


if __name__ == '__main__':
    main()