# ============================================================================
# 模块：base/__init__.py
# ============================================================================
# 【__init__.py 的作用】
# 1. 标识一个目录是 Python 包（Package），使该目录下的模块可以被 import
# 2. 控制 from package import * 时导出哪些内容（通过 __all__ 列表）
# 3. 可以在包被导入时执行初始化逻辑
#
# 【Python 包导入机制】
# - import base               → 执行 base/__init__.py，然后 base 可作为命名空间使用
# - from base import Config   → 导入 base 包中的 Config 类
# - from base import *        → 导入 __init__.py 中 __all__ 列表指定的所有内容
#
# 【为什么这里用 import * ？】
# 方便项目内其他模块只需 from base import * 就能拿到 Config 类和 logger 函数，
# 减少重复的 import 语句。但注意：import * 在大型项目中不推荐，因为会造成命名空间污染。
# 这里项目规模可控，且 base 模块只导出核心工具，因此可以接受。
# ============================================================================

from base.config import Config
from base.logger import setup_logger, get_logger

# __all__ 控制 from base import * 时导出哪些名称
# 如果不定义 __all__，则默认导出所有不以 _ 开头的名称
__all__ = ['Config', 'setup_logger', 'get_logger']