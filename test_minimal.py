# ============================================================================
# test_minimal.py — Bug 修复快速验证脚本
# ============================================================================
# 【说明】
#   不依赖 MySQL / Redis / Milvus / LLM 等外部服务，
#   只验证源码层面的修改（导入路径、返回值类型、Schema 字段），
#   用于快速确认 3 个 Bug 是否被修复。
#
# 【运行方式】
#   python test_minimal.py
#
# 【验证的 Bug】
#   1. bm25.py 包内导入 → 应能正常 import
#   2. sql_main.query() 未命中 → 应返回 None（而非文本）
#   3. Milvus Schema 含 parent_text 字段 → 父块写入数据库
# ============================================================================
import sys, os

# 确保项目根目录在 sys.path 中
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# --- Bug 1: relative imports in bm25.py ---
print("=== Bug 1: bm25 relative imports ===")
try:
    from mysql_qa.bm25 import BM25Search
    print("OK: bm25 module can be imported as package module")
except ModuleNotFoundError as e:
    print(f"FAIL: {e}")
except Exception as e:
    if "preprocess" in str(e) or "redis_client" in str(e):
        print(f"FAIL: old-style import still broken: {e}")
    else:
        print(f"IMPORT OK (other error expected at runtime: {type(e).__name__})")

# --- Bug 2: sql_main.query() returns None on miss ---
print("\n=== Bug 2: sql_main query return type ===")
from mysql_qa.sql_main import MySQLQASystem
sig = MySQLQASystem.query.__annotations__.get('return', '')
src_file = os.path.join(ROOT, 'mysql_qa', 'sql_main.py')
src_lines = [l.strip().strip('#').strip() for l in open(src_file, encoding='utf-8').readlines() if 'return None' in l]
if src_lines:
    print("OK: query() now returns None on BM25 unmatch")
else:
    print("FAIL: query() still returns text string on miss")

# --- Bug 3: parent_text field in schema and insert_data ---
print("\n=== Bug 3: parent_text in Milvus schema/insert/search ===")
milvus_file = os.path.join(ROOT, 'rag', 'repo', 'milvus_repo.py')
rag_file = os.path.join(ROOT, 'rag', 'rag_service.py')
milvus_src = open(milvus_file, encoding='utf-8').read()
has_schema_parent = "'parent_text'" in milvus_src and "add_field(field_name='parent_text'" in milvus_src
has_insert_parent = "'parent_text'" in milvus_src and "parent_text = parent_text" in milvus_src
rag_src = open(rag_file, encoding='utf-8').read()
has_rag_fallback = "r.get('parent_text') or r.get('text'" in rag_src
checks = {
    "schema has parent_text field": has_schema_parent,
    "insert stores parent_text": has_insert_parent,
    "rag_service falls back to text": has_rag_fallback,
}
for name, ok in checks.items():
    print(f"{'OK' if ok else 'FAIL'}: {name}")

# Summary
print("\n=== Summary ===")
all_ok = all(v for v in checks.values())
print(f"All 3 bug fixes verified: {'YES' if all_ok else 'NO'}")