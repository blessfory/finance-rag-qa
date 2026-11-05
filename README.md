# 金融知识问答 RAG 系统

面向银行年报与产品资料的问答系统。用户在自然语言提问后，系统先做意图识别，再分流到
**FAQ 精确匹配**或**向量检索 + 重排**两条链路，最后由 LLM 生成带出处的回答。

## 架构

```
用户提问
   │
   ├─ 意图识别（正则关键词优先，未命中交 LLM 判定）
   │     ├── 闲聊 / 投诉 ──────────────→ 直接回复 / 安抚转人工
   │     ├── FAQ ──→ MySQL(bank_faq) + BM25 + Redis 缓存 ──→ 命中即返回（约 0.4s）
   │     └── 知识问答
   │
   ├─ Query 改写（LLM 把口语问题改写为检索友好的查询）
   │
   ├─ 向量检索：BGE-M3 → Milvus（IVF_FLAT / 内积，nprobe=16），粗排 Top-10
   │
   ├─ 精排：bge-reranker-large 重排，取 Top-3
   │
   ├─ 父子块回溯：命中子块 → 取所属父块作为上下文
   │
   └─ Prompt 组装 → LLM 生成 → 返回答案 + 出处（文件名、页码、分数）
```

## 技术栈

| 层次 | 选型 |
|---|---|
| Embedding | BGE-M3（1024 维，本地部署） |
| 向量库 | Milvus 2.6（standalone，IVF_FLAT + IP） |
| 重排 | bge-reranker-large（FlagEmbedding） |
| 关键词检索 | BM25 + Softmax 归一化 + 阈值过滤 |
| 关系库 / 缓存 | MySQL 8（FAQ 语料）、Redis（问题级缓存） |
| LLM | OpenAI 兼容接口（当前接自建 MaaS 网关） |
| 服务层 | FastAPI / Streamlit / CLI 三个入口 |

## 关键设计

**父子分块**：父块 1500 字符、子块 400 字符、重叠 80。子块入向量库做检索；父块随子块
一并存入 `parent_text` 字段，命中后回溯父块喂给 LLM，兼顾检索精度与上下文完整性。

**两阶段检索**：BGE-M3 粗排召回 10 条，Reranker 精排取 3 条。实测中真正相关的 2024 年报
利润表段落由 0.7766 提升到 0.9979，并重排了后续候选的先后顺序。

**FAQ 前置**：FAQ 走 MySQL + BM25，比向量检索快一个数量级。规则上非闲聊、非投诉的问题
都先试 FAQ，未命中自动降级到向量检索——两条链路互补，而不是互斥。

## 快速开始

```bash
# 1. 起依赖服务（Milvus / MySQL / Redis）
cd <docker-compose.yml 所在目录>
docker compose up -d

# 2. 配 API Key（写入用户环境变量，代码优先读环境变量）
set DASHSCOPE_API_KEY=sk-xxxx

# 3. 导入 FAQ 语料（首次执行）
python -m mysql_qa.db.mysql_client

# 4. 构建向量库（首次执行，或换语料后重建）
python -m rag.pipeline

# 5. 启动服务（三选一）
python app/main.py                   # CLI
uvicorn app.api:app --port 8000      # FastAPI，/docs 查看接口
streamlit run app/ui.py              # Web 界面
```

## 实测数据

运行环境：Python 3.11 / Milvus 2.6.21 / MySQL 8.0.40 / Redis 7.4，语料为兴业银行与联化科技
2022–2025 年年报（8 份 PDF）及行业知识、产品手册。

| 指标 | 数值 |
|---|---|
| 向量库规模 | 7,938 个父子块 |
| 检索 Top-1 命中 | 2024 年报利润表，score 0.7766 |
| 精排后得分 | 0.9979 |
| FAQ 命中延迟 | 0.4 s |
| 知识问答延迟 | 22–23 s（含 LLM 生成） |
| 端到端用例 | 6 条回归覆盖闲聊 / 投诉 / FAQ / 知识四类 |

完整的运行日志与修复记录见 [docs/运行测试记录.md](docs/运行测试记录.md)。

## 测试

```bash
python -m unittest discover -s tests -v
# Ran 7 tests ... OK
```

`tests/test_bug_fixes.py` 针对三个真实缺陷做回归：包内裸 import、FAQ 未命中时返回文案而非
`None`、父块只存内存未落库。`test_minimal.py` 是源码级快速自检脚本，不是单元测试。

## 相对基线版本的改动

**代码结构**：按 core / repo / loaders / evaluation 分层重新组织模块，配置集中到 `config.ini`，
统一日志与异常处理，补齐类型注解。

**功能与修复**：

- **FAQ 通道前置**：原实现只在意图判定为 `faq` 时才查 FAQ，导致「银行承兑汇票是什么」这类改写
  问法被绕过；改为 `faq` 与 `knowledge` 都先查 FAQ，未命中再降级到向量检索。
- **FAQ 阈值 0.85 → 0.70**：对 0.85 / 0.75 / 0.70 / 0.65 逐档实测后确定，兼顾命中率与误命中
  （「兴业银行 2024 年营业收入」在 0.70 下仍正确降级为知识问答）。
- **FAQ 导入幂等**：修掉重复执行导致数据翻倍的问题（改为按问题文本去重）。
- **BERT 权重优先读本地 HF 缓存**：修掉离线环境下联网检查卡死的问题。
- **补 7 条单元测试**：覆盖包内裸 import、FAQ 未命中返回 None、父块落库三个真实缺陷。

> 检索架构（两级检索、父子分块、BGE-M3 + Milvus + BGE-Reranker）沿用基线设计，未做改动。

## 已知限制

- FAQ 语料仅 17 条，覆盖面窄，未命中时依赖向量检索。
- 知识问答首问约 22 s，主要耗时在模型冷启动与 LLM 生成长答案；未做流式输出。
- 评估目前是文件级粒度（是否命中目标 PDF），库只有 8 份文档时指标区分度有限。
- 财报分析中的「事实抽取 / 指标计算 / 跨年对比」已移除，仅保留可验证的检索链路。
- 无鉴权、无限流，CORS 目前放开，仅适用于本地演示。
