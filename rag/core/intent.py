# ============================================================================
# 模块：rag/core/intent.py
# 意图识别：判断用户问题类型，决定走哪条处理路径
# ============================================================================
# 【为什么需要意图识别？】
# 不是所有用户输入都需要走 RAG 检索。比如：
#   "你好"                → 闲聊，直接回复，不需要检索
#   "今天天气"            → 超出知识库范围，不应该浪费算力检索
#   "兴业银行2024年净利润" → 金融知识问答，需要 RAG 检索
#
# 意图识别的作用就是"路由"——把不同的问题分流到不同的处理逻辑。
# 好处：
#   1. 节省算力：闲聊不触发检索，省 GPU + 省 Milvus 查询
#   2. 节省 Token：不需要把无关上下文喂给 LLM
#   3. 提升体验：闲聊回复更快，专业问题回复更准
#   4. 安全控制：识别投诉/敏感问题，转人工处理
#
# 【实现方式对比】
#   ┌──────────────────┬──────────────────┬──────────────────┐
#   │ 方式              │ 优点              │ 缺点              │
#   ├──────────────────┼──────────────────┼──────────────────┤
#   │ 关键词匹配        │ 极快、零成本       │ 准确率低，覆盖不全 │
#   │ 正则表达式        │ 快、可定制         │ 维护成本高        │
#   │ 小模型分类        │ 准确率高、成本低   │ 需额外训练        │
#   │ LLM 判断          │ 最灵活、最准确     │ 有延迟、有成本    │
#   │ 混合策略 ← 本模块  │ 兼顾速度和准确率   │ 实现稍复杂        │
#   └──────────────────┴──────────────────┴──────────────────┘
#
# 本模块采用"混合策略"：
#   第一步：关键词/正则快速匹配（命中了直接返回，不调 LLM）
#   第二步：LLM 兜底判断（关键词没命中时，调 LLM 做意图分类）
# ============================================================================

import re
from typing import Literal, Optional

from base.config import Config
from base.logger import setup_logger

cfg = Config()
logger = setup_logger('RAG.intent', log_file=cfg.log_file)

# 意图类型（用 Literal 做类型约束，IDE 可以自动补全）
IntentType = Literal['chitchat', 'faq', 'knowledge', 'complaint', 'other']


class IntentRecognizer:
    """
    意图识别器：混合策略（关键词快速匹配 + LLM 兜底判断）

    【使用方式】
        recognizer = IntentRecognizer()
        intent = recognizer.recognize("你好")                    # → 'chitchat'
        intent = recognizer.recognize("兴业银行2024年净利润")     # → 'knowledge'
        intent = recognizer.recognize("LPR利率是多少")           # → 'faq'
        intent = recognizer.recognize("我要投诉！")               # → 'complaint'
    """

    # ========================================================================
    # 关键词规则（快速匹配层）
    # 面试点：为什么用关键词而不是全部交给 LLM 判断？
    #   答：降低延迟和成本。闲聊类问题占比高（约30%），关键词匹配
    #       可以在毫秒级完成，不需要调 LLM。只有关键词没命中时才调 LLM。
    # ========================================================================

    # 闲聊关键词：打招呼、问候、感谢、告别等
    CHITCHAT_PATTERNS = [
        r'^(你好|嗨|hi|hello|hey|在吗|在不在|在线吗)[\s!！。.]*$',
        r'^(早上好|下午好|晚上好|晚安|早安|午安)[\s!！。.]*$',
        r'^(谢谢|感谢|多谢|3q|thank|thx)[\s!！。.]*$',
        r'^(再见|拜拜|bye|88|回头见)[\s!！。.]*$',
        r'^(你是谁|你叫什么|你是什么|你的名字)[\s!！。.?？]*$',
        r'^(你能做什么|你会什么|你有什么功能)[\s!！。.?？]*$',
        r'^[哈哈嘿嘿嗯哦啊哎哟呵]{1,5}[!！。.]*$',
    ]

    # FAQ 关键词：常见金融高频问题，有标准答案
    FAQ_PATTERNS = [
        r'(怎么|如何|怎样)(办理|申请|开户|贷款|还款|转账)',
        r'(利率|费率|利息|LPR|手续费|年费|多少钱)',
        r'(额度|期限|条件|要求|门槛)',
        r'(网点|营业厅|地址|在哪里|怎么走)',
        r'(联系电话|客服电话|联系方式|人工客服)',
        r'(理财|基金|存款|保险|信用卡|借记卡)',
    ]

    # 投诉关键词
    COMPLAINT_PATTERNS = [
        r'(投诉|举报|差评|退款|赔偿|维权|坑|骗)',
        r'(太差|垃圾|不好|失望|生气|愤怒)',
        r'(我要找你们领导|我要投诉|我要曝光)',
    ]

    def __init__(self, llm_service=None):
        """
        【参数】
            llm_service: LLMService — 可选的 LLM 服务，用于兜底判断。
                        如果为 None，当关键词匹配失败时，默认返回 'knowledge'
        """
        self.llm = llm_service

    def recognize(self, question: str) -> IntentType:
        """
        识别用户意图

        【参数】
            question: str — 用户输入的问题

        【返回值】
            IntentType — 'chitchat' | 'faq' | 'knowledge' | 'complaint' | 'other'

        【内部流程】
            1. 关键词匹配（毫秒级）
            2. 命中 → 直接返回意图
            3. 未命中 + 有 LLM → LLM 判断
            4. 未命中 + 无 LLM → 默认 'knowledge'
        """
        question = question.strip()

        # 第一层：关键词匹配
        intent = self._keyword_match(question)
        if intent:
            logger.info("关键词匹配意图：%s → %s", question[:30], intent)
            return intent

        # 第二层：LLM 兜底判断
        if self.llm:
            intent = self._llm_classify(question)
            logger.info("LLM 判断意图：%s → %s", question[:30], intent)
            return intent

        # 默认：走知识问答
        logger.info("默认意图：%s → knowledge", question[:30])
        return 'knowledge'

    def _keyword_match(self, question: str) -> Optional[IntentType]:
        """关键词匹配：按优先级依次匹配，命中即返回"""
        for pattern in self.CHITCHAT_PATTERNS:
            if re.search(pattern, question, re.IGNORECASE):
                return 'chitchat'

        for pattern in self.COMPLAINT_PATTERNS:
            if re.search(pattern, question, re.IGNORECASE):
                return 'complaint'

        for pattern in self.FAQ_PATTERNS:
            if re.search(pattern, question, re.IGNORECASE):
                return 'faq'

        return None

    def _llm_classify(self, question: str) -> IntentType:
        """LLM 兜底判断：调用 LLM 做意图分类"""
        from rag.prompts import get_prompt

        prompt = get_prompt('intent_classify', question=question)
        try:
            result = self.llm.generate(
                [{"role": "user", "content": prompt}],
                temperature=0.0,  # 意图识别不要随机性，temperature=0 最稳定
                max_tokens=20,
            )
            result = result.strip().lower()
            valid_intents = {'chitchat', 'faq', 'knowledge', 'complaint', 'other'}
            if result in valid_intents:
                return result
        except Exception as e:
            logger.warning("LLM 意图判断失败：%s", e)

        return 'knowledge'


# ============================================================================
# 便捷函数：一行代码做意图识别
# ============================================================================

def recognize_intent(question: str, llm_service=None) -> IntentType:
    """
    便捷函数：一行代码识别意图

    【使用示例】
        from rag.core.intent import recognize_intent
        intent = recognize_intent("你好")
        if intent == 'chitchat':
            return "你好！有什么可以帮助你的吗？"
    """
    return IntentRecognizer(llm_service).recognize(question)