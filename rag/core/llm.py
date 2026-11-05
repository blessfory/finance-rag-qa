# ============================================================================
# 模块：rag/core/llm.py
# LLM 调用服务：兼容 OpenAI 格式，用于生成回答
# ============================================================================
# 【OpenAI Python SDK 是什么？】
# OpenAI 官方提供的 Python 库，用于调用 OpenAI 的大模型 API。
# 核心对象：OpenAI 客户端，通过 chat.completions.create() 发起对话。
#
# 【为什么用 OpenAI SDK 而不是直接发 HTTP 请求？】
# 1. 自动重试：网络抖动时自动重试，减少手动处理
# 2. 流式输出：支持 SSE（Server-Sent Events），实现逐字输出效果
# 3. 类型安全：完整的类型提示，IDE 自动补全
# 4. 兼容性好：兼容所有 OpenAI 格式的 API（通义千问、DeepSeek、Moonshot 等）
#
# 【OpenAI SDK 核心 API】
#   client.chat.completions.create(
#       model="模型名",             # 必填：模型名称
#       messages=[...],            # 必填：对话消息列表
#       temperature=0.3,           # 可选：温度参数（0-2），越高越随机
#       max_tokens=1024,           # 可选：最大输出 token 数
#       top_p=0.9,                 # 可选：核采样参数
#       stream=False,              # 可选：是否流式输出
#   )
#
# 【messages 参数详解】
#   messages 是一个列表，每个元素是一个 Dict，包含 role 和 content：
#   [
#       {"role": "system", "content": "你是一个...（系统指令）"},
#       {"role": "user", "content": "用户问题"},
#       {"role": "assistant", "content": "LLM 之前的回答（多轮对话时用）"},
#   ]
#
#   role 三种类型：
#   - system：系统级指令，定义 LLM 的行为和角色（最高优先级）
#   - user：用户输入
#   - assistant：LLM 的历史回答（多轮对话记忆）
#
# 【temperature 参数详解】
#   温度控制输出的随机性，值域 0-2：
#   - 0.0：确定性输出，每次回答完全一样（适合意图识别、代码生成）
#   - 0.3-0.5：适度随机，保持准确性（适合 RAG 回答）
#   - 0.7-0.9：较随机，更有创意（适合闲聊、写作）
#   - 1.0+：非常随机，可能出现幻觉（不推荐用于 RAG）
# ============================================================================

from typing import List, Dict, Optional

from openai import OpenAI

from base.config import Config
from base.logger import setup_logger

cfg = Config()
logger = setup_logger('RAG.llm', log_file=cfg.log_file)


class LLMService:
    """
    LLM 调用服务：封装大模型对话 API，生成 RAG 最终回答。

    【使用方式】
        llm = LLMService()
        answer = llm.answer_with_context("问题", ["上下文1", "上下文2"])
        # 或直接调用
        messages = [{"role": "user", "content": "你好"}]
        answer = llm.generate(messages)

    【支持切换模型】
        llm = LLMService(model="qwen-plus")   # 用通义千问 Plus
        llm = LLMService(model="gpt-4")       # 用 GPT-4
        # 只要 base_url 兼容 OpenAI 格式即可
    """

    def __init__(
        self,
        model: str = None,
        api_key: str = None,
        base_url: str = None,
    ):
        """
        【参数】
            model: str — 模型名称，默认从 config.ini 读取（qwen3.7-max）
            api_key: str — API 密钥，默认从 config.ini 读取
            base_url: str — API 基础 URL，默认阿里云 DashScope 兼容模式
                         兼容 OpenAI 格式的都可以用，如：
                         - DeepSeek: https://api.deepseek.com/v1
                         - Moonshot: https://api.moonshot.cn/v1
                         - 本地 vLLM: http://localhost:8000/v1
        """
        self.model = model or cfg.llm_model
        self.client = OpenAI(
            api_key=api_key or cfg.dashscope_api_key,
            base_url=base_url or cfg.dashscope_base_url,
        )
        logger.info("LLM 服务已初始化：model=%s", self.model)

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> str:
        """
        调用 LLM 生成回答

        【参数】
            messages: List[Dict] — 对话消息，格式 [{"role": "...", "content": "..."}]
            temperature: float — 温度参数，0=确定性，1=随机性（默认 0.3）
            max_tokens: int — 最大输出 token 数（默认 1024）

        【返回值】
            str — LLM 生成的回答文本，失败时返回空字符串

        【异常处理】
            LLM 调用可能因网络、限流、余额不足等原因失败，
            这里捕获异常并记录日志，返回空字符串让上层处理。
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error("LLM 调用失败：%s", e)
            return ""

    def answer_with_context(
        self,
        query: str,
        contexts: List[str],
        prompt_template: str = None,
    ) -> str:
        """
        根据检索到的上下文生成回答（RAG 核心方法）

        【参数】
            query: str — 用户问题
            contexts: List[str] — 检索到的上下文列表
            prompt_template: str — 自定义 prompt 模板（可选），
                              如果不传则使用默认模板

        【返回值】
            str — LLM 生成的回答

        【内部流程】
            1. 如果没传模板，从 prompts.py 获取默认 RAG 模板
            2. 拼接上下文（用 --- 分隔多个上下文）
            3. 填充模板变量（context + question）
            4. 调用 LLM 生成回答
        """
        if prompt_template is None:
            from rag.prompts import get_prompt
            prompt_template = get_prompt(
                'rag_qa',
                context='\n\n---\n\n'.join(contexts),
                question=query,
            )
        else:
            prompt_template = prompt_template.format(
                context='\n\n---\n\n'.join(contexts),
                question=query,
            )

        messages = [{"role": "user", "content": prompt_template}]
        return self.generate(messages)