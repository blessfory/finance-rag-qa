# ============================================================================
# 模块：app/main.py
# 顶层入口：交互式命令行问答
# 内部流程：意图识别 → FAQ 关键词匹配 → RAG 向量检索 → LLM 生成回答
# ============================================================================

import os
import sys

from base.config import Config
from base.logger import setup_logger

cfg = Config()
logger = setup_logger('RAG.main', log_file=cfg.log_file)


def ask(question: str) -> str:
    """两级检索融合：意图识别 → 分流处理 → 返回回答"""
    logger.info("收到问题：%s", question)

    try:
        from rag.rag_service import RAGService
        rag = RAGService()
        result = rag.ask(question)
        rag.close()
        return result['answer']
    except ImportError as e:
        logger.error("模块导入失败：%s", e)
        return "系统初始化失败，请检查依赖安装。"
    except Exception as e:
        logger.error("处理失败：%s", e, exc_info=True)
        return f"处理出错：{e}"


def main():
    """交互式命令行入口"""
    logger.info("=" * 60)
    logger.info("RAG 智能问答系统启动")
    logger.info("=" * 60)

    print("=" * 60)
    print("欢迎使用 RAG 智能问答系统")
    print("流程：意图识别 → FAQ匹配 → RAG检索 → LLM生成")
    print("输入 'exit' 退出")
    print("=" * 60)

    while True:
        try:
            question = input("\n请输入您的问题: ").strip()
            if question.lower() == 'exit':
                break
            if not question:
                continue

            answer = ask(question)
            print(f"\n答案: {answer}")

        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error("处理问题出错：%s", e, exc_info=True)
            print(f"\n出错：{e}")

    print("\n感谢使用，再见！")
    logger.info("系统退出")


if __name__ == '__main__':
    main()