from dotenv import load_dotenv
from app.core.rag_chain import prepare_vector_store, rag_qa, get_qa_chain, DeepSeekLLMWrapper
import os
import logging

# 添加日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()


def main():
    # 调试：打印 API 密钥
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if api_key:
        print(f"✅ 已加载 DeepSeek API 密钥: {api_key[:6]}...{api_key[-4:]}")
    else:
        print("⚠ 警告: 未找到 DEEPSEEK_API_KEY 环境变量")
        logger.warning("未找到 DEEPSEEK_API_KEY 环境变量")

    DATA_DIR = os.getenv("DATA_DIR", "data")
    INDEX_PATH = os.getenv("INDEX_PATH", "vector_store")

    # 验证数据目录是否存在
    if not os.path.exists(DATA_DIR):
        logger.error(f"数据目录不存在: {DATA_DIR}")
        print(f"❌ 错误: 数据目录不存在: {DATA_DIR}")
        return

    print("正在初始化系统...")

    try:
        vector_store, doc_metas, embedding_model = prepare_vector_store(DATA_DIR, INDEX_PATH)
        llm = DeepSeekLLMWrapper()
        qa_chain = get_qa_chain(vector_store, llm)
    except Exception as e:
        logger.exception("初始化失败")
        print(f"❌ 初始化失败: {str(e)}")
        return

    print("✅ 系统初始化完成！输入 'exit' 退出")
    while True:
        try:
            question = input("\n问题: ")
            if question.lower() == 'exit':
                break

            answer, sources = rag_qa(question, qa_chain, doc_metas)

            print(f"\n答案: {answer}")
            if sources:
                print("\n来源文档:")
                for i, doc in enumerate(sources):
                    print(f"{i + 1}. {doc['source']}")
                    print(f"   内容: {doc['content'][:100]}...")
            else:
                print("⚠ 未找到相关来源")
        except Exception as e:
            logger.error(f"处理问题时出错: {str(e)}")
            print(f"⚠ 处理问题时出错: {str(e)}")


if __name__ == "__main__":
    main()