import gradio as gr
import os
import time
from dotenv import load_dotenv
from app.core.rag_chain import prepare_vector_store, rag_qa, get_qa_chain, DeepSeekLLMWrapper
import logging

load_dotenv()
logger = logging.getLogger(__name__)

# 全局状态
state = {
    "qa_chain": None,
    "doc_metas": None,
    "ready": False
}


def init_system():
    try:
        DATA_DIR = os.getenv("DATA_DIR", "data")
        INDEX_PATH = os.getenv("INDEX_PATH", "vector_store")

        yield "🚀 正在初始化系统..."
        yield "🔄 加载向量知识库..."
        vector_store, doc_metas, _ = prepare_vector_store(DATA_DIR, INDEX_PATH)

        yield "🧠 初始化DeepSeek模型..."
        llm = DeepSeekLLMWrapper()
        state["qa_chain"] = get_qa_chain(vector_store, llm)
        state["doc_metas"] = doc_metas

        # 测试连接 - 使用新的 invoke 方法
        test_response = state["qa_chain"].invoke({"query": "你好，请回复'系统就绪'"})["result"]
        if "系统就绪" not in test_response:
            raise ConnectionError(f"模型连接失败: {test_response}")

        state["ready"] = True
        yield "✅ 系统初始化完成！"

    except Exception as e:
        state["ready"] = False
        logger.error(f"初始化失败: {str(e)}")
        yield f"❌ 初始化失败: {str(e)}"


def answer_question(question, history):
    if not state["ready"]:
        return "系统未初始化，请先点击'初始化系统'", []

    try:
        start_time = time.time()
        answer, sources = rag_qa(
            question,
            state["qa_chain"],
            state["doc_metas"]
        )
        response_time = time.time() - start_time

        # 格式化回答
        formatted_answer = f"{answer}\n\n---\n⏱️ 响应时间: {response_time:.2f}秒"
        return formatted_answer, sources

    except Exception as e:
        logger.error(f"回答问题错误: {str(e)}")
        return f"错误: {str(e)}", []


def format_source(source):
    """格式化来源文档显示"""
    return f"""
**来源**: {source['source']}
**路径**: {source['file_path']}
**内容**: {source['content'][:200]}{'...' if len(source['content']) > 200 else ''}
"""

with gr.Blocks(title="行业知识问答系统 - DeepSeek RAG", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧠 行业知识问答系统")
    gr.Markdown("基于DeepSeek RAG技术的垂直领域智能问答")

    with gr.Row():
        with gr.Column(scale=1):
            init_btn = gr.Button("初始化系统", variant="primary")
            init_status = gr.Textbox(label="系统状态", interactive=False)
        with gr.Column(scale=3):
            gr.Markdown("### 使用说明")
            gr.Markdown("1. 点击'初始化系统'按钮加载知识库和模型")
            gr.Markdown("2. 在下方输入框提问")
            gr.Markdown("3. 系统将基于行业知识库回答")

    with gr.Row():
        chatbot = gr.Chatbot(
            height=500,
            avatar_images=(
                os.path.join(os.path.dirname(__file__), "assets", "user.jpg"),
                os.path.join(os.path.dirname(__file__), "assets", "bot.png")
            ),
            type="messages"
        )

    msg = gr.Textbox(label="输入问题", placeholder="请输入行业相关问题...", lines=2)

    with gr.Row():
        submit_btn = gr.Button("提交问题", variant="primary")
        clear_btn = gr.ClearButton([msg, chatbot])

    with gr.Accordion("来源文档详情", open=False):
        source_display = gr.JSON(label="检索结果", show_label=False)  # 结构化展示

    gr.Examples(
        examples=["行业标准是什么？", "技术规范有哪些？", "设备操作流程是什么？"],
        inputs=msg,
        label="示例问题"
    )

    # 事件处理
    init_btn.click(
        init_system,
        outputs=init_status
    )


    def respond(question, chat_history):
        response, sources = answer_question(question, chat_history)
        # 使用新的消息格式
        chat_history.append({
            "role": "user",
            "content": question
        })
        chat_history.append({
            "role": "assistant",
            "content": response
        })
        return "", chat_history, sources


    msg.submit(
        respond,
        [msg, chatbot],
        [msg, chatbot, source_display]
    )

    submit_btn.click(
        respond,
        [msg, chatbot],
        [msg, chatbot, source_display]
    )

if __name__ == "__main__":
    demo.launch(
        server_port=8899,
        auth=(os.getenv("WEB_USER", "admin"), os.getenv("WEB_PASS", "password")),
        share=False,
        show_error=True
    )