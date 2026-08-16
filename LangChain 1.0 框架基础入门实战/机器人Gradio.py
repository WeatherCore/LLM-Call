import gradio as gr
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage,AIMessage

load_dotenv()

# ──────────────────────────────────────────────
# 1. 初始化模型与系统设定
# ──────────────────────────────────────────────
model = init_chat_model(
    model="qwen3.7-max",
    model_provider="openai",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

system_message = SystemMessage(
    content="你叫Weather，是一名乐于助人的智能助手。请在对话中保持友好、有耐心、温和的语气。"
)

# ──────────────────────────────────────────────
# 2. 定义 Gradio 界面
# ──────────────────────────────────────────────
CSS = """
.main-container {max-width: 1200px; margin: 0 auto; padding: 20px;}
.header-text {text-align: center; margin-bottom: 20px;}
"""

def create_chatbot() -> gr.Blocks:
    with gr.Blocks(title="DeepSeek Chat",css=CSS) as demo:
        with gr.Column(elem_classes=["main-container"]):
            gr.Markdown("# 🤖 LangChain 1.0 × DeepSeek Chatbot", elem_classes=["header-text"])
            gr.Markdown("基于 LangChain 1.0 标准接口的流式对话机器人", elem_classes=["header-text"])

            chatbot = gr.Chatbot(
                height=500,
                show_copy_button=True,
                avatar_images=(
                    "https://cdn.jsdelivr.net/gh/twitter/twemoji@v14.0.2/assets/72x72/1f464.png",
                    "https://cdn.jsdelivr.net/gh/twitter/twemoji@v14.0.2/assets/72x72/1f916.png",
                ),
            )
            msg = gr.Textbox(placeholder="请输入您的问题...", container=False, scale=7)
            submit = gr.Button("发送", scale=1, variant="primary")
            clear = gr.Button("清空", scale=1)

        # 状态：保存消息历史（LangChain Message 对象）
        state = gr.State([])

        # ─────────────── 主响应函数（流式输出） ───────────────
        def respond(user_msg: str, chat_hist: list, messages_list: list):

            # 1️⃣ 输入为空则直接返回
            if not user_msg.strip():
                yield "", chat_hist, messages_list
                return

            # 2️⃣ 构建消息上下文（包括系统提示）
            if not messages_list:
                messages_list = [system_message]

            messages_list.append(HumanMessage(content=user_msg))

            # 3️⃣ 添加用户消息到聊天历史
            chat_hist = chat_hist + [(user_msg, "")]

            # 4️⃣ 流式生成模型回复
            partial = ""
            for chunk in model.stream(messages_list):
                if chunk.content:
                    partial += chunk.content
                    # 每次更新最后一条消息
                    chat_hist[-1] = (user_msg, partial)
                    # 立即 yield，让 UI 实时更新
                    # 返回空字符串给 msg，清空输入框
                    yield "", chat_hist, messages_list

            # 5️⃣ 保存完整 AI 回复并截断历史（保留50轮）
            messages_list.append(AIMessage(content=partial))
            messages_list = messages_list[-50:]

            # 6️⃣ 最后一次 yield 确保状态同步
            yield "", chat_hist, messages_list

        # ─────────────── 清空对话函数 ───────────────
        def clear_history():
            return "", [], []

        # ─────────────── Gradio 事件绑定 ───────────────
        # 返回 msg、chatbot 和 state
        # msg 返回空字符串来清空输入框
        msg.submit(respond, [msg, chatbot, state], [msg, chatbot, state])
        submit.click(respond, [msg, chatbot, state], [msg, chatbot, state])
        clear.click(clear_history, outputs=[msg, chatbot, state])

    return demo

# ──────────────────────────────────────────────
# 3. 启动 Gradio 应用
# ──────────────────────────────────────────────

print("\n🚀 启动 Gradio 应用...")
demo = create_chatbot()
demo.launch(server_name="0.0.0.0", server_port=7860, share=False, debug=True)