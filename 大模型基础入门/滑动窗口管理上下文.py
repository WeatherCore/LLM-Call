from openai import OpenAI
import os
from dotenv import load_dotenv

# 加载你的.env文件（路径和你之前的一致）
load_dotenv("D:\\VS Code-wenjian\\大模型基础入门\\project_api\\.env")

# 初始化OpenRouter客户端
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

# ---------------------- 滑动窗口管理函数 ----------------------
def manage_conversation_history(history, max_turns=5):
    """使用滑动窗口管理对话历史，控制Token消耗"""
    # 分离system消息和对话消息
    system_messages = [msg for msg in history if msg["role"] == "system"]
    dialog_messages = [msg for msg in history if msg["role"] != "system"]
    
    # 只保留最近max_turns轮对话（每轮2条消息：user+assistant）
    recent_messages = dialog_messages[-(max_turns * 2):] if dialog_messages else []
    
    # 重新拼接：system + 最近对话
    return system_messages + recent_messages

# ---------------------- 带滑动窗口的多轮对话 ----------------------
def chat_loop(
    system_prompt="你是一位友好的AI助手，回答简洁易懂。",
    max_rounds=10,
    max_turns=5  # 滑动窗口：只保留最近5轮对话
):
    # 初始化对话历史
    conversation_history = [
        {"role": "system", "content": system_prompt}
    ]
    
    print("=" * 60)
    print("多轮对话开始（输入 'quit' 退出）")
    print("=" * 60)
    
    for round_num in range(1, max_rounds + 1):
        # 获取用户输入
        user_input = input(f"\n[轮次 {round_num}] 你：").strip()
        
        # 退出机制
        if user_input.lower() in ['quit', 'exit', '退出']:
            print("\n对话已结束")
            break
        
        # 输入验证
        if not user_input:
            print("输入不能为空，请重新输入")
            continue
        
        # 把用户消息加入历史
        conversation_history.append({"role": "user", "content": user_input})
        
        # 调用API前，先压缩对话历史（关键！）
        compressed_history = manage_conversation_history(conversation_history, max_turns=max_turns)
        
        try:
            # 用压缩后的历史调用API
            response = client.chat.completions.create(
                model="nvidia/nemotron-3-super-120b-a12b:free",
                messages=compressed_history,
                temperature=0.7,
                max_tokens=300
            )
            
            # 提取AI回复，加入原始对话历史
            assistant_message = response.choices[0].message.content
            conversation_history.append({"role": "assistant", "content": assistant_message})
            
            print(f"AI：{assistant_message}")
            print(f"📊 Token消耗：输入 {response.usage.prompt_tokens} | 输出 {response.usage.completion_tokens} | 总计 {response.usage.total_tokens}")
            
        except Exception as e:
            print(f"❌ 调用失败：{e}")
            # 出错时移除刚加的用户消息
            conversation_history.pop()
            continue
    
    return conversation_history

# 运行聊天机器人
if __name__ == "__main__":
    history = chat_loop(max_rounds=10, max_turns=10)
    # 可选：打印最终压缩后的对话历史
    print("\n" + "="*60)
    print("压缩后的对话历史：")
    print("="*60)
    compressed = manage_conversation_history(history, max_turns=5)
    for msg in compressed:
        print(f"[{msg['role']}]：{msg['content']}")





