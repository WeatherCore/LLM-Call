from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv("D:\\VS Code-wenjian\\大模型基础入门\\project_api\\.env")


def chat_loop(system_prompt="你是一位友好的AI助手，擅长回答各种问题。", max_rounds=3):
    """
    交互式多轮对话函数
    Args:
        system_prompt: 系统提示词，给模型定人设
        max_rounds: 最大对话轮数，防止对话太长
    Returns:
        conversation_history: 完整的对话历史列表
    """
    # 1. 初始化对话历史
    conversation_history = [
        {"role": "system", "content": system_prompt}
    ]
    
    # 2. 初始化OpenRouter客户端
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY")
    )
    
    print("=" * 40)
    print("多轮对话大模型开始")
    print("=" * 40)
    
    # 3. 循环对话，最多max_rounds轮
    for round_num in range(1, max_rounds + 1):
        # 获取用户输入，去掉前后空格
        user_input = input(f"\n[第{round_num}次对话] 我：").strip()
        
        # 退出机制：输入quit/exit/退出结束对话
        if user_input.lower() in ['quit', 'exit', '退出']:
            print("\n对话已结束")
            break
        
        # 输入验证：不能输入空内容
        if not user_input:
            print("输入不能为空，请重新输入")
            continue
        
        # 把用户消息加入对话历史
        conversation_history.append({"role": "user", "content": user_input})
        
        # 调用API，带错误处理
        try:
            response = client.chat.completions.create(
                model="nvidia/nemotron-3-super-120b-a12b:free",  # 你之前用的OpenRouter免费模型
                messages=conversation_history,
                temperature=0.7
            )
            
            # 提取AI的回复内容
            assistant_message = response.choices[0].message.content
            
            # 把AI回复加入对话历史
            conversation_history.append({"role": "assistant", "content": assistant_message})
            
            # 打印AI回复
            print(f"\nAI：{assistant_message}")
            
            # 打印Token消耗，方便统计
            print(f"📊 Token消耗：输入 {response.usage.prompt_tokens} | 输出 {response.usage.completion_tokens} | 总计 {response.usage.total_tokens}")
        
        except Exception as e:
            # 调用失败，打印错误信息
            print(f"\n❌ 调用失败：{e}")
            # 关键：把刚才加入的用户消息从历史里删掉，不然下一轮会出错
            conversation_history.pop()
            continue
    
    # 对话结束，返回完整的对话历史
    return conversation_history

# 调用函数开始聊天
if __name__ == "__main__":
    history = chat_loop(max_rounds=5)
    
    # 可选：对话结束后打印完整的对话历史
    print("\n" + "="*60)
    print("完整对话历史：")
    print("="*60)
    for msg in history:
        print(f"[{msg['role']}]：{msg['content']}")