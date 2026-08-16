from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 改成你自己的模型文件夹路径！
model_path = r"D:\ModelScope\models\Qwen\Qwen2___5-7B-Instruct"

# 加载翻译官分词器
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 加载模型本体
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True
)

# 验证是否加载成功
print(f"✅ 模型加载成功！运行设备: {model.device}")


prompt = "你好，请用有水平地介绍你自己。"
messages = [
    {"role": "system", "content": "你是一个有用的 AI 助手。"},
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=512,
    temperature=0.7
)

generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("\n问题：", prompt)
print("回答：", response)