from modelscope import snapshot_download

model_dir = snapshot_download(
    # 模型ID（和老师的代码一致）
    # 'Qwen/Qwen2.5-7B-Instruct',
    'deepseek-ai/DeepSeek-R1-Distill-Qwen-14B',
    # 下载根目录（模型会自动拼接到 D:\ModelScope\models/Qwen/Qwen2.5-7B-Instruct）
    cache_dir=r"D:\ModelScope\models",
    # 版本控制：用master下载最新版，也可以改成固定版本号
    revision='master'
)

print(f"✅ 模型下载完成，路径：{model_dir}")
print("💡 你可以直接用这个路径加载模型：AutoModel.from_pretrained(model_dir)")