# 查看已安装的依赖包版本
import importlib.metadata

# 定义需要检查的包列表
packages = ['openai', 'transformers', 'tiktoken', 'python-dotenv', 'requests', 'httpx']

# 循环检查每个包的版本
for package in packages:
    try:
        version = importlib.metadata.version(package)
        print(f"{package:<20} v{version}")
    except importlib.metadata.PackageNotFoundError:
        print(f"{package:<20} 未安装")