import sys
print(f"Python 版本: {sys.version}")
assert sys.version_info >= (3, 9), "需要 Python 3.9 或更高版本"
print("✅ Python 版本检查通过")