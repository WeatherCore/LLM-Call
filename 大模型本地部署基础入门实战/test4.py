# 验证transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
print("✅ transformers安装成功")

# 验证PyTorch GPU
import torch
print("✅ PyTorch CUDA可用:", torch.cuda.is_available())

# 验证accelerate
from accelerate import Accelerator
print("✅ accelerate安装成功")