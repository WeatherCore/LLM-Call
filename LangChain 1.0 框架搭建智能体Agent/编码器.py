import tiktoken

# 获取编码器
enc = tiktoken.encoding_for_model("gpt-4o-mini")
# 文本转token列表
tokens = enc.encode("你好，我叫陈明")
print("token数量：", len(tokens))
# token还原文字
text = enc.decode(tokens)
print(text)