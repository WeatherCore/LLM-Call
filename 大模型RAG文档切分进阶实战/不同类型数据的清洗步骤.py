import re

# 【基础纯文本清洗】
def clean_text(text):
    # 去除多余空白：多个空格/换行/tab → 单个空格
    text = re.sub(r'\s+', ' ', text)
    # 修复断开的单词（PDF经典问题：换行把单词切两半，例：know-\nledge → knowledge）
    text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)
    # 标准化引号：把异形弯引号，统一成直引号（避免embedding把“”和""识别成不同语义）
    text = text.replace('“', '"').replace('”', '"')
    return text.strip()

# 【HTML网页清洗】
def clean_html(text):
    # 正则直接剔除全部<xxx>标签，只留下页面正文文字
    return re.sub(r'<.*?>', '', text)

# 【Markdown清洗（重点！只剥标记、保留文字本体）】
def clean_markdown(text):
    # 移除**粗体标记，保留中间文字
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    # 移除链接[文字](url)，只保留链接显示文字，扔掉url地址（url大多是噪声）
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    return text

# 安装pandas
# pip install pandas

import pandas as pd

def clean_table(df):
    if type(df) != pd.DataFrame:
        df = pd.DataFrame(df)

    # 去除完全空白的行和列
    df = df.dropna(how='all').dropna(axis=1, how='all')

    # 填充NaN值
    df = df.fillna('')

    # 删除完全重复的行 (不要使用 inplace=True，它会返回 None)
    df = df.drop_duplicates()

    # 标准化列名
    df.columns = [str(col).strip() for col in df.columns]

    return df

#安装依赖包
# pip install cv2 matplotlib
import cv2
from matplotlib import pyplot as plt

# 显示图像的函数
def show_image(img, title="Image", cmap=None):
    plt.figure(figsize=(4, 2))
    if cmap:
        plt.imshow(img, cmap=cmap)
    else:
        plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.show()

def clean_image(img_path):
    """
    图像降噪 - 去除扫描文档中的噪声点
    """
    # 读取图像
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print("原始图像:")
    show_image(img_rgb, "Original Image")

    # 方法1: 高斯模糊去噪
    gaussian_denoised = cv2.GaussianBlur(gray, (5, 5), 0)

    # 方法2: 中值滤波去噪 (对椒盐噪声效果好)
    median_denoised = cv2.medianBlur(gray, 3)

    # 方法3: 双边滤波 (保持边缘)
    bilateral_denoised = cv2.bilateralFilter(gray, 9, 75, 75)

    print("高斯模糊去噪:")
    show_image(gaussian_denoised, "Gaussian Denoised", cmap='gray')

    print("中值滤波去噪:")
    show_image(median_denoised, "Median Denoised", cmap='gray')

    print("双边滤波去噪:")
    show_image(bilateral_denoised, "Bilateral Denoised", cmap='gray')

    return gray, gaussian_denoised, median_denoised, bilateral_denoised

# 使用示例
# gray, gaussian, median, bilateral = clean_image("D:/1/formula.png")

import re

def clean_code(code_text, remove_comments=False):
    """
    增强版代码清洗
    """
    if not code_text:
        return ""

    cleaned_lines = []
    in_multiline_comment = False

    for line in code_text.split('\n'):
        # 移除行尾空白
        clean_line = line.rstrip()

        # 可选：移除单行注释
        if remove_comments:
            if not in_multiline_comment:
                # 检查是否进入多行注释
                if '"""' in clean_line or "'''" in clean_line:
                    in_multiline_comment = not in_multiline_comment
                    # 简单处理：直接跳过含有多行注释符号的行
                    continue
                # 移除单行注释（# 后面的内容）
                clean_line = re.sub(r'#.*$', '', clean_line)
            else:
                # 在多行注释中，跳过这行
                if '"""' in clean_line or "'''" in clean_line:
                    in_multiline_comment = False
                continue


        # 如果行不为空，或者我们保留空行（这里保留一个空行）
        if clean_line or (cleaned_lines and not cleaned_lines[-1]):
            cleaned_lines.append(clean_line)

    # 重新组合并确保首尾没有空行
    result = '\n'.join(cleaned_lines).strip()

    # 确保以换行符结束（可选）
    if result and not result.endswith('\n'):
        result += '\n'

    return result

# # 测试示例
# dirty_code = """
# def example_function():
#     # 这是一个示例函数
#     data = [1, 2, 3]
#
#     for item in data:
#         print(item)
#
#
#
#     return True
#
# """
# print(clean_code(dirty_code))


def clean_mixed_content(content_type, content):
        """
        统一清洗入口
        content_type: 'text', 'table', 'image', 'code'
        """
        try:
            if content_type == 'text':
                text = clean_text(content)
                text = clean_html(text)
                text = clean_markdown(text)
                return text

            elif content_type == 'table':
                return clean_table(content)

            elif content_type == 'image':
                return clean_image(content)

            elif content_type == 'code':
                return clean_code(content)

            else:
                print(f"未知内容类型: {content_type}")
                return content

        except Exception as e:
            print(f"清洗 {content_type} 时出错: {e}")
            return content

def batch_clean(documents):
    """
    批量清洗文档
    documents: 列表，每个元素是 (content_type, content) 元组
    """
    results = []
    for i, (doc_type, content) in enumerate(documents):
        print(f"清洗文档 {i+1}: {doc_type}")
        cleaned = clean_mixed_content(doc_type, content)
        results.append((doc_type, cleaned))

    return results