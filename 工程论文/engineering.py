from PIL import Image
import win32com.client

def resize_emf(input_emf_path, output_emf_path, scale_factor):
    # 打开EMF文件
    img = Image.open(input_emf_path)

    # 计算新的尺寸
    new_size = (int(img.width * scale_factor), int(img.height * scale_factor))

    # 调整图像大小
    resized_img = img.resize(new_size, Image.ANTIALIAS)

    # 保存为EMF格式
    resized_img.save(output_emf_path, format='EMF')

# 使用示例
resize_emf('input.emf', 'output.emf', 0.5)