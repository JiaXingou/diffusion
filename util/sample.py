from PIL import Image
import numpy as np

def matrix_to_image(matrix):
    # 将矩阵转换为NumPy数组
    array = np.array(matrix, dtype=np.uint8) * 255

    # 创建图像对象
    image = Image.fromarray(array, 'L')

    # 保存图像为.jpg格式
    image.save('output.jpg')

# 示例矩阵
matrix = [[0, 1, 1, 0],
          [1, 0, 0, 1],
          [1, 1, 1, 0],
          [0, 1, 0, 1]]

# 将矩阵转换为图像并保存为.jpg格式
matrix_to_image(matrix)