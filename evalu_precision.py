#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image, ImageDraw, ImageChops

def evaluate_precision(predicted_image, ground_truth_image, mask_image, n):
    # Apply mask to the predicted and ground truth images
    predicted_masked = predicted_image * (1-mask_image)
    ground_truth_masked = ground_truth_image * (1-mask_image)

    # Sort predicted_masked in descending order of brightness
    sorted_indices = np.argsort(predicted_masked, axis=None)[::-1]  # Sort in descending order

    # Get the top n brightest positions
    top_n_indices = np.unravel_index(sorted_indices[:n], predicted_masked.shape)

    # Check if the ground truth values at the top n positions are white pixels
    correct_count = 0
    for i in range(n):
        if ground_truth_masked[top_n_indices[0][i], top_n_indices[1][i]] == 1.0:
            print(f"Position: ({top_n_indices[0][i]}, {top_n_indices[1][i]})")
            correct_count += 1
    
    # Calculate accuracy
    precision = correct_count / n

    return precision
class process():
    def pad_image(image):
        #width, height = image.size
        #max_size = max(width, height)

        padded_image = Image.new("L", (512, 512), color=0)
        padded_image.paste(image, (0, 0))

        #resized_image = padded_image.resize((512, 512))
        return padded_image
    def pro_mask(text):
        # 计算对角线方形区域的边界
        start_x = 0
        start_y = 0
        end_x = 0
        end_y = 0
        # 创建掩码图像
        mask = Image.new("L", (512, 512), color=0)
        for length in text:
            end_x = min(end_x+length, image_size[1])
            end_y = min(end_y+length, image_size[0])
        # 在图像上绘制对角线方形区域
        #cv2.rectangle(image, (start_x, start_y), (end_x, end_y), (255, 255, 255), -1)   
            # 在掩码上绘制对角线方形区域
            draw = ImageDraw.Draw(mask)
            draw.rectangle([(start_x, start_y), (end_x, end_y)], fill=255)
            #cv2.rectangle(mask, (start_x, start_y), (end_x, end_y), 255, -1)
            start_x = end_x
            start_y = end_y
            if start_x>=512 and start_y>=512:
                break
        return mask
    def _load_text_file(text_path):
        with open(text_path, "r") as file:
            text = file.read()
        text = text.strip()  # 去除首尾的空格和换行符
        text = text.split()  # 按空格分割为多个字符串
        text = [int(num) for num in text]  # 将字符串转换为整数
        return text
# 生成示例数据
if __name__ == None:
    # 示例图像
    predicted_image = np.array([[0.8, 0.9, 0.7],
                            [0.6, 1, 0.4],
                            [0.3, 0.2, 0.1]])

    ground_truth_image = np.array([[1.0, 0.0, 1.0],
                               [0.0, 1.0, 0.0],
                               [1.0, 0.0, 1.0]])

    mask_image = np.array([[0, 0, 0],
                       [0, 1, 0],
                       [0, 0, 0]])

    n = 3

# 调用 evaluate_precision 函数
    precision = evaluate_precision(predicted_image, ground_truth_image, mask_image, n)

    print("Precision:", precision)

if __name__ == '__main__':
    image_size = (512, 512)
    image_path='./deephomo_data/contact/valid_images/1AZT.jpg'
    text_path='./deephomo_data/contact/valid_chains/1AZT.txt'
    preimage_path='./output/1AZT.jpg'
    predicted_image = Image.open(preimage_path).convert("L")
    image = Image.open(image_path).convert("L")
    re_image=process.pad_image(image) 
    text = process._load_text_file(text_path)
    mask=process.pro_mask(text)
    #####
    predicted_image = np.array(predicted_image)
    predicted_image = predicted_image/ 255.0
    image = np.array(re_image)
    ground_truth_image = image/ 255.0
    mask = np.array(mask)
    mask_image = (mask > 0).astype(int)
    n = 50  
# 前10个位置
    precision = evaluate_precision(predicted_image, ground_truth_image, mask_image, n)
    print("precision:", precision)