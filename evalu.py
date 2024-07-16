import numpy as np
from train_mask2 import ImageDataset
from PIL import Image, ImageDraw, ImageChops
def precision(true, prediction, threshold=0.9):
    # 将预测值和真实值转换为二进制图像
    binary_mask = np.where(true > 0, 1, 0)
    binary_prediction = np.where(prediction > threshold, 1, 0)

    # 计算预测区域中接近白色像素的精确率
    true_positive = np.sum(np.logical_and(binary_mask, binary_prediction))
    predicted_positive = np.sum(binary_prediction)
    if predicted_positive == 0:
        return 0.0
    else:
        precision = true_positive / predicted_positive
        return precision

image_path='./deephomo_data/contact/train_images/1AZT.jpg'
text_path='./deephomo_data/contact/train_chains/1AZT.txt'
out_path='./output/1AZT.jpg'
true=Image.open(image_path).convert("L")
text = ImageDataset._load_text_file(text_path)
prediction=Image.open(out_path).convert("L")


p=precision(true, prediction, threshold=0.9)
print(p)