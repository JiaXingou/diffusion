# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

# 读取loss数据
loss_file = '/home/u2022103607/diffusion/latent-diffusion/loss/deephomo/false/epoch_loss.txt'
losses = []

with open(loss_file, 'r') as file:
    for line in file:
        # 将每一行转换为float并添加到losses列表中
        losses.append(float(line.strip()))

# 绘制训练loss图像并保存为JPEG文件
plt.figure(figsize=(10, 6))
plt.plot(losses, label='Training Loss')
plt.xlabel('Training Steps')
plt.ylabel('Loss')
plt.title('Deephomo Training Loss')
plt.legend()
plt.grid(True)

# 保存图像为JPEG文件
plt.savefig('Deephomo_Training_Loss.jpg', format='jpg')

# 显示图像
plt.show()
