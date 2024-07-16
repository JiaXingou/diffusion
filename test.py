from diffusers import DiffusionPipeline
import torch
#from text_encoder import text_encoder_pretrained
#from vision_auto_encoder import vae_pretrained
#from unet import unet_pretrained
from unet import UNet
from PIL import Image, ImageDraw, ImageChops
import numpy as np
import warnings
from torchvision import transforms
from vision_auto_encoder import VAE
from evalu_precision import evaluate_precision
import os

warnings.filterwarnings('ignore')
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
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# diffusion model 噪声生成器
pipeline = DiffusionPipeline.from_pretrained('./pretrained-params/', safety_checker=None, local_files_only=True,tokenizer=None)
scheduler = pipeline.scheduler
tokenizer = pipeline.tokenizer
del pipeline

print('Device :', device)
print('Scheduler settings: ', scheduler)
print('Tokenizer settings: ', tokenizer)
#vae.requires_grad_(False)
#unet.requires_grad_(True)
# 模型加载
#text_encoder = text_encoder_pretrained()
vision_encoder = VAE()
#unet = unet_pretrained()
unet = UNet()
# 加载unet模型参数
#unet.load_state_dict(torch.load('./model_params/deephomo/last_checkpoint.pth.tar')['state_dict'])
vision_encoder.load_state_dict(torch.load('./model_params/deephomo_vae/last_checkpoint.pth.tar')['state_dict'])
unet.load_state_dict(torch.load('./model_params/deephomo/last_checkpoint.pth.tar')['state_dict'])
#text_encoder.eval()
vision_encoder.eval()
unet.eval()

#text_encoder.to(device)
vision_encoder.to(device)
unet.to(device)

# 根据文本生成图像
@torch.no_grad()
def generate(masked, flag):
    # 词编码 [1, 77]
    #pos = tokenizer(text, padding='max_length', max_length=77, 
                    #truncation=True, return_tensors='pt').input_ids.to(device)
    #neg = tokenizer('', padding='max_length', max_length=77,
                    #truncation=True, return_tensors='pt').input_ids.to(device)
    
    #pos_out = text_encoder(pos) # (1, 77, 768)
    #neg_out = text_encoder(neg) # -
    #text_out = torch.cat((neg_out, pos_out), dim=0) # (2, 77, 768)
    masked = masked.unsqueeze(0).to(device)
    mask_out = vision_encoder.encoder(masked)
    mask_out = vision_encoder.sample( mask_out)
    mask_out = mask_out * 0.18215
    #zeros_tensor = np.zeros_like(mask_out)
    #mask_out = torch.cat((mask_out, zeros_tensor), dim=0)
    # 全噪声图
    vae_out = torch.randn(1,8,64,64, device=device)
    # 生成时间步
    scheduler.set_timesteps(950, device=device)

    for time in scheduler.timesteps:
        #noise = torch.cat((vae_out, vae_out), dim=0)
        noise=vae_out
        #noise = vae_out
        noise = scheduler.scale_model_input(noise, time)
        # 预测噪声分布
        # print('text out', text_out.shape)
        pred_noise = unet(noise, mask_out, time)
        # 降噪
        #pred_noise = pred_noise[0] + 7.5 * (pred_noise[1] - pred_noise[0])
        # 继续添加噪声
        vae_out = scheduler.step(pred_noise, time, vae_out).prev_sample
    
    # 从压缩图恢复成图片
    vae_out = 1/0.18215 * vae_out
    #print(vae_out.shape)
    image = vision_encoder.decoder(vae_out)
    #print(image.shape)
    # 转换并保存
    image = image.cpu()
    image = (image + 1) / 2
    image = image.clamp(0, 1)
    image = image.permute(0, 2, 3, 1)
    image =  np.squeeze(image.numpy()[0])
    image = Image.fromarray(np.uint8(image*255),mode='L')
    #image.save(f'./output/deephomo/{flag}.jpg')
    return image

if __name__ == None:
    image_size = (512, 512)
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])])
    image_path='./deephomo_data/contact/valid_images/1AZT.jpg'
    text_path='./deephomo_data/contact/valid_chains/1AZT.txt'
    image = Image.open(image_path).convert("L")
    text = process._load_text_file(text_path)
    mask=process.pro_mask(text)
    re_image=process.pad_image(image)
    masked = ImageChops.composite(re_image, Image.new('L', re_image.size, 0), mask)
    masked = transform(masked)
    predicted_image = generate(masked, f'1AZT')
    print(predicted_image)
    predicted_image = np.array(predicted_image)
    predicted_image = predicted_image/ 255.0
    print(predicted_image)
    image = np.array(image)
    image = image/ 255.0
    mask = np.array(mask)
    mask = (mask > 0).astype(int)
    n=10
    precision=evaluate_precision(predicted_image, image, mask, n)
    print("precision:", precision)
    print(f'1AZT, finished') 
#for i,masked in enumerate(masked):
    #image = generate(masked, f'gen_img{i}')
    #print(f'text: {masked}, finished')  
if __name__ == '__main__':
    image_size = (512, 512)
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])])
    image_folder = './deephomo_data/contact/valid_images'
    text_folder = './deephomo_data/contact/valid_chains'
    output_folder = './output/deephomo'  # 保存输出结果的文件夹路径
    precision_file='./output/deephomo/precision'
    # 获取文件夹中的所有文件路径
    image_files = os.listdir(image_folder)
    precision_values = []
    for image_file in image_files:
        # 构建图片路径和文本路径
        image_path = os.path.join(image_folder, image_file)
        pdb_id=image_file.split('.')[0]
        text_file = image_file.split('.')[0] + '.txt'
        text_path = os.path.join(text_folder, text_file)

        # 读取图片和文本数据
        image = Image.open(image_path).convert("L")
        text = process._load_text_file(text_path)

        # 进行其他处理操作
        mask = process.pro_mask(text)
        re_image = process.pad_image(image)
        masked = ImageChops.composite(re_image, Image.new('L', re_image.size, 0), mask)
        masked = transform(masked)
        predicted_image = generate(masked, pdb_id)  # 这里使用文本文件的名称作为生成图像的标识

        # 保存预测图像
        predicted_image.save(os.path.join(output_folder, f"{pdb_id}.jpg"))

        # 其他处理操作
        predicted_image = np.array(predicted_image)
        predicted_image = predicted_image / 255.0
        image = np.array(re_image)
        image = image / 255.0
        mask = np.array(mask)
        mask = (mask > 0).astype(int)
        n=20
        precision = evaluate_precision(predicted_image, image, mask, n)
        precision_values.append(precision)
        with open(precision_file, 'a') as f:
            f.write(f'{pdb_id}:{precision}\n')
        print(f'Precision for {pdb_id}:{precision}')
    avg_precision = np.mean(precision_values)  # 计算平均precision
    print(f'Average Precision: {avg_precision}')
    print('Finished processing all files.')      