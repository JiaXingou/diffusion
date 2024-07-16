import os,torch
from diffusers import DiffusionPipeline
from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from text_encoder import TextEncoder
from vision_auto_encoder import VAE
from unet import UNet
import cv2
from torch.cuda.amp import autocast, GradScaler
from PIL import Image, ImageDraw, ImageChops
import io
import warnings
from torch.utils.data import Dataset, DataLoader
import numpy as np
warnings.filterwarnings('ignore')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# diffusion model 噪声生成器
pipeline = DiffusionPipeline.from_pretrained('./pretrained-params/', safety_checker=None,local_files_only=True,tokenizer=None)
scheduler = pipeline.scheduler
#tokenizer = pipeline.tokenizer
del pipeline

print('Device :', device)
print('Scheduler settings: ', scheduler)
#print('Tokenizer settings: ', tokenizer)

# 数据处理与加载
#dataset = load_dataset('parquet',data_files={'train':'./data/train.parquet'}, split='train')
class ImageDataset(Dataset):
    def __init__(self, image_folder, text_folder, transform=None):
        self.image_folder = image_folder
        self.text_folder = text_folder
        self.transform = transform
        self.file_names = self._get_file_names()

    def _get_file_names(self):
        image_files = os.listdir(self.image_folder)
        text_files = os.listdir(self.text_folder)
        file_names = sorted(list(set([os.path.splitext(f)[0] for f in image_files]) & set([os.path.splitext(f)[0] for f in text_files])))
        return file_names

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        image_path = os.path.join(self.image_folder, file_name + ".jpg")
        text_path = os.path.join(self.text_folder, file_name + ".txt")

        image = Image.open(image_path).convert("L")
        text = self._load_text_file(text_path)
        mask=self.pro_mask(text)
        re_image=self.pad_image(image)        
        #mask = Image.open("./mask/mask2.jpg").convert("L")
        #mask = cv2.imread(".mask/mask2.jpg", cv2.IMREAD_GRAYSCALE)
        #re_image = np.array(re_image)
        #mask = np.array(mask)  
        #masked = cv2.bitwise_and(re_image, re_image, mask=mask)
        
        masked = ImageChops.composite(re_image, Image.new('L', re_image.size, 0), mask)
        #re_image.save("./1/re_image.jpg")
        #masked.save("./1/masked.jpg")
        #cv2.imwrite("./1/re_image.jpg", re_image)
        #cv2.imwrite("./1/masked.jpg", masked)
        if self.transform:
            image = self.transform(re_image)
            masked = self.transform(masked)
        return {'image': image, 'text': text, 'masked':masked}

    def _load_text_file(self, text_path):
        with open(text_path, "r") as file:
            text = file.read()
        text = text.strip()  # 去除首尾的空格和换行符
        text = text.split()  # 按空格分割为多个字符串
        text = [int(num) for num in text]  # 将字符串转换为整数
        return text
    def pad_image(self, image):
        #width, height = image.size
        #max_size = max(width, height)

        padded_image = Image.new("L", (512, 512), color=0)
        padded_image.paste(image, (0, 0))

        #resized_image = padded_image.resize((512, 512))
        return padded_image
    def pro_mask(self, text):
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
# 定义数据增强模块
image_size = (512, 512)
data_transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

image_dir='./deephomo_data/contact/train_images/'
text_dir='./deephomo_data/contact/train_chains/'
dataset = ImageDataset(image_dir, text_dir, transform=data_transform)


def pair_mask_image_process(data):
    # 对图像数据进行增强处理
    pixel = [compose(Image.open(io.BytesIO(i['bytes']))) for i in data['image']]
    # 文本
    text = tokenizer.batch_encode_plus(data['mask'], padding='max_length', truncation=True, max_length=77).input_ids
    return {'pixel_values': pixel, 'input_ids': mask}

#dataset = dataset.map(pair_text_image_process, batched=True, num_proc=1, remove_columns=['image', 'text'])
#dataset = dataset.map(pair_mask_image_process, batched=True, num_proc=1, remove_columns=['image', 'mask_image'])
#dataset.set_format(type='torch')

def collate_fn(data):
    image = [i['image'] for i in data]
    masked = [i['masked'] for i in data]
    image = torch.stack(image).to(device)
    masked = torch.stack(masked).to(device)
    return {'image': image, 'masked': masked}

#loader = DataLoader(dataset, shuffle=True, collate_fn=collate_fn, batch_size=1)
# 创建数据加载器实例
#batch_size = 48
batch_size = 1
loader = DataLoader(dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=True)
# 模型加载
#text_encoder = TextEncoder()
vision_encoder = VAE()
unet = UNet()

#text_encoder.eval()
vision_encoder.load_state_dict(torch.load('./model_params/deephomo_vae/last_checkpoint.pth.tar')['state_dict'])
vision_encoder.eval()
unet.train()

#text_encoder.to(device)
vision_encoder.to(device)
unet.to(device)
 
# 优化器, 损失函数, 混合精度
optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-5, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-8)
criterion = torch.nn.MSELoss().to(device)
scaler = GradScaler()
# 一个epoch的训练
def train_one_epoch(unet, vision_encoder, train_loader, optimizer, criterion, noise_scheduler, scaler):
    loss_epoch = 0.
    for step, pair in enumerate(train_loader):
        #img = image
        img = pair['image']
        #img = img.to('cuda')
        #mask = cv2.imread(".mask/mask2.jpg", cv2.IMREAD_GRAYSCALE)
        # 将掩码应用于原始图像
        #masked = cv2.bitwise_and(img, img, mask=mask)
        masked = pair['masked']
        #masked = masked.to('cuda')
        with torch.no_grad():
            # 文本编码
            #mask_out = mask_encoder(mask)
            mask_out = vision_encoder.encoder(masked)
            mask_out = vision_encoder.sample( mask_out)
            mask_out = mask_out * 0.18215
            # 图像特征
            vision_out = vision_encoder.encoder(img)
            vision_out = vision_encoder.sample(vision_out)
            vision_out = vision_out * 0.18215

        # 添加噪声
        noise = torch.randn_like(vision_out)
        noise_step = torch.randint(0, 1000, (1,)).long().to(device)
        vision_out_noise = noise_scheduler.add_noise(vision_out, noise, noise_step)

        with autocast():
            noise_pred = unet(vision_out_noise, mask_out, noise_step)
            #vae_pred = scheduler.step(noise_pred, noise_step, mask_out).prev_sample
            # 从压缩图恢复成图片
            #vae_pred = 1/0.18215 * vae_pred
            #pred = vision_encoder.decoder(vae_pred)
            #loss = criterion(pred,img,masked)
            loss = criterion(noise_pred, noise)
        
        loss_epoch += loss.item()
        # loss.backward()
        scaler.scale(loss).backward()
        # optimizer.step()
        scaler.step(optimizer)
        scaler.update()
        #对模型的梯度进行裁剪，将模型参数的梯度限制在1.0范围内。
        torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
        #清空梯度
        optimizer.zero_grad()
        #with open("./loss/deephomo/step_loss.txt", 'a') as train_los:
            #train_los.write(str(loss.item())+ '\n')
        print(f'step: {step}  loss: {loss.item():.8f}')
    
    return loss_epoch
# 检查点保存
def save_checkpoint(model, optimizer, epoch, loss, last=False):
    state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss
    }
    torch.save(state, f'./model_params/deephomo/checkpoint_{epoch}.pth.tar')
    if last:
        torch.save(state, './model_params/deephomo/last_checkpoint.pth.tar')
        

epochs = 200
loss_recorder = []
print('start training ...')
for epoch in range(epochs):
    epoch_loss = train_one_epoch(unet, vision_encoder, loader, optimizer, criterion, scheduler, scaler)   
    save_checkpoint(unet, optimizer, epoch, epoch_loss, True)
    loss_recorder.append((epoch, epoch_loss))
    loss_recorder = sorted(loss_recorder, key=lambda e:e[-1])
    if len(loss_recorder) > 10:
        del_check = loss_recorder.pop()
        os.remove(f'./model_params/deephomo/checkpoint_{del_check[0]}.pth.tar')
    with open("./loss/deephomo/epoch_loss.txt", 'a') as ep_los:
            ep_los.write(str(epoch_loss)+ '\n')       
    print(f'epoch: {epoch:03}  loss: {epoch_loss:.8f}')

    if epoch % 1 == 0:
        print('Top 10 checkpoints:')
        for i in loss_recorder:
            print(i)

print('end training.')