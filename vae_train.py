import os,torch
from vision_auto_encoder import VAE
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from PIL import Image, ImageDraw, ImageChops
import numpy as np
from torchvision import transforms
from datasets import Dataset
device = 'cuda' if torch.cuda.is_available() else 'cpu'
vae=VAE()
vae.to(device)
vae.train()
device

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
        return 2*len(self.file_names)

    def __getitem__(self, idx):
        if idx < len(self.file_names):
            file_name = self.file_names[idx]
            image_path = os.path.join(self.image_folder, file_name + ".jpg")
            #text_path = os.path.join(self.text_folder, file_name + ".txt")
            image = Image.open(image_path).convert("L")
            #text = self._load_text_file(text_path)
            #mask=self.pro_mask(text)
            re_image=self.pad_image(image)        
            #masked = ImageChops.composite(re_image, Image.new('L', re_image.size, 0), mask)
            if self.transform:
                image = self.transform(re_image)
                #masked = self.transform(masked)
            return {'image': image}
        else:
            file_name = self.file_names[idx-len(self.file_names)]
            image_path = os.path.join(self.image_folder, file_name + ".jpg")
            text_path = os.path.join(self.text_folder, file_name + ".txt")
            image = Image.open(image_path).convert("L")
            text = self._load_text_file(text_path)
            mask=self.pro_mask(text)
            re_image=self.pad_image(image)
            masked = ImageChops.composite(re_image, Image.new('L', re_image.size, 0), mask)
            if self.transform:
                image = self.transform(re_image)
                masked = self.transform(masked)
            return {'image': masked}
        #return {'image': image, 'text': text, 'masked':masked}

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

#dataset = dataset.map(function=f,
                          #batched=True,
                          #batch_size=1000,
                          #num_proc=4,
                          #remove_columns=list(dataset.features)[1:])

    #加载为numpy数据
#data = np.empty((_, 1, 512, 512), dtype=np.float32)
#for i in range(len(dataset)):
    #data[i] = dataset[i]['image']
def collate_fn(data):
    image = [i['image'] for i in data]
    image = torch.stack(image).to(device)
    return {'image': image}
loader = DataLoader(dataset, shuffle=True, collate_fn=collate_fn, batch_size=1)
# 创建数据加载器实例
#batch_size = 48
optimizer = torch.optim.Adam(vae.parameters(), lr=2e-4)
#scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,
                                              #start_factor=1,
                                              #end_factor=0,
                                              #total_iters=1000 * len(loader))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                              lr_lambda=lambda step: 1 - step/(1000 * len(loader))
                                              )
criterion = torch.nn.MSELoss(reduction='none')

def show(image, i):
    #if type(image) == torch.Tensor:
        #image = image.to('cpu').detach().numpy()
    image = image.cpu()
    image = (image + 1) / 2
    image = image.clamp(0, 1)
    image = image.permute(0, 2, 3, 1)
    image =  np.squeeze(image.detach().numpy()[0])
    image = Image.fromarray(np.uint8(image*255),mode='L')
    image.save(f'./output/vae/vae_epoch{i}.jpg')
def save_checkpoint(model, optimizer, epoch, loss, last=False):
    state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss
    }
    torch.save(state, f'./model_params/deephomo_vae/checkpoint_{epoch}.pth.tar')
    if last:
        torch.save(state, './model_params/deephomo_vae/last_checkpoint.pth.tar')
        
def train():
    epochs = 1000
    loss_recorder = []
    for epoch in range(epochs):
        loss_epoch=0
        for _, data in enumerate(loader):
            data = torch.tensor(data['image']).to(device)
            #print(data.shape)
            pred, mu, log_var = vae(data)

            loss_mse = criterion(pred, data) * 10000
            loss_mse = loss_mse.mean(dim=(1, 2, 3))

            loss_kl = 1 + log_var - mu**2 - log_var.exp()
            loss_kl = loss_kl.sum(dim=1) * -0.5

            loss = (loss_mse + loss_kl).mean()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            loss_epoch += loss.item()
        if epoch % 10 == 0:
            #print(epoch, loss_epoch, optimizer.param_groups[0]['lr'])
            print(epoch, loss_epoch)

            #with torch.no_grad():
                #gen = decoder(torch.randn(10, 128, device=device))
            show(pred,epoch)
        epoch_loss=loss_epoch
        save_checkpoint(vae, optimizer, epoch, epoch_loss, True)
        loss_recorder.append((epoch, epoch_loss))
        loss_recorder = sorted(loss_recorder, key=lambda e:e[-1])
        if len(loss_recorder) > 10:
            del_check = loss_recorder.pop()
            os.remove(f'./model_params/deephomo_vae/checkpoint_{del_check[0]}.pth.tar')
        with open("./loss/deephomo_vae/epoch_loss.txt", 'a') as ep_los:
                ep_los.write(str(epoch_loss)+ '\n')       
        print(f'epoch: {epoch:03}  loss: {epoch_loss:.8f}')

        if epoch % 1 == 0:
            print('Top 10 checkpoints:')
            for i in loss_recorder:
                print(i)

print('end training.')


local_training = True

if local_training:
    train()