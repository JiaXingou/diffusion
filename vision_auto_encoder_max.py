import torch
from torch import nn
from diffusers import AutoencoderKL

class CrossAttention(nn.Module):
    def __init__(self, dim_q, dim_kv, heads=8):
        super().__init__()
        #dim_q -> 320
        #dim_kv -> 768
        self.dim_q = dim_q
        self.heads = heads
        self.wq = nn.Linear(dim_q, dim_q, bias=False)
        self.wk = nn.Linear(dim_kv, dim_q, bias=False)
        self.wv = nn.Linear(dim_kv, dim_q, bias=False)
        self.out_proj = nn.Linear(dim_q, dim_q)
    def multihead_reshape(self, x):
        #x -> [1, 4096, 320]
        b, lens, dim = x.shape
        #[1, 4096, 320] -> [1, 4096, 8, 40]
        x = x.reshape(b, lens, self.heads, dim // self.heads)
        #[1, 4096, 8, 40] -> [1, 8, 4096, 40]
        x = x.transpose(1, 2)
        #[1, 8, 4096, 40] -> [8, 4096, 40]
        x = x.reshape(b * self.heads, lens, dim // self.heads)
        return x
    def multihead_reshape_inverse(self, x):
        #x -> [8, 4096, 40]
        b, lens, dim = x.shape
        #[8, 4096, 40] -> [1, 8, 4096, 40]
        x = x.reshape(b // self.heads, self.heads, lens, dim)
        #[1, 8, 4096, 40] -> [1, 4096, 8, 40]
        x = x.transpose(1, 2)
        #[1, 4096, 320]
        x = x.reshape(b // self.heads, lens, dim * self.heads)
        return x
    def forward(self, q, kv):
        #x -> [1, 4096, 320]
        #kv -> [1, 77, 768]
        #[1, 4096, 320] -> [1, 4096, 320]
        q = self.wq(q)
        #[1, 77, 768] -> [1, 77, 320]
        k = self.wk(kv)
        #[1, 77, 768] -> [1, 77, 320]
        v = self.wv(kv)

        #[1, 4096, 320] -> [8, 4096, 40]
        q = self.multihead_reshape(q)
        #[1, 77, 320] -> [8, 77, 40]
        k = self.multihead_reshape(k)
        #[1, 77, 320] -> [8, 77, 40]
        v = self.multihead_reshape(v)
        #[8, 4096, 40] * [8, 40, 77] -> [8, 4096, 77]
        atten = q.bmm(k.transpose(1, 2)) * (self.dim_q // self.heads)**-0.5

        #从数学上是等价的,但是在实际计算时会产生很小的误差
        # atten = torch.baddbmm(
        #     torch.empty(q.shape[0], q.shape[1], k.shape[1], device=q.device),
        #     q,
        #     k.transpose(1, 2),
        #     beta=0,
        #     alpha=(self.dim_q // 8)**-0.5,
        # )
        atten = atten.softmax(dim=-1)
        #[8, 4096, 77] * [8, 77, 40] -> [8, 4096, 40]
        atten = atten.bmm(v)
        #[8, 4096, 40] -> [1, 4096, 320]
        atten = self.multihead_reshape_inverse(atten)
        #[1, 4096, 320] -> [1, 4096, 320]
        atten = self.out_proj(atten)
        return atten

class ResNetBlock(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.seq = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=dim_in, eps=1e-6, affine=True),
            nn.SiLU(),
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=dim_out, eps=1e-6, affine=True),
            nn.SiLU(),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1))
        
        self.resil = nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1) if dim_in != dim_out else nn.Identity()
        self.dim_in = dim_in
        self.dim_out = dim_out
    def forward(self, x):
        res = self.resil(x)
        out = self.seq(x) + res
        return out

class SelfAttention(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.norm = nn.GroupNorm(num_channels=512, num_groups=32, eps=1e-6, affine=True)
        self.wq = torch.nn.Linear(embed_dim, embed_dim)
        self.wk = torch.nn.Linear(embed_dim, embed_dim)
        self.wv = torch.nn.Linear(embed_dim, embed_dim)
        self.out_proj = torch.nn.Linear(embed_dim, embed_dim)
    def forward(self, x):
        # x: (b, 512, 64, 64)
        res = x
        b,c,h,w = x.shape
        x = self.norm(x)
        x = x.flatten(start_dim=2).transpose(1,2) # (1, 4096, 512)
        q = self.wq(x) # (1, 4096, 512)
        k = self.wk(x)
        v = self.wv(x)
        k = k.transpose(1,2) # (1, 512, 4096
        #[1, 4096, 512] * [1, 512, 4096] -> [1, 4096, 4096]
        #0.044194173824159216 = 1 / 512**0.5
        atten = q.bmm(k) / 512**0.5

        atten = torch.softmax(atten, dim=2)
        atten = atten.bmm(v)
        atten = self.out_proj(atten)
        atten = atten.transpose(1, 2).reshape(b, c, h, w)
        atten = atten + res
        return atten

class Pad(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return nn.functional.pad(x, (0, 1, 0, 1),
                                    mode='constant',
                                    value=0)

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            #in
            nn.Conv2d(1, 128, kernel_size=3, stride=1, padding=1),
            #down
            nn.Sequential(
                ResNetBlock(128, 128),
                ResNetBlock(128, 128),
                nn.Sequential(
                    Pad(),
                    torch.nn.Conv2d(128, 128, 3, stride=2, padding=0),
                ),
            ),
            torch.nn.Sequential(
                ResNetBlock(128, 256),
                ResNetBlock(256, 256),
                torch.nn.Sequential(
                    Pad(),
                    nn.Conv2d(256, 256, 3, stride=2, padding=0),
                ),
            ),
            nn.Sequential(
                ResNetBlock(256, 512),
                ResNetBlock(512, 512),
                nn.Sequential(
                    Pad(),
                    nn.Conv2d(512, 512, 3, stride=2, padding=0),
                ),
            ),
            nn.Sequential(
                ResNetBlock(512, 512),
                ResNetBlock(512, 512),
            ),
            #mid
            nn.Sequential(
                ResNetBlock(512, 512),
                SelfAttention(),
                ResNetBlock(512, 512),
            ),
            #out
            nn.Sequential(
                nn.GroupNorm(num_channels=512, num_groups=32, eps=1e-6),
                nn.SiLU(),
                nn.Conv2d(512, 1, 3, padding=1),
            ),
            #正态分布层
            nn.Conv2d(1, 1, 1))
        
        self.encoder2 = nn.Sequential(
            #in
            nn.Conv2d(1, 128, kernel_size=3, stride=1, padding=1),
            #down
            nn.Sequential(
                ResNetBlock(128, 128),
                ResNetBlock(128, 128),
                nn.Sequential(
                    Pad(),
                    torch.nn.Conv2d(128, 128, 3, stride=2, padding=0),
                ),
            ),
            torch.nn.Sequential(
                ResNetBlock(128, 256),
                ResNetBlock(256, 256),
                torch.nn.Sequential(
                    Pad(),
                    nn.Conv2d(256, 256, 3, stride=2, padding=0),
                ),
            ),
            nn.Sequential(
                ResNetBlock(256, 512),
                ResNetBlock(512, 512),
                nn.Sequential(
                    Pad(),
                    nn.Conv2d(512, 512, 3, stride=2, padding=0),
                ),
            ),
            nn.Sequential(
                ResNetBlock(512, 512),
                ResNetBlock(512, 512),
            ),
            #mid
            nn.Sequential(
                ResNetBlock(512, 512),
                SelfAttention(),
                ResNetBlock(512, 512),
            ),
            #out
            nn.Sequential(
                nn.GroupNorm(num_channels=512, num_groups=32, eps=1e-6),
                nn.SiLU(),
                nn.Conv2d(512, 16, 3, padding=1),
            ),
            #正态分布层
            nn.Conv2d(16, 16, 1))
        
        self.atten = CrossAttention(16,1)

        self.decoder = nn.Sequential(
            #正态分布层
            nn.Conv2d(16, 16, 1),
            #in
            nn.Conv2d(16, 512, kernel_size=3, stride=1, padding=1),
            #middle
            nn.Sequential(ResNetBlock(512, 512), 
                                SelfAttention(), 
                                ResNetBlock(512, 512)),
            #up
            nn.Sequential(
                ResNetBlock(512, 512),
                ResNetBlock(512, 512),
                ResNetBlock(512, 512),
                nn.Upsample(scale_factor=2.0, mode='nearest'),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
            ),
            nn.Sequential(
                ResNetBlock(512, 512),
                ResNetBlock(512, 512),
                ResNetBlock(512, 512),
                nn.Upsample(scale_factor=2.0, mode='nearest'),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
            ),
            nn.Sequential(
                ResNetBlock(512, 256),
                ResNetBlock(256, 256),
                ResNetBlock(256, 256),
                nn.Upsample(scale_factor=2.0, mode='nearest'),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
            ),
            nn.Sequential(
                ResNetBlock(256, 128),
                ResNetBlock(128, 128),
                ResNetBlock(128, 128),
            ),
            #out
            nn.Sequential(
                nn.GroupNorm(num_channels=128, num_groups=32, eps=1e-6),
                nn.SiLU(),
                nn.Conv2d(128, 1, 3, padding=1),
            ))
        


    #def sample(self, h):
        #h -> [1, 8, 64, 64]
        #[1, 4, 64, 64]
        #mean = h[:, :4]
        #logvar = h[:, 4:]
        #std = logvar.exp()**0.5
        #[1, 4, 64, 64]
        #h = torch.randn(mean.shape, device=mean.device)
        #h = mean + std * h
        #return h
    def forward(self, x,y):
        #x -> [1, 3, 512, 512]
        #[1, 3, 512, 512] -> [1, 8, 64, 64]
        hidden = self.encoder(x)
        hidden2 = self.encoder2(y)
        h=self.atten(hidden, hidden2)
        #[1, 8, 64, 64] -> [1, 4, 64, 64]
        #h = self.sample(h)
        #[1, 4, 64, 64] -> [1, 3, 512, 512]
        #h = self.decoder(h)
        return self.decoder(h)
