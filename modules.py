import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import math
from dall_e import map_pixels, unmap_pixels, load_model
from utils import folder

#Unet
"""reference: https://colab.research.google.com/drive/1sjy9odlSSy0RBVgMTgP7s99NXsqglsUL?usp=sharing"""
class Block(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, time_emb_dim=32, up=False):
        super(Block, self).__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2*in_ch, out_ch, 3, padding=1)  #image size does not change
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)  ##image size n by n ==> 2n
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)  #image size does not change
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)  ##image size n by n ==> n/2
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.BatchNorm2d(out_ch)
        self.norm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """x shape (B, C, H, W), element range (-1, 1)
        t shape (B, 1)"""
        h = self.norm1(self.relu(self.conv1(x)))

        time_emb = self.relu(self.time_mlp(t))
        time_emb = torch.unsqueeze(torch.unsqueeze(time_emb, -1), -1)

        h = h + time_emb

        h = self.norm2(self.relu(self.conv2(h)))
        h = self.transform(h)
        return h
    

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time=0):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        # TODO: Double check the ordering here
        return embeddings


class Unet(nn.Module):
    def __init__(self, input_channels=1, image_channels=3, down_channels=(64, 128, 256, 512, 1024),
                 up_channels = (1024, 512, 256, 128, 64), time_emb_dim = 32):
        super(Unet, self).__init__()

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        # Initial projection
        self.conv0 = nn.Conv2d(input_channels, down_channels[0], 3, padding=1)

        # Downsample
        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i+1], time_emb_dim) \
                                    for i in range(len(down_channels)-1)])
        # Upsample
        self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i+1], time_emb_dim, up=True) \
                                  for i in range(len(up_channels)-1)])
        
        self.output = nn.Conv2d(up_channels[-1], image_channels, 1)  #does not change image size
        # The 1x1 convolution acts as a channel-wise linear combination
        # allowing the model to learn the optimal weights for combining the features extracted by the U-Net to produce the denoised output.


    def forward(self, x, timestep):
        # Embedd time
        t = self.time_mlp(timestep)
        # Initial conv
        x = self.conv0(x)
        # Unet
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
        for up in self.ups:
            residual_x = residual_inputs.pop()
            # Add residual x as additional channels
            x = torch.cat((x, residual_x), dim=1)           
            x = up(x, t)
        return self.output(x)
    

class ED(nn.Module):
    def __init__(self, folder, device):
        super(ED, self).__init__()
        """
        folder: contains the data and params
        """
        self.enc = load_model(os.path.join(folder, 'params/dalle/encoder.pkl'), device)
        self.dec = load_model(os.path.join(folder, 'params/dalle/decoder.pkl'), device)
        self.device = device
    
    @torch.no_grad()
    def encode(self, xs: torch.Tensor):
        """encoder an image tensor, shape(B, C, H, W)"""
        z_logits = self.enc(map_pixels(xs).to(self.device))
        z = torch.argmax(z_logits, axis=1)
        return z  #shape (B, H/8, W/8)
    
    @torch.no_grad()
    def decode(self, zs: torch.Tensor):
        """encoder an image tensor, shape(B, C, H, W)"""
        z = torch.clamp(zs.round().int(), 0, self.enc.vocab_size)
        z = F.one_hot(z, num_classes=self.enc.vocab_size).permute().permute(0, 3, 1, 2).float()

        stats = self.dec(z).float()
        rec = unmap_pixels(torch.sigmoid(stats[:, :3]))
        return rec
