"""
Reference
Rombach paper: https://arxiv.org/abs/2112.10752
DDPM paper: https://arxiv.org/pdf/2006.11239.pdf
Dalle paper: https://arxiv.org/abs/2102.12092
Diffusion tutorial: https://colab.research.google.com/drive/1sjy9odlSSy0RBVgMTgP7s99NXsqglsUL?usp=sharing
"""

import wandb
import os
import shutil
import tarfile
import socket
hostname = socket.gethostname()
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms, models
from data import img_dataset
from modules import Unet, ED
from SSIM import ssim
import matplotlib.pyplot as plt
from PIL import Image
from dall_e import map_pixels, unmap_pixels, load_model
from utils import folder
from tqdm import tqdm


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
# device = torch.device("cpu")

#parameters
if "pro" in hostname.lower():  #my mac
    train_path = "/Users/liu5/Documents/10-617HW/project/data/train"
    test_path = "/Users/liu5/Documents/10-617HW/project/data/test"
elif "exp" in hostname.lower():  #expanse 
    #copy, extract tar file to the local computing node scratch
    tar_path = "/expanse/lustre/projects/cwr109/zhen1997/img_deblur.tar"
    local_scratch = f"/scratch/{os.environ['USER']}/job_{os.environ['SLURM_JOB_ID']}"
    print("computing node local scratch path: %s" % local_scratch)
    train_path = os.path.join(local_scratch, "img_deblur/train")
    test_path = os.path.join(local_scratch, "img_deblur/test")
    if os.path.exists(train_path) == False:
        shutil.copy(tar_path, local_scratch)
        tar = tarfile.open(os.path.join(local_scratch, "img_deblur.tar"))
        tar.extractall(local_scratch)
        tar.close()
    print("Finished extracting the dataset")
elif ("braavos" in hostname.lower()) or ( "storm" in hostname.lower()):  #braavos/stromland
    train_path = os.path.join(folder, "img_deblur/train")
    test_path = os.path.join(folder, "img_deblur/test")


transform = transforms.Compose([
    transforms.Lambda(lambda t: 2 * t - 1)  #change element to (-1, 1)
])

def show_tensor_image(image: torch.Tensor):
    """
    image shape (C, H, W)"""
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: 255 * ((t+1)/2)),  #change element to (0, 255)
        transforms.Lambda(lambda t: t.permute(1, 2, 0)),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage()
    ])
    plt.imshow(reverse_transforms(image))


def calc_PSNR(img1, img2):
    """
    calculating PSNR based on 10.3390/s20133724
    img1/img1: tensor of shape (N, C, H, W)
    """
    N, C, H, W = img1.shape
    denominator = (1/(N*C*H*W)) * torch.sum(torch.square(img1 - img2))
    psnr = 10*torch.log10((255*255)/denominator)
    return psnr.item()


class diffusion(nn.Module):
    def __init__(self, steps=300, start=0.0001, end=0.02):
        super(diffusion, self).__init__()
        """
        steps: the number of steps from the sharp image to random noise"""
        self.betas = torch.linspace(start, end, steps)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.Tensor([1]), self.alphas_cumprod[:-1]])
        self.sqrt_recip_alphas = torch.sqrt(1/self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumpord = torch.sqrt(1 - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)

        self.model = Unet(input_channels=1, image_channels=1, down_channels=(64, 128, 256, 512, 1024),
                          up_channels = (1024, 512, 256, 128, 64), time_emb_dim = 32)

    
    def get_step_vals(self, vals: torch.Tensor, t: torch.Tensor, x_shape: torch.Size):
        """
        vals: a list of betas/alphas
        t: int or a list of time stpes (the index for the vals)
        x_shape: (B, C, H, W)
        
        return elements of vals that are coorespond to t"""
        batch_size = x_shape[0]
        out = vals.gather(-1, t.cpu()).reshape(-1, 1)
        # out_reshaped = out.reshape(batch_size, 1, 1, 1)
        out_reshaped = torch.unsqueeze(torch.unsqueeze(out, -1), -1)
        return out_reshaped


    def forward_diffusion(self, x_tilde: torch.Tensor, x0: torch.Tensor, t: torch.Tensor, device='cpu'):
        """
        x0: a single sharp image of a batch of sharp images
        x_tilde: the blury image with the same shape as x0
        Takes an image and a timestep as input and returns the noisy version of it at step t
        """
        noise = x0 - x_tilde
        sqrt_alphas_cumprod_t = self.get_step_vals(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_step_vals(
            self.sqrt_one_minus_alphas_cumpord, t, x0.shape
        )
        x_t = sqrt_alphas_cumprod_t.to(device) * x0.to(device) + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device)
        return (x_t, noise)

    
    def backward_diffusion(self, x_noisy: torch.Tensor, t: torch.Tensor, device='cpu'):
        """
        x_noisy the noisy image at step t,
        return the nose of the image when compared with its previous time step
        """
        noise = self.model(x_noisy, t)
        return noise


class process_latent(nn.Module):
    def __init__(self, voc_size=1):
        """preprocess the image tensor the postprocess the image
        preprocess: (B,H, W) ==> (B, 1, H, W) ==> pad H ==> scale to (-1, 1)
        starts from the encoded image from Dall-E

        postprocess is the reverse 
        """
        assert(voc_size > 0)
        self.voc_size = voc_size

    def preprocess(self, xs: torch.Tensor):
        """shape (B, H, W)"""
        xs = torch.unsqueeze(xs, dim=1)
        pad_size = (0, 0, 3, 3)
        xs = F.pad(xs, pad_size)  #(B, 1, 90, 160) ==> (B, 1, 96, 160)
        xs = (xs/self.voc_size) * 2 - 1  #rescale xs and ys to be in the range of (-1, 1)
        return xs

    def postprocess(self, xs: torch.Tensor):
        """shape (B, 1, H, W)"""
        xs = ((xs + 1)/2) * self.voc_size
        xs = xs[:, :, 3:, :]
        xs = xs[:, :, :-3, :]
        xs = torch.squeeze(xs, dim=1)
        return xs
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--diffusion_steps", type=int, default=300)
    args = parser.parse_args()
    wandb.init(project="diffusion", entity="liu97", config=args)
    config = wandb.config

    #load datasets
    train = img_dataset(train_path, debug=False, scale=True) # debug = False
    train_loader = DataLoader(train, config.batch_size, num_workers=4)
    test = img_dataset(test_path, debug=False, scale=True) # debug = False
    test_loader = DataLoader(test, config.batch_size, num_workers=4)
    print("Number of training and testing: %i, %i" % (len(train), len(test)))

    encoder_decoder = ED(folder, device)
    processor = process_latent(encoder_decoder.vocab_size)
    diffuser = diffusion(config.diffusion_steps).to(device)

    optimizer = torch.optim.Adam(diffuser.model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=config.patience, mode="min")

    for i in tqdm(range(config.epochs)):
        print(f'epoch {i}')
        epoch_losses = []
        for xs, ys in train_loader:
            xs = xs.to(device)
            ys = ys.to(device)  #shape (B, 3, H, W)

            #move to latent space
            ys = encoder_decoder.encode(ys)
            ys = processor.preprocess(ys)
            xs = encoder_decoder.encode(xs)
            xs = processor.preprocess(xs) 
            
            
            optimizer.zero_grad()
            t = torch.randint(0, config.diffusion_steps, (ys.shape[0],), device=device).long()
            y_noisy, noise = diffuser.forward_diffusion(xs, ys, t, device)
            noise_pred = diffuser.backward_diffusion(y_noisy, t, device)
            loss = F.l1_loss(noise_pred, noise)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        if i%20 == 0:
            torch.save(diffuser.model.state_dict(), os.path.join(folder, f'params/diffusion/con_latent_diffusion{i}.pt'))
        
        epoch_loss = round(np.mean(epoch_losses), 3)
        lr = optimizer.param_groups[0]['lr']
        wandb.log({
            'loss': epoch_loss,
            'lr': lr
        })
        print(f'Epoch {i+1} Loss {epoch_loss}')
    torch.save(diffuser.model.state_dict(), os.path.join(folder, f'params/diffusion/con_latent_final.pt'))
