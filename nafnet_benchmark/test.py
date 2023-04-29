import argparse
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
import sys
sys.path.insert(0, '../')
from data import img_dataset
from SSIM import ssim
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image
from nafnet import *


# parameters
train_path = '../../../../net/projects/ranalab/kh310/img_deblur/train'
test_path = '../../../../net/projects/ranalab/kh310/img_deblur/test'


if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    print("CUDA not available!")
    device = torch.device("cpu")


exp_name = 'nafnet_big'
# model = NAFNet(img_channel=3, middle_blk_num=1, enc_blk_nums=[1, 1], dec_blk_nums=[1, 1])
model = NAFNet(img_channel=3, middle_blk_num=1, enc_blk_nums=[1, 1, 1, 28], dec_blk_nums=[1, 1, 1, 1])
model.load_state_dict(torch.load(f"{exp_name}_latest.pt")['model_state_dict'])
model.to(device)


blur = '../../../../net/projects/ranalab/kh310/img_deblur/test/blur_gamma/GOPR0862_11_00_000013.png'
sharp = '../../../../net/projects/ranalab/kh310/img_deblur/test/sharp/GOPR0862_11_00_000013.png'
convertor = transforms.ToTensor()
img_blur = Image.open(blur)
x = convertor(img_blur)
yhat = model((255 * torch.unsqueeze(x, dim=0)).to(device)).detach().cpu().numpy()
yhat2 = np.squeeze(yhat).transpose(1, 2, 0)
print("yhat type:", type(yhat2))
print("yhat shape:", yhat2.shape)
img = Image.fromarray(yhat2.astype(np.uint8), mode="RGB")
img.save(f'{exp_name}_GOPR0862_11_00_000013.png')

#
# for i, (xs, ys) in enumerate(test_loader):
#     xs = xs.to(device)
#     ys = ys.to(device)
#     yhat = model(xs)
#     N, C, H, W = ys.shape
#
#     for j in range(batchsize):
#         blur = xs[j]
#         sharp = ys[j]
#         pred = yhat[j]
#         save_image(blur, f'./demo/blur{j}.png')
#         save_image(sharp, f'./demo/sharp{j}.png')
#         save_image(pred, f'./demo/pred{j}.png')
#
#     break



