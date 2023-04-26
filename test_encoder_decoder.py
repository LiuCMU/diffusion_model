"""
Reference
Dalle-E: https://arxiv.org/abs/2102.12092
https://github.com/openai/DALL-E/blob/master/notebooks/usage.ipynb
Other useful pretrained encoder and decoders: https://colab.research.google.com/github/CompVis/taming-transformers/blob/master/scripts/reconstruction_usage.ipynb
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
from torch.utils.data import DataLoader, ConcatDataset
import torch.nn.functional as F
from torchvision import transforms, models
from data import img_dataset
# from modules import NAFNet
from SSIM import ssim
import matplotlib.pyplot as plt
from PIL import Image
from dall_e import map_pixels, unmap_pixels, load_model
from utils import folder


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

parser = argparse.ArgumentParser()
parser.add_argument("--batchsize", type=int, default=1)
args = parser.parse_args()
wandb.init(project="617", entity="img_deblur", config=args)
config = wandb.config


batchsize = config.batchsize


def calc_PSNR(img1, img2):
    """
    calculating PSNR based on 10.3390/s20133724
    img1/img1: tensor of shape (N, C, H, W)
    """
    N, C, H, W = img1.shape
    denominator = (1/(N*C*H*W)) * torch.sum(torch.square(img1 - img2))
    psnr = 10*torch.log10((255*255)/denominator)
    return psnr.item()


#load datasets
train = img_dataset(train_path, debug=False, scale=False) # debug = False
# train_loader = DataLoader(train, batchsize, num_workers=4)
test = img_dataset(test_path, debug=False, scale=False) # debug = False
# test_loader = DataLoader(test, batchsize, num_workers=4)
print("Number of training and testing: %i, %i" % (len(train), len(test)))
all = ConcatDataset([train, test])
print("Number of all images: %i" % (len(all)))
all_loader = DataLoader(all, batchsize, num_workers=4)

enc = load_model(os.path.join(folder, 'params/dalle/encoder.pkl'), device)  #input tensor shape (B, C, H, W) ranging from -1 to 1
dec = load_model(os.path.join(folder, 'params/dalle/decoder.pkl'), device)


blury_psnr, blury_ssim = [], []
sharp_psnr, sharp_ssim = [], []
og_psnr, og_ssim = [], []
rec_psnr, rec_ssim = [], []


for i, (xs, ys) in enumerate(all_loader):
    print(f'epoch {i}')

    with torch.no_grad():
        ## blury image encoder and decoder performance
        z_logits = enc(map_pixels(xs).to(device))
        z = torch.argmax(z_logits, axis=1)
        z = F.one_hot(z, num_classes=enc.vocab_size).permute(0, 3, 1, 2).float()

        stats = dec(z).float()
        rec = unmap_pixels(torch.sigmoid(stats[:, :3]))

    if i % 500 == 0:
        #visualize the sharp image and reconstructed sharp image
        y_image = transforms.ToPILImage()(xs[0])
        y_image.save(f'blury_{i}.png')

        y_image_re = transforms.ToPILImage()(rec[0])
        y_image_re.save(f'blury_re{i}.png')

    with torch.no_grad():
        ## sharp image encoder and decoder performance
        z_logits2 = enc(map_pixels(ys).to(device))
        z2 = torch.argmax(z_logits2, axis=1)
        z2 = F.one_hot(z2, num_classes=enc.vocab_size).permute(0, 3, 1, 2).float()

        stats2 = dec(z2).float()
        rec2 = unmap_pixels(torch.sigmoid(stats2[:, :3]))

    if i % 500 == 0:
        #visualize the sharp image and reconstructed sharp image
        y_image = transforms.ToPILImage()(ys[0])
        y_image.save(f'sharp_{i}.png')

        y_image_re = transforms.ToPILImage()(rec2[0])
        y_image_re.save(f'sharp_re{i}.png')

    xs, rec, ys, rec2 = xs.to(device)*255, rec*255, ys.to(device)*255, rec2*255

    # calculate the PSNR and SSIM value
    psnr = calc_PSNR(rec, xs)
    print(f'PSNR blury reconstruction: {psnr}')
    ssim_i = ssim(rec, xs).item()
    print(f'SSIM blury reconstruction: {ssim_i}')
    blury_psnr.append(psnr)
    blury_ssim.append(ssim_i)
    
    # calculate the PSNR and SSIM value for the sharp mage and reconstructed sharp image
    psnr = calc_PSNR(rec2, ys)
    print(f'PSNR sharp reconstruction: {psnr}')
    ssim_i = ssim(rec2, ys).item()
    print(f'SSIM sharp reconstruction: {ssim_i}')
    sharp_psnr.append(psnr)
    sharp_ssim.append(ssim_i)

    psnr = calc_PSNR(xs, ys)
    print(f'PSNR blury sharp: {psnr}')
    ssim_i = ssim(xs, ys).item()
    print(f'SSIM blury sharp: {ssim_i}')
    og_psnr.append(psnr)
    og_ssim.append(ssim_i)

    psnr = calc_PSNR(rec, rec2)
    print(f'PSNR blury sharp reconstruction: {psnr}')
    ssim_i = ssim(rec, rec2).item()
    print(f'SSIM blury sharp reconstruction: {ssim_i}')
    rec_psnr.append(psnr)
    rec_ssim.append(ssim_i)

    # torch.cuda.empty_cache()  #free up memory


pickle.dump([blury_psnr, blury_ssim,
            sharp_psnr, sharp_ssim,
            og_psnr, og_ssim,
            rec_psnr, rec_ssim], open('encoder_decoder_metrics.pkl', 'wb'))
