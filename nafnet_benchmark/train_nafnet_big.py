import argparse
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
import sys
sys.path.insert(0, '../')
from data import img_dataset
from nafnet import *
from SSIM import ssim
# from torch.utils.tensorboard import SummaryWriter
from modules import ED
from dall_e import map_pixels, unmap_pixels, load_model
import torch.nn.functional as F


train_path = '../../../../net/projects/ranalab/kh310/img_deblur/train'
test_path = '../../../../net/projects/ranalab/kh310/img_deblur/test'

exp_name = 'nafnet_big'

lr = 0.0005
patience = 10
epochs = 100
batchsize = 4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def calc_PSNR(img1, img2):
    """
    calculating PSNR based on 10.3390/s20133724
    img1/img1: tensor of shape (N, C, H, W)
    """
    N, C, H, W = img1.shape
    denominator = (1 / (N * C * H * W)) * torch.sum(torch.square(img1 - img2))
    psnr = 10 * torch.log10((255 * 255) / denominator)
    return psnr.item()


def validate(loader):
    model.eval()
    with torch.no_grad():
        psnrs, ssims, sizes = [], [], []
        for i, batch in enumerate(loader):
            xs_, ys_ = batch
            xs_ = xs_.to(device)
            ys_ = ys_.to(device)

            yhat = model(xs_).detach()
            ys = ys_.detach()
            psnr = calc_PSNR(yhat, ys)
            ssim_i = ssim(yhat, ys).item()

            psnrs.append(psnr)
            ssims.append(ssim_i)
            sizes.append(len(batch))

    score = np.dot(psnrs, sizes) / np.sum(sizes)
    ssim_score = np.dot(ssims, sizes) / np.sum(sizes)
    return (score, ssim_score)


def init_params(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, a=1.0)
        torch.nn.init.zeros_(m.bias)


# load datasets
train = img_dataset(train_path, debug=False, scale=True)
train_loader = DataLoader(train, batchsize, num_workers=4)
test = img_dataset(test_path, debug=False, scale=True)
test_loader = DataLoader(test, batchsize, num_workers=4)
print("Number of training and testing: %i, %i" % (len(train), len(test)))

model = NAFNet(img_channel=3, middle_blk_num=1, enc_blk_nums=[1, 1, 1, 28], dec_blk_nums=[1, 1, 1, 1])
model.to(device)
model.apply(init_params)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=patience, mode="max")
print("Total number of trainable parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

# train_psnr_list, train_ssim_list, test_psnr_list, test_ssim_list = [], [], [], []

# load ckpt
model.load_state_dict(torch.load(f"{exp_name}_latest.pt")['model_state_dict'])
optimizer.load_state_dict(torch.load(f"{exp_name}_latest.pt")['optimizer_state_dict'])
start_epoch = torch.load(f"{exp_name}_latest.pt")['epoch']
train_psnr_list = list(np.load(f'{exp_name}_train_psnr.npy'))
test_psnr_list = list(np.load(f'{exp_name}_test_psnr.npy'))
train_ssim_list = list(np.load(f'{exp_name}_train_ssim.npy'))
test_ssim_list = list(np.load(f'{exp_name}_test_ssim.npy'))


for i in range(start_epoch, epochs):
    epoch = i
    learning_rate = optimizer.param_groups[0]['lr']

    model.train()

    for j, (xs, ys) in enumerate(train_loader):
        xs = xs.to(device)
        ys = ys.to(device)
        yhat = model(xs)
        N, C, H, W = ys.shape
        denominator = (1 / (N * C * H * W)) * torch.sum(torch.square(ys - yhat))
        loss = -10 * torch.log10((255 * 255) / denominator)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch} Batch {j}/{len(train_loader)} Loss: {loss}")

    train_psnr, train_ssim = validate(train_loader)
    test_psnr, test_ssim = validate(test_loader)

    train_psnr_list.append(train_psnr)
    train_ssim_list.append(train_ssim)
    test_psnr_list.append(test_psnr)
    test_ssim_list.append(test_ssim)

    print("saving latest checkpoint")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, f"{exp_name}_latest.pt")

    with open(f'{exp_name}_train_psnr.npy', 'wb') as f:
        np.save(f, np.array(train_psnr_list))
    f.close()
    with open(f'{exp_name}_test_psnr.npy', 'wb') as f:
        np.save(f, np.array(test_psnr_list))
    f.close()
    with open(f'{exp_name}_train_ssim.npy', 'wb') as f:
        np.save(f, np.array(train_ssim_list))
    f.close()
    with open(f'{exp_name}_test_ssim.npy', 'wb') as f:
        np.save(f, np.array(test_ssim_list))
    f.close()

    scheduler.step(test_psnr)

torch.save(model.state_dict(), f"{exp_name}_model.pt")

