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

exp_name = 'nafnet_latent'

lr = 0.0005
patience = 10
epochs = 100
batchsize = 1

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

            # move to latent space
            xs_latent = torch.unsqueeze(encoder_decoder.encode(xs_), dim=1).float()  # shape(B, 1, H/8, W/8)
            # xs_latent = encoder_decoder.encode(xs_).float()  # shape(B, 1, H/8, W/8)
            yhat_latent = torch.squeeze(model(xs_latent), dim=1)  # B, H/8, W/8
            yhat = encoder_decoder.decode(yhat_latent).float()  # B, 3, H, W

            psnr = calc_PSNR(255 * yhat, 255 * ys_)
            ssim_i = ssim(255 * yhat, 255 * ys_).item()

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
train = img_dataset(train_path, debug=False, scale=False)
train_loader = DataLoader(train, batchsize, num_workers=4)
test = img_dataset(test_path, debug=False, scale=False)
test_loader = DataLoader(test, batchsize, num_workers=4)
print("Number of training and testing: %i, %i" % (len(train), len(test)))


encoder_decoder = ED('', device)
model = NAFNet(middle_blk_num=1, enc_blk_nums=[1, 1], dec_blk_nums=[1, 1])
model.to(device)
model.apply(init_params)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=patience, mode="max")
print("Total number of trainable parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

train_psnr_list, train_ssim_list, test_psnr_list, test_ssim_list, train_loss = [], [], [], [], []

# # load ckpt
# model.load_state_dict(torch.load(f"{exp_name}_latest.pt")['model_state_dict'])
# optimizer.load_state_dict(torch.load(f"{exp_name}_latest.pt")['optimizer_state_dict'])

MSELoss = torch.nn.MSELoss()
L1loss = torch.nn.L1Loss()
for i in range(epochs):
    epoch = i
    learning_rate = optimizer.param_groups[0]['lr']

    model.train()
    epoch_loss = 0
    for j, (xs, ys) in enumerate(train_loader): # scale=False
        xs = xs.to(device)
        ys = ys.to(device)

        # move to latent space
        xs_latent = torch.unsqueeze(encoder_decoder.encode(xs), dim=1).float()  # shape(B, 1, H/8, W/8)
        ys_latent = torch.unsqueeze(encoder_decoder.encode(ys), dim=1).float()

        yhat_latent = model(xs_latent) # B, 1, H/8, W/8
        loss = L1loss(yhat_latent, ys_latent)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch} Batch {j}/{len(train_loader)} Loss: {loss}")
        epoch_loss += loss.detach().item()

    train_psnr, train_ssim = validate(train_loader)
    test_psnr, test_ssim = validate(test_loader)

    train_psnr_list.append(train_psnr)
    train_ssim_list.append(train_ssim)
    test_psnr_list.append(test_psnr)
    test_ssim_list.append(test_ssim)
    train_loss.append(epoch_loss / len(train_loader))

    # print("saving latest checkpoint")
    # torch.save({
    #     'epoch': epoch,
    #     'model_state_dict': model.state_dict(),
    #     'optimizer_state_dict': optimizer.state_dict()
    # }, f"{exp_name}_latest.pt")
    #
    # with open('train_psnr.npy', 'wb') as f:
    #     np.save(f, np.array(train_psnr_list))
    # f.close()
    # with open('test_psnr.npy', 'wb') as f:
    #     np.save(f, np.array(test_psnr_list))
    # f.close()
    # with open('train_ssim.npy', 'wb') as f:
    #     np.save(f, np.array(train_ssim_list))
    # f.close()
    # with open('test_ssim.npy', 'wb') as f:
    #     np.save(f, np.array(test_ssim_list))
    # f.close()
    # with open('train_loss.npy', 'wb') as f:
    #     np.save(f, np.array(train_loss))
    # f.close()

    scheduler.step(test_psnr)

torch.save(model.state_dict(), f"{exp_name}_model.pt")

