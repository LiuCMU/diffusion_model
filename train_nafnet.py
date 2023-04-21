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
from torch.utils.data import DataLoader
from torchvision import transforms, models
from data import img_dataset
from model import NAFNet
from SSIM import ssim
from PIL import Image


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

#parameters
if "pro" in hostname.lower():  #my mac
    train_path = "/Users/liu5/Documents/10-617HW/project/data/train"
    test_path = "/Users/liu5/Documents/10-617HW/project/data/test"
elif "exp" in hostname.lower():  #expanse 
    #copy, extract tar file to the local computing node scratch
    tar_path = "/expanse/lustre/projects/cwr109/zhen1997/img_deblur.tar"
    local_scratch = f"/scratch/{os.environ['USER']}/job_{os.environ['SLURM_JOB_ID']}"
    print("computing node local scratch path: %s" % local_scratch)
    shutil.copy(tar_path, local_scratch)
    tar = tarfile.open(os.path.join(local_scratch, "img_deblur.tar"))
    tar.extractall(local_scratch)
    tar.close()
    print("Finished extracting the dataset")
    train_path = os.path.join(local_scratch, "img_deblur/train")
    test_path = os.path.join(local_scratch, "img_deblur/test")

    device = torch.device("cuda")  #only using 1 GPU

elif ("braavos" in hostname.lower()) or ( "storm" in hostname.lower()):  #braavos/stromland
    train_path = "/storage/users/jack/MS_ML_datasets/img_deblur/train"
    test_path = "/storage/users/jack/MS_ML_datasets/img_deblur/test"

parser = argparse.ArgumentParser()
parser.add_argument("--num_layers", type=int, default=1)
parser.add_argument("--conv_pad", type=int, default=1)
parser.add_argument("--hidden_channels", type=int, default=20)
parser.add_argument("--pool_pad", type=int, default=2)
args = parser.parse_args()
wandb.init(project="617", entity="img_deblur", config=args)
config = wandb.config

lr = 0.0005
patience = 10
epochs = 1000
batchsize = 2


def calc_PSNR(img1, img2):
    """
    calculating PSNR based on 10.3390/s20133724
    img1/img1: tensor of shape (N, C, H, W)
    """
    N, C, H, W = img1.shape
    denominator = (1/(N*C*H*W)) * torch.sum(torch.square(img1 - img2))
    psnr = 10*torch.log10((255*255)/denominator)
    return psnr.item()

# https://github.com/KupynOrest/DeblurGAN/blob/master/models/losses.py
class PerceptualLoss():
	def contentFunc(self):
		conv_3_3_layer = 14
		cnn = models.vgg19(pretrained=True).features
		cnn = cnn.cuda()
		model = torch.nn.Sequential()
		model = model.cuda()
		for i,layer in enumerate(list(cnn)):
			model.add_module(str(i),layer)
			if i == conv_3_3_layer:
				break
		return model
		
	def __init__(self, loss):
		self.criterion = loss
		self.contentFunc = self.contentFunc()
			
	def get_loss(self, fakeIm, realIm):
		f_fake = self.contentFunc.forward(fakeIm)
		f_real = self.contentFunc.forward(realIm)
		f_real_no_grad = f_real.detach()
		loss = self.criterion(f_fake, f_real_no_grad)
		return loss

def validate(loader):
    model.eval()
    with torch.no_grad():
        psnrs, ssims, sizes = [], [], []
        for batch in loader:
            xs_, ys_ = batch
            xs_ = xs_.to(device)
            ys_ = ys_.to(device)
            yhat = model(xs_).detach()
            ys = ys_.detach()
            psnr = calc_PSNR(yhat, ys)
            ssim_i = ssim(yhat, ys).item()
            # ssim_i = 0

            psnrs.append(psnr)
            ssims.append(ssim_i)
            sizes.append(len(batch))
    score = np.dot(psnrs, sizes)/np.sum(sizes)
    ssim_score = np.dot(ssims, sizes)/np.sum(sizes)
    return (score, ssim_score)


def init_params(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, a=1.0)
        torch.nn.init.zeros_(m.bias)

#load datasets
train = img_dataset(train_path, debug=False, scale=True) # debug = False
train_loader = DataLoader(train, batchsize, num_workers=4)
test = img_dataset(test_path, debug=False, scale=True) # debug = False
test_loader = DataLoader(test, batchsize, num_workers=4)
print("Number of training and testing: %i, %i" % (len(train), len(test)))

# model = Conv(args.num_layers, args.conv_pad, args.hidden_channels, args.pool_pad)
# model = NAFNet()
model = NAFNet(width=32, middle_blk_num=1, enc_blk_nums=[1, 1, 1, 28], dec_blk_nums=[1, 1, 1, 1])
model.to(device)
model.apply(init_params)
# model.load_state_dict(torch.load("latest_nafnet100.pt", map_location=device))
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=patience, mode="max")
print("Total number of trainable parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

results = []
for i in range(epochs):
    epoch = i+1
    learning_rate = optimizer.param_groups[0]['lr']
    model.train()
    Yhats, Ys = [], []
    for xs, ys in train_loader:
        xs = xs.to(device)
        ys = ys.to(device)
        yhat = model(xs)
        N, C, H, W = ys.shape
        denominator = (1/(N*C*H*W)) * torch.sum(torch.square(ys - yhat))
        loss = -10*torch.log10((255*255)/denominator)
        # content_loss = PerceptualLoss(torch.nn.MSELoss())
        # loss = content_loss.get_loss(yhat, ys)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    
    train_psnr, train_ssim = validate(train_loader)
    test_psnr, test_ssim = validate(test_loader)
    print("Epoch %i LR: %.4f Training PSNR: %.2f SSIM: %.2f Test PSNR: %.2f SSIM %.2f" % (epoch, learning_rate,  train_psnr, train_ssim, test_psnr, test_ssim))
    wandb.log({
        "learning_rate": learning_rate,
        "train_psnr": train_psnr,
        "train_ssim": train_ssim,
        "test_psnr": test_psnr,
        "test_ssim": test_ssim
    })
    results.append((epoch, learning_rate,  train_psnr, train_ssim, test_psnr, test_ssim))
    if i//10 == 0:
        torch.save(model.state_dict(), "latest_nafnet36blocks.pt")
    scheduler.step(test_psnr)
torch.save(model.state_dict(), "model_nafnet36blocks.pt")
pickle.dump(results, open("results_nafne36blockst.pkl", "wb"))
