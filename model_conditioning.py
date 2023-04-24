import torch
import torch.nn as nn
import torchvision
from data import *
from torch.utils.data import DataLoader


class ConditioningEncoder(nn.Module):
    def __init__(self, encoder_name, embed_dim):
        super(ConditioningEncoder, self).__init__()
        # images need to be rescaled to [0.0, 1.0] and then normalized using mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]
        if encoder_name == 'vgg':
            self.vgg = torchvision.models.vgg11(weights='DEFAULT')
            self.model = nn.Sequential(
                self.vgg,
                nn.Linear(1000, embed_dim)
            )
        elif encoder_name == 'resnet':
            self.resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', weights='DEFAULT')
            self.model = nn.Sequential(
                self.resnet,
                nn.Linear(1000, embed_dim)
            )
        else:
            raise Exception("invalid model name")


    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    train_path = '../../../net/projects/ranalab/kh310/img_deblur/train'
    train = img_dataset(train_path, debug=False, scale=True)
    train_loader = DataLoader(train, 4, num_workers=4)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = ConditioningEncoder('resnet', 256).to(device)

    for i, (xs, ys) in enumerate(train_loader):
        xs = xs.to(device)
        ys = ys.to(device)
        yhat = model(xs)  # (B, embed_dim)

        break
