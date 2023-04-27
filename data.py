import os
import glob
import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset


class img_dataset(Dataset):
    def __init__(self, path, debug=False, scale=True):
        super(img_dataset, self).__init__()
        """
        path: a folder containing blurred/x and sharp/y folders
        """
        self.path = path

        #collect data points
        data = []
        x_paths = os.path.join(self.path, "blur_gamma", "*.png")
        xs = glob.glob(x_paths)
        for x in xs:
            basename = os.path.basename(x)
            y = os.path.join(self.path, "sharp", basename)
            data.append((x, y))
        if debug:
            self.data = data[:10]
        else:
            self.data = data

        self.convertor = transforms.ToTensor()
        self.scale = scale

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        if self.scale:
            scale = 255  #convert the range between (0, 255)
        else:
            scale = 1
        x_tensor = scale * self.convertor(Image.open(x))  #(3, 720, 1280), before scale range(0, 1)
        y_tensor = scale * self.convertor(Image.open(y))  #(3, 720, 1280)
        return (x_tensor, y_tensor)
