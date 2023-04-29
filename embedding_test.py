import numpy as np
import torch
from torch.utils.data import DataLoader
from data import img_dataset
from modules import ED
from dall_e import map_pixels, unmap_pixels, load_model
import torch.nn.functional as F
from tsnecuda import TSNE
import matplotlib.pyplot as plt


train_path = '../../../net/projects/ranalab/kh310/img_deblur/train'
test_path = '../../../net/projects/ranalab/kh310/img_deblur/test'


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load datasets
batch_size = 4
test = img_dataset(test_path, debug=False, scale=False, fname=True)
test_loader = DataLoader(test, batch_size, num_workers=0)


encoder_decoder = ED('./nafnet_benchmark', device)

xs_latent_all = []
fname = []
for j, (xs, ys, xfile, yfile) in enumerate(test_loader): # scale=False
    print(f"batch: {j}/{len(test_loader)}")
    xs = xs.to(device)
    ys = ys.to(device)
    # move to latent space
    xs_latent = encoder_decoder.encode(xs).float()  # shape(B, H/8, W/8)
    xs_latent_all.append(xs_latent / encoder_decoder.vocab_size)
    fname += list(xfile)


xs_latent_all = torch.cat(xs_latent_all)
xs_latent_all = xs_latent_all.view(xs_latent_all.size()[0], xs_latent_all.size()[1]*xs_latent_all.size()[2])

labels = {}
curr_label = 0
for file in fname:
    if file[:14] not in labels:
        labels[file[:14]] = curr_label
        curr_label += 1


X_labels = np.array([labels[f[:14]] for f in fname])
xs_latent_all = xs_latent_all.cpu().numpy()

with open('./latent_x_new.npy', 'wb') as f:
    np.save(f, xs_latent_all)
f.close()

with open('./embedded_labels_new.npy', 'wb') as f:
    np.save(f, X_labels)
f.close()


# X_embedded = TSNE(n_components=2, perplexity=15, learning_rate=10).fit_transform(xs_latent_all.cpu())




