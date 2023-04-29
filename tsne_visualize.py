import numpy as np
import matplotlib.pyplot as plt
# from tsnecuda import TSNE
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# X_embedded = TSNE(n_components=2, perplexity=15, learning_rate=10).fit_transform(xs_latent_all.cpu())

X_latent = np.load('./latent_x_new.npy')
X_labels = np.load('./embedded_labels_new.npy')
print(X_latent.shape)

import pdb; pdb.set_trace()
# do PCA before t-SNE
pca = PCA(n_components=50)
X_pca = pca.fit_transform(X_latent)

# t-SNE
# X_embedded = TSNE(n_components=2, perplexity=15, learning_rate=10).fit_transform(X_pca)
X_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=50).fit_transform(X_latent)

fig, ax = plt.subplots()
# s=0.6,c=labels,cmap='tab10'
ax.scatter(X_embedded[:,0], X_embedded[:,1], s=1.7, c=X_labels, cmap='tab10')

fig.suptitle('t-SNE of GOPRO image embeddings')
fig.tight_layout()
plt.savefig(f'./tsne_sklearn_norm.png')



# X_embedded = np.load('./embedded_x.npy')
# X_labels = np.load('./embedded_labels.npy')
# print(X_embedded.shape)
#
# fig, ax = plt.subplots()
# # s=0.6,c=labels,cmap='tab10'
# ax.scatter(X_embedded[:,0], X_embedded[:,1], s=1.7, c=X_labels, cmap='tab10')
#
# fig.suptitle('t-SNE of GOPRO image embeddings')
# fig.tight_layout()
# plt.savefig(f'./tsne.png')
