import numpy as np
import matplotlib.pyplot as plt


fig, (ax, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
exp_name = ['latent_variance_large', 'pixel_linear_64', 'pixel_variance_32', 'pixel_variance_64']
for name in exp_name:
    loss = np.load(f'./npy/{name}.npy')
    ax.plot(np.arange(len(loss)), loss, label=name)
loss = np.load(f'./train_loss.npy')
ax1.plot(np.arange(len(loss)), loss, label='latent_nafnet')
ax1.legend()
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Training L1 Loss')

ax.legend()
ax.set_xlabel('Epoch')
ax.set_ylabel('Training L1 Loss')

fig.tight_layout()
plt.savefig(f'./loss.png')

#
# fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(7, 7))
# # fig, ax1 = plt.subplots()
#
# names = ['nafnet_small_', 'nafnet_big_', '']
# # train psnr
# for exp_name in names:
#     train_psnr_list = np.load(f'{exp_name}train_psnr.npy')
#     if exp_name == '':
#         ax1.plot(np.arange(len(train_psnr_list)), train_psnr_list, label='latent nafnet')
#     elif exp_name == 'nafnet_small_':
#         ax1.plot(np.arange(50), train_psnr_list[:50], label='small nafnet')
#     else:
#         ax1.plot(np.arange(50), train_psnr_list[:50], label='big nafnet')
# ax1.legend()
# ax1.set_xlabel('Epoch')
# ax1.set_ylabel('Train PSNR')
#
# # test psnr
# for exp_name in names:
#     train_psnr_list = np.load(f'{exp_name}test_psnr.npy')
#     if exp_name == '':
#         ax2.plot(np.arange(len(train_psnr_list)), train_psnr_list, label='latent nafnet')
#     elif exp_name == 'nafnet_small_':
#         ax2.plot(np.arange(50), train_psnr_list[:50], label='small nafnet')
#     else:
#         ax2.plot(np.arange(50), train_psnr_list[:50], label='big nafnet')
# ax2.legend()
# ax2.set_xlabel('Epoch')
# ax2.set_ylabel('Test PSNR')
#
#
# # train ssim
# for exp_name in names:
#     train_psnr_list = np.load(f'{exp_name}train_ssim.npy')
#     if exp_name == '':
#         ax3.plot(np.arange(len(train_psnr_list)), train_psnr_list, label='latent nafnet')
#     elif exp_name == 'nafnet_small_':
#         ax3.plot(np.arange(50), train_psnr_list[:50], label='small nafnet')
#     else:
#         ax3.plot(np.arange(50), train_psnr_list[:50], label='big nafnet')
# ax3.legend()
# ax3.set_xlabel('Epoch')
# ax3.set_ylabel('Train SSIM')
#
#
# # test ssim
# for exp_name in names:
#     train_psnr_list = np.load(f'{exp_name}test_ssim.npy')
#     if exp_name == '':
#         ax4.plot(np.arange(len(train_psnr_list)), train_psnr_list, label='latent nafnet')
#     elif exp_name == 'nafnet_small_':
#         ax4.plot(np.arange(50), train_psnr_list[:50], label='small nafnet')
#     else:
#         ax4.plot(np.arange(50), train_psnr_list[:50], label='big nafnet')
# ax4.legend()
# ax4.set_xlabel('Epoch')
# ax4.set_ylabel('Test SSIM')
#
# fig.tight_layout()
# plt.savefig(f'./metrics.png')


