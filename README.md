# Efficient Dynamic Scene Deblurring with Diffusion Models


## Abstract
Image deblurring is a seminal challenge in the field of computer vision, whose objective is to restore a clear image from a blurred one. Dynamic scene deblurring is particularly challenging due to the presence of multiple moving objects, which makes it difficult to recover the sharp image using conventional deblurring techniques. Currently, most deblurring methods operate directly in the pixel space using convolutional neural networks (CNNs), and while they have produced impressive results, they are computationally demanding, limiting their application to high-resolution images. This project aims to explore more efficient deblurring methods in the latent space using diffusion models. By transforming the images into the latent space, the computational cost is reduced and irrelevant pixels are removed, enabling the model to focus on task-relevant details. Diffusion models are well-suited for this task as they do not exhibit mode collapse and can capture complex distributions. The performance of the latent diffusion model will be compared against several baseline methods, including CNNs and NAFNETs (https://arxiv.org/abs/2204.04676). If time permits, the project will also investigate the applicability of the method to related image editing tasks, such as superresolution and object removal.

## Background
Diffusion models (Sohl-Dickstein, et al. 2015) are a relative new area of research in ML. They are a type of generative models. They can generate new data after learning the underlying diffusion process of the training dataset. 
They are typically trained on a large dataset. The models use density funtions (for example Gaussian) to desceribe the distribution of data. The parameters are optmized through MLE or gradient-based methods. Diffusion models have many potential applications, including image and audio synthesis, anormal data detection(https://arxiv.org/abs/1809.04758), data augmentation, targeted molecule generation, etc.



