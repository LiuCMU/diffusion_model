# Exploring Latent Diffusion Models and Latent NAFNET for High-resolution Image Deblurring


## Link to dataset
https://seungjunnah.github.io/Datasets/gopro

## Run the models

```
# train a variance diffuser32
python con_latent_diffusion32.py

# train a variance diffuser64
python con_latent_diffusion64.py

# train a linear diffuser
python con_latent_diffusion_lin.py

# train latent nafnet
cd nafnet_benchmark
python train_nafnet_latent.py

# train large nafnet
cd nafnet_benchmark
python train_nafnet_big.py

# train small nafnet
cd nafnet_benchmark
python train_nafnet_small.py

# inference
cd nafnet_benchmark
python test.py

# t-SNE plot
python embedding_test.py
python tsne_visualize.py

```


