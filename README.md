# Inserting an image within an image (ImageInImage)
## Authors
* Léo Berger
* Titouan Guibert-Lode
* Clément Potier

# Aim of the project
The aim of the project is to implement, test, and compare two distinct approaches for image steganography, alongside proposing improvements or a novel algorithm:

1. **Traditional Method:** 
   - *Reference:* A Steganographic Approach Based on Homomorphic Encryption (Norman Hutte, William Puech, 2026).

2. **AI-Based Method:** 
   - *Reference:* SteganoGAN: High Capacity Image Steganography with GANs (Kevin A. Zhang, Alfredo Cuesta-Infante, Lei Xu, Kalyan Veeramachaneni, 2019).

## Prerequisites
* **Python:** `3.15.2`

## Training
We use the data set COCO_val2017
If you want to train the model you will have to download it and put it in `./src/ia_based/_method/data/` and create a `COCO` folder and image are stored in `COCO/val2017`