# Secura - Steganography Using Generative Adversarial Networks

Secura is a TensorFlow implementation that enables the steganography of RGB images (secret images) within another RGB image (cover image). This project utilizes a Generative Adversarial Network (GAN) to achieve the steganographic embedding.

## Overview
- A TensorFlow implementation enabling steganography of RGB images within another RGB image.
- The model is trained on any image dataset. In this implementation, a subset of the testing subset of the ImageNet Dataset, consisting of 8000 images, was used.
- Images are further categorized into Cover-images and Secret-images. A pair of 4000 images was used to train the model.
- Training was conducted on an Nvidia A100 card for 200 epochs.

## Usage
- Install the required dependencies specified in `requirements.txt` preferably in a new environment
- Jupyter notebook: Includes the implementation of the GAN.
- Dataset Used: https://www.kaggle.com/datasets/lijiyu/imagenet
- Docker image for those who want to test out Secura: `docker pull docker pull daskh/secura:latest`

## Note

- Ensure you have sufficient computing resources, especially if training the model on a large dataset or for a high number of epochs.
- Experiment with different datasets and hyperparameters for optimal results.

#### P.S: utlis.py and train.py are not finished yet, to test out Secura kindly use the docker image or the provided notebook