
from juliaDataset import JuliaDataset
from autoencoder import Autoencoder

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
import matplotlib.pyplot as plt
import numpy as np

import os

BATCH_SIZE = 128
TRAINING_SET_SIZE = 10
# Width to which each image will be downsampled
W = 28
LATENT_DIMS = 2
EPOCHS = 100
TEST_SET_PROP = .2

device = 'cuda' if torch.cuda.is_available() else 'cpu'

juliaDataset = JuliaDataset()
juliaDataset.load_images(os.path.join('..','trainingData'), TRAINING_SET_SIZE, True, W)

data_loader = torch.utils.data.DataLoader(juliaDataset, batch_size=BATCH_SIZE, shuffle=True)


def autoencode():
    # Train autoencoder
    autoencoder = Autoencoder(juliaDataset.image_vec_size, LATENT_DIMS).to(device)
    autoencoder.train(data_loader, device, EPOCHS)

    # Save trained model
    autoencoder.save()

    # En- and Decoded training examples
    output = autoencoder.decoder(autoencoder.encoder(torch.from_numpy(juliaDataset.x)))
    images = output.to('cpu').detach().numpy()

    # Plot Training examples vs output of autoencoder
    plot_comparison(images)
    plt.show()


def plot_comparison(img):
    n = min(5,TRAINING_SET_SIZE)
    f, ax = plt.subplots(n,2)
    for i in range(n):
        # Plot training example
        ax[i,0].imshow(juliaDataset.x[i].reshape(W, W), cmap='gray', vmin=0, vmax=1)
        # Plot reconstructed training example
        ax[i,1].imshow(img[i].reshape(W, W), cmap='gray', vmin=0, vmax=1)


if __name__ == "__main__":
    autoencode()