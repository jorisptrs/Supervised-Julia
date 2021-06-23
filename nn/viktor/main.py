
from data import JuliaDataset
from feedforward import Net

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
import matplotlib.pyplot as plt
import numpy as np

import os

BATCH_SIZE = 32
TRAINING_SET_SIZE = 320


# Width to which each image will be downsampled
W = 64
LATENT_DIMS = 2
EPOCHS = 80
TEST_SET_PROP = .8

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def feedforward():
    # Train a simple feed-forward NN to predict constants
    # based on https://www.analyticsvidhya.com/blog/2019/10/building-image-classification-models-cnn-pytorch/
    model = Net(W)
    # defining the optimizer
    model.optimizer = Adam(model.parameters(), lr=0.005)
    # defining the loss function
    criterion = nn.MSELoss()
    # checking if GPU is available
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()

    model.train(
        data_loader,
        validation_loader,
        device,
        criterion, 
        EPOCHS
    )

    model.validation(validation_loader, device, criterion, True)


def plot_comparison(img):
    n = min(5,TRAINING_SET_SIZE)
    f, ax = plt.subplots(n,2)
    for i in range(n):
        # Plot training example
        ax[i,0].imshow(juliaDataset.x[i].reshape(W, W), cmap='gray', vmin=0, vmax=1)
        # Plot reconstructed training example
        ax[i,1].imshow(img[i].reshape(W, W), cmap='gray', vmin=0, vmax=1)

def plot_reconstructed(autoencoder, img):
    r0=(-5, 10)
    r1=(-10, 5)
    n=12
    plt.figure()
    img = np.zeros((n*W, n*W))
    if LATENT_DIMS != 2:
        print("Please implement plot_reconstructed() for more than 2 latent dims")
        return
    for i, y in enumerate(np.linspace(*r1, n)):
        for j, x in enumerate(np.linspace(*r0, n)):
            z = torch.Tensor([[x, y]]).to(device)
            x_hat = autoencoder.decoder(z)
            x_hat = x_hat.reshape(W, W).to('cpu').detach().numpy()
            img[(n-1-i)*W:(n-1-i+1)*W, j*W:(j+1)*W] = x_hat # UGLY :-()---<===D
    plt.imshow(img, extent=[*r0, *r1])


if __name__ == "__main__":
    juliaDataset = JuliaDataset()
    juliaDataset.load_images(os.path.join('..','trainingData'), TRAINING_SET_SIZE, True, W, pooling =False)

    train_n = int(TRAINING_SET_SIZE * TEST_SET_PROP)
    valid_n = TRAINING_SET_SIZE - train_n
    train, validation = torch.utils.data.random_split(juliaDataset, [train_n, valid_n])
    data_loader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation, shuffle=False, batch_size=len(validation))
    feedforward()