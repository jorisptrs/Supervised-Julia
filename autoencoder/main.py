
from juliaDataset import JuliaDataset
from autoencoder import Autoencoder
from feedforward import Net

import torch
import matplotlib.pyplot as plt
import numpy as np

import os

BATCH_SIZE = 128
TRAINING_SET_SIZE = 8
# Width to which each image will be downsampled
W = 26
LATENT_DIMS = 2
EPOCHS = 100
TEST_SET_PROP = .1

device = 'cuda' if torch.cuda.is_available() else 'cpu'

juliaDataset = JuliaDataset()
juliaDataset.load_images(os.path.join('..','trainingData','data'), TRAINING_SET_SIZE, True, W)

data_loader = torch.utils.data.DataLoader(juliaDataset, batch_size=BATCH_SIZE, shuffle=True)


def feedforward():
    # Train a simple feed-forward NN to predict constants
    net = Net(W)
    net.train(juliaDataset.x, juliaDataset.y, device, TEST_SET_PROP, EPOCHS)
    out = net(input)
    print(out)
    #net.zero_grad()
    #out.backward(torch.randn(1, 10))


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
    # Plot random genereations ONLY WORKS FOR 2 LATENT DIMENSIONS!
    plot_reconstructed(autoencoder, images)
    plt.show()

def plot_comparison(img):
    n = min(5,TRAINING_SET_SIZE)
    f, ax = plt.subplots(n,2)
    for i in range(n):
        # Plot training example
        ax[i,0].imshow(juliaDataset.x[i].reshape(W, W))
        # Plot reconstructed training example    
        ax[i,1].imshow(img[i].reshape(W, W))

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
            img[(n-1-i)*W:(n-1-i+1)*W, j*W:(j+1)*W] = x_hat
    plt.imshow(img, extent=[*r0, *r1])

autoencode()
#feedforward()