
from data import JuliaDataset
from feedforward import Net

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
import matplotlib.pyplot as plt
import numpy as np

import os

BATCH_SIZE = 128
TRAINING_SET_SIZE = 3000

# Width to which each image will be downsampled
W = 28
LATENT_DIMS = 2
EPOCHS = 60
TEST_SET_PROP = .2

device = 'cuda' if torch.cuda.is_available() else 'cpu'

juliaDataset = JuliaDataset()
juliaDataset.load_images(os.path.join('..','trainingData'), TRAINING_SET_SIZE, True, W)

data_loader = torch.utils.data.DataLoader(juliaDataset, batch_size=BATCH_SIZE, shuffle=True)

print("Device: " + device)

def shuffle_split_data(X, y, test_prop):
    # taken from https://stackoverflow.com/questions/35932223/writing-a-train-test-split-function-with-numpy
    arr_rand = np.random.rand(X.shape[0])
    split = arr_rand < np.percentile(arr_rand, 100*(1-test_prop))

    X_train = X[split]
    y_train = y[split]
    X_test =  X[~split]
    y_test = y[~split]

    return X_train, y_train, X_test, y_test

def reshape(x, y, test_prop = .2,):
    train_x, train_y, val_x, val_y = shuffle_split_data(x, y, test_prop)

    train_x = train_x.reshape((-1,1,W,W))

    train_x = torch.Tensor(train_x)

    train_y = train_y.reshape((-1,2))
    train_y = torch.Tensor(train_y)

    val_x = val_x.reshape((-1,1,W,W))
    val_x = torch.Tensor(val_x)

    val_y = val_y.reshape((-1,2))
    val_y = torch.Tensor(val_y)

    return train_x, train_y, val_x, val_y

def feedforward():
    # Train a simple feed-forward NN to predict constants
    # based on https://www.analyticsvidhya.com/blog/2019/10/building-image-classification-models-cnn-pytorch/
    model = Net(W)
    # defining the optimizer
    model.optimizer = Adam(model.parameters(), lr=0.07)
    # defining the loss function
    criterion = nn.MSELoss()
    # checking if GPU is available
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()

    print(model(torch.from_numpy(juliaDataset.x).float()))


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
    feedforward()