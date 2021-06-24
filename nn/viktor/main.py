
from data import JuliaDataset
from feedforward import Net

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
import matplotlib.pyplot as plt
import numpy as np

import os

BATCH_SIZE = 32
TRAINING_SET_SIZE = 10


# Width to which each image will be downsampled
W = 64
LATENT_DIMS = 2
EPOCHS = 15
TEST_SET_PROP = .8
POOLING = True

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


if __name__ == "__main__":
    juliaDataset = JuliaDataset()
    juliaDataset.load_images(os.path.join('..','trainingData'), TRAINING_SET_SIZE, True, W, pooling =POOLING)

    train_n = int(TRAINING_SET_SIZE * TEST_SET_PROP)
    valid_n = TRAINING_SET_SIZE - train_n
    train, validation = torch.utils.data.random_split(juliaDataset, [train_n, valid_n])
    data_loader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation, shuffle=False, batch_size=len(validation))
    feedforward()