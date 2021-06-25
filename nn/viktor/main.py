
# Pregrine Modules
import torch
import torch.nn as nn
from torch.optim import Adam, SGD

import os
from data import JuliaDataset
from feedforward import CNN


BATCH_SIZE = 32
TRAINING_SET_SIZE = 10
W = 64 # Scale width
LATENT_DIMS = 2
EPOCHS = 15
TEST_SET_PROP = .8
POOLING = True
LEARNING_RATE = 0.005
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def feedforward(data_loader, validation_loader):
    """
    Train a simple feed-forward NN to predict constants
    based on https://www.analyticsvidhya.com/blog/2019/10/building-image-classification-models-cnn-pytorch/
    """    
    model = CNN(W)

    model.optimizer = Adam(model.parameters(), lr=LEARNING_RATE) # also try SGD
    loss_func = nn.MSELoss()

    # checking if GPU is available
    if torch.cuda.is_available(): # Check if this is actually correct
        model = model.cuda()
        loss_func = loss_func.cuda()

    model.train(data_loader, validation_loader, DEVICE, loss_func, EPOCHS)
    model.validation(validation_loader, DEVICE, loss_func, True)

def load_data():
    juliaDataset = JuliaDataset()
    juliaDataset.load_images(os.path.join('..','trainingData'), TRAINING_SET_SIZE, True, 26, pooling=POOLING)

    train_n = int(TRAINING_SET_SIZE * TEST_SET_PROP)
    valid_n = TRAINING_SET_SIZE - train_n
    train, validation = torch.utils.data.random_split(juliaDataset, [train_n, valid_n])

    data_loader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation, shuffle=False, batch_size=len(validation))

    return (data_loader, validation_loader)


if __name__ == "__main__":
    (data_loader, validation_loader) = load_data()
    feedforward(data_loader, validation_loader)