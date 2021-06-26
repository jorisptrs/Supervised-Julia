
import torch
import torch.nn as nn
from torch.optim import Adam, SGD

import os
import multiprocessing
from data import JuliaDataset
from feedforward import CNN
import save


DATASET_SIZE = 100
BATCH_SIZE = 128
TEST_SET_PROP = 0.7

EPOCHS = 15
LEARNING_RATE = 0.005
L2_NORM_REG_PENALTY = 0.09

CORES = multiprocessing.cpu_count()
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_NAME = "cnn_fractal_model_v1.jmodel"
DATASET_PATH = os.path.join('..','trainingData')


def load_data():
    juliaDataset = JuliaDataset(CORES)
    juliaDataset.load_images(DATASET_PATH, DATASET_SIZE)

    # Split Data
    training_size = int(DATASET_SIZE * TEST_SET_PROP)
    validation_size = DATASET_SIZE - training_size
    training_set, validation_set = torch.utils.data.random_split(juliaDataset, [training_size, validation_size])

    data_loader = torch.utils.data.DataLoader(training_set, shuffle=True, batch_size=BATCH_SIZE, num_workers=CORES)
    validation_loader = torch.utils.data.DataLoader(validation_set, shuffle=False, batch_size=len(validation_set), num_workers=CORES)

    return (data_loader, validation_loader)

def feedforward():
    """
    Train a simple feed-forward NN to predict constants
    based on https://www.analyticsvidhya.com/blog/2019/10/building-image-classification-models-cnn-pytorch/
    """
    (data_loader, validation_loader) = load_data()    
    model = CNN()
    model.to(DEVICE)

    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=L2_NORM_REG_PENALTY) # also try SGD
    loss_func = nn.MSELoss().to(DEVICE)

    model.train(data_loader, validation_loader, optimizer, loss_func, DEVICE, EPOCHS)
    model.validation(validation_loader, loss_func, DEVICE, output=True)

    save.model_save(model, MODEL_NAME)
    save.graph_loss(model.losses, model.valLosses)
    save.save_loss(model.losses, model.valLosses)
    save.save_predictions(model.y_compare)

if __name__ == "__main__":
    feedforward()