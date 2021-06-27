
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from sklearn.model_selection import KFold

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

DEBUG = True


def load_data():
    """
    Returns the loaded dataset
    """
    dataset = JuliaDataset(CORES, DEBUG)
    dataset.load_images(DATASET_PATH, DATASET_SIZE)
    return dataset

def onetime_split(dataset):
    """
    Split into training- and validation sets.
    """
    training_size = int(DATASET_SIZE * TEST_SET_PROP)
    validation_size = DATASET_SIZE - training_size
    training_set, validation_set = torch.utils.data.random_split(dataset, [training_size, validation_size])

    training_loader = torch.utils.data.DataLoader(training_set, shuffle=True, batch_size=BATCH_SIZE, num_workers=CORES)
    validation_loader = torch.utils.data.DataLoader(validation_set, shuffle=False, batch_size=len(validation_set), num_workers=CORES)

    return (training_loader, validation_loader)

def crossvalidation(dataset, n_folds=5):
    kfold = KFold(n_splits=n_folds, shuffle=True)

    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset.x)):
        if DEBUG:
            print("Fold " + str(fold + 1) + " out of " + str(n_folds))

        train_sampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_sampler = torch.utils.data.SubsetRandomSampler(val_ids)

        train_loader = torch.utils.data.DataLoader(dataset, sampler=train_sampler, batch_size=BATCH_SIZE, num_workers=CORES)
        val_loader = torch.utils.data.DataLoader(dataset, sampler=val_sampler, batch_size=BATCH_SIZE, num_workers=CORES)
        
        feedforward(train_loader, val_loader)
    
    # find average accuracy
    

def feedforward(train_loader, val_loader):
    """
    Train a CNN to predict constants
    """
    model = CNN()
    model.to(DEVICE)

    # TODO also try SGD
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=L2_NORM_REG_PENALTY)
    loss_func = nn.MSELoss().to(DEVICE)

    model.train(train_loader, val_loader, optimizer, loss_func, DEVICE, EPOCHS)
    model.validation(val_loader, loss_func, DEVICE, output=True)

    save.model_save(model, MODEL_NAME)
    save.graph_loss(model.losses, model.val_losses)
    save.save_loss(model.losses, model.val_losses)
    save.save_predictions(model.y_compare)

if __name__ == "__main__":
    juliaDataset = load_data()
    crossvalidation(juliaDataset)

    # go back to old format
    # training_loader, validation_loader = onetime_split(juliaDataset)
    # feedforward(training_loader, validation_loader)