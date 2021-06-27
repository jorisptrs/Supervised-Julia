from numpy import mod
import torch.nn as nn
import torch
from torch.optim import Adam, SGD
import torch.optim as optim
from sklearn.model_selection import KFold

import os
import multiprocessing

import data
from feedforward import CNN
import save

# Hyper hyper
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler


DATASET_SIZE = 1000
BATCH_SIZE = 128
TEST_SET_PROP = 0.7

N_FOLDS = 2
EPOCHS = 10
LEARNING_RATE = 0.005
L2_NORM_REG_PENALTY = 0.09

CORES = multiprocessing.cpu_count()
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_NAME = "cnn_fractal_model_v1.jmodel"
DATASET_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)),'..','trainingData')

DEBUG = True


def load_data():
    """
    Returns the loaded dataset
    """
    dataset = data.JuliaDataset(DATASET_PATH, DATASET_SIZE, CORES, DEBUG)
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


def feedforward(config):
    """
    Train a CNN for EPOCHS epochs.
    Notify tune as shown in https://docs.ray.io/en/master/tune/index.html
    """
    juliaDataset = load_data()
    train_loader, val_loader = onetime_split(juliaDataset)
    model = CNN(config)
    model.to(DEVICE)

    optimizer = SGD(model.parameters(), config['lr'], weight_decay=L2_NORM_REG_PENALTY)
    loss_func = nn.MSELoss().to(DEVICE)
    model.losses = []
    model.val_losses = []

    for epoch in range(EPOCHS):
        print("Epoch: " + str(epoch + 1) + " out of " + str(EPOCHS))
        train_loss = model.train(train_loader, val_loader, optimizer, loss_func, DEVICE)
        val_loss = model.validation(val_loader, loss_func, DEVICE, output=True)
        print("Loss: " + str(val_loss))
        tune.report(mean_loss=val_loss)
    
    # save.model_save(model, MODEL_NAME)
    # save.graph_loss(model.losses, model.val_losses)
    # save.save_loss(model.losses, model.val_losses)
    # save.save_predictions(model.y_compare)


def crossvalidation(dataset):
    """
    Outer loop of k-fold crossvalidation.
    """
    kfold = KFold(n_splits=N_FOLDS, shuffle=True)

    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset.x)):
        if DEBUG:
            print("Fold " + str(fold + 1) + " out of " + str(N_FOLDS))

        # Shuffle data
        train_sampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_sampler = torch.utils.data.SubsetRandomSampler(val_ids)
        
        # Load data in shuffled order
        train_loader = torch.utils.data.DataLoader(dataset, sampler=train_sampler, batch_size=BATCH_SIZE, num_workers=CORES)
        val_loader = torch.utils.data.DataLoader(dataset, sampler=val_sampler, batch_size=BATCH_SIZE, num_workers=CORES)
        
        feedforward(train_loader, val_loader, {'lr' : LEARNING_RATE})
    
    # find average accuracy

if __name__ == "__main__":
    if DEBUG:
        print("Operating on " + DEVICE)
    #juliaDataset = load_data()
    #crossvalidation(juliaDataset)

    #feedforward({'lr' : LEARNING_RATE})

    #ray.init(log_to_driver=False)
    param_space = {"lr": ray.tune.grid_search([0.001, 0.01, 0.1])}
    analysis = ray.tune.run(
        feedforward, config=param_space)
    print("Best config: ", analysis.get_best_config(metric="mean_loss", mode="min"))
    #df = analysis.dataframe()
