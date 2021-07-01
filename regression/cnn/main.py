
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from sklearn.model_selection import KFold

import os
import multiprocessing
import itertools

import data
from feedforward import CNN
import save

DATASET_SIZE = 2000
BATCH_SIZE = 128
TRAINING_SET_PROP = 0.7

N_FOLDS = 3
EPOCHS = 3
CROSSVALIDATION = False

CORES = multiprocessing.cpu_count()
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_NAME = "cnn_fractal_model_v1.jmodel"
DATASET_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)),'..','trainingData')
DEBUG = True


def load_data(path, size):
    """
    Returns the loaded dataset
    """
    dataset = data.JuliaDataset(path, size, CORES, DEBUG)
    return dataset


def onetime_split(dataset):
    """
    Split into training- and validation sets.
    """
    training_size = int(DATASET_SIZE * TRAINING_SET_PROP)
    validation_size = DATASET_SIZE - training_size
    training_set, validation_set = torch.utils.data.random_split(dataset, [training_size, validation_size])

    training_loader = torch.utils.data.DataLoader(training_set, shuffle=True, batch_size=BATCH_SIZE, num_workers=CORES)
    validation_loader = torch.utils.data.DataLoader(validation_set, shuffle=False, batch_size=len(validation_set), num_workers=CORES)

    return (training_loader, validation_loader)


def feedforward(train_loader, val_loader, config):
    """
    Train a CNN for EPOCHS epochs.
    """
    model = CNN(config)
    model.to(DEVICE)

    optimizer = Adam(model.parameters(), config['lr'], weight_decay=config['alpha'])
    loss_func = nn.L1Loss().to(DEVICE)

    train_losses = []
    val_losses = []
        
    for epoch in range(EPOCHS):
        if DEBUG:
            print("Epoch: " + str(epoch + 1) + " out of " + str(EPOCHS))
        train_loss = model.train(train_loader, optimizer, loss_func, DEVICE) 
        train_losses.append(train_loss)
        val_loss, y_actual, y_pred = model.validation(val_loader, loss_func, DEVICE)
        val_losses.append(val_loss)

    return (train_losses, val_losses, y_actual, y_pred)


def crossvalidation(dataset):
    """
    Outer loop of k-fold crossvalidation.
    """
    kfold = KFold(n_splits=N_FOLDS, shuffle=True)
    dataframe = save.DataFrame()

    lrs = [.03]#[.001, .003, .01, .03]
    alphas = [0, 0.09]#,.4,.3,.2,.1,.07,.05] # Originally .09

    # iterate through flexibilities
    for (comb, (lr, alpha)) in enumerate(itertools.product(lrs, alphas)):
        config = {'lr' : lr, 'alpha' : alpha}
        if DEBUG:
            print("Learning rate: " + str(lr))
            print("Alpha: " + str(alpha))
        
        running_val_risk = 0.0
        # iterate through k folds
        for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset.x)):
            if DEBUG:
                print("Fold " + str(fold + 1) + " out of " + str(N_FOLDS))

            # Shuffle data
            train_sampler = torch.utils.data.SubsetRandomSampler(train_ids)
            val_sampler = torch.utils.data.SubsetRandomSampler(val_ids)
            
            # Load data in shuffled order
            train_loader = torch.utils.data.DataLoader(dataset, sampler=train_sampler, batch_size=BATCH_SIZE, num_workers=CORES)
            val_loader = torch.utils.data.DataLoader(dataset, sampler=val_sampler, batch_size=BATCH_SIZE, num_workers=CORES)
     
            # Add final validation risk
            train_losses, val_losses, y_actual, y_pred = feedforward(train_loader, val_loader, config)
            running_val_risk += val_losses[-1]

            dataframe.append_fold(comb, fold, train_losses, val_losses)

        dataframe.append_risk(running_val_risk / N_FOLDS, lr, alpha)

    dataframe.save()


if __name__ == "__main__":
    if DEBUG:
        print("Operating on: " + DEVICE)

    if CROSSVALIDATION:
        juliaDataset = load_data(DATASET_PATH, DATASET_SIZE)
        crossvalidation(juliaDataset)
    else:
        config = {'lr' : 0.03, 'alpha' : 0} # use best config

        juliaDataset = load_data(DATASET_PATH, DATASET_SIZE) # change to final test set
        
        training_loader, validation_loader = onetime_split(juliaDataset)
        train_losses, val_losses, y_actual, y_pred = feedforward(training_loader, validation_loader, config)

        predictions = save.PredictionData()
        predictions.append(y_actual, y_pred)
        predictions.save()
    