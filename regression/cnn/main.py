
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

DATASET_SIZE = 10000
BATCH_SIZE = 128

N_FOLDS = 5
EPOCHS = 25
CROSSVALIDATION = False

TRAINING_SET_PROP = 0.8

CORES = 4 # multiprocessing.cpu_count()
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_NAME = "cnn_fractal_model_v1.jmodel"
MODEL_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)),'..','models')
DATASET_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)),'..','trainingData')
DEBUG = True


def load_data(path, size):
    """
    Returns the loaded dataset
    """
    return data.JuliaDataset(path, size, CORES, DEBUG)   


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
    loss_func = nn.MSELoss().to(DEVICE)

    train_losses = []
    val_losses = []
        
    for epoch in range(EPOCHS):
        if DEBUG:
            print("Epoch: " + str(epoch + 1) + " out of " + str(EPOCHS))
        model.train(train_loader, optimizer, loss_func, DEVICE)
        train_loss, _ , _ = model.compute_error(train_loader, loss_func, DEVICE)
        train_losses.append(train_loss)
        val_loss, y_actual, y_pred = model.compute_error(val_loader, loss_func, DEVICE)
        val_losses.append(val_loss)

    #save.model_save(model, MODEL_NAME)

    return (train_losses, val_losses, y_actual, y_pred)


def crossvalidation(dataset):
    """
    Outer loop of k-fold crossvalidation.
    """
    kfold = KFold(n_splits=N_FOLDS, shuffle=True)
    dataframe = save.DataFrame()

    lrs = [0.0001, 0.001, 0.01, 0.1]
    alphas = [0.0, 0.0001, 0.001, 0.01, 0.1]

    # iterate through flexibilities
    for (comb, (lr, alpha)) in enumerate(itertools.product(lrs, alphas)):
        config = {'lr' : lr, 'alpha' : alpha}
        if DEBUG:
            print("Combination: " + str(comb + 1))
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
        config = {'lr' : 0.001, 'alpha' : 0.01} # use best config

        juliaDataset = load_data(DATASET_PATH, DATASET_SIZE) # change to final test set
        
        training_loader, validation_loader = onetime_split(juliaDataset)
        train_losses, val_losses, y_actual, y_pred = feedforward(training_loader, validation_loader, config)

        predictions = save.PredictionData()
        predictions.append(y_actual, y_pred)
        predictions.save()
        save.graph_loss(train_losses, val_losses)
        save.save_loss(train_losses, val_losses)
        
    