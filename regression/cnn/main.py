from numpy import mod
import torch.nn as nn
import torch
from torch.optim import Adam, SGD
from sklearn.model_selection import KFold

import os
import multiprocessing

import data
from feedforward import CNN
import save
import itertools

DATASET_SIZE = 1000
BATCH_SIZE = 128
TEST_SET_PROP = 0.7

N_FOLDS = 2
EPOCHS = 15

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


def feedforward(train_loader, val_loader, config, show_results = False):
    """
    Train a CNN for EPOCHS epochs.
    """
    model = CNN(config)
    model.to(DEVICE)

    optimizer = SGD(model.parameters(), config['lr'], weight_decay=config['alpha'])
    loss_func = nn.MSELoss().to(DEVICE)

    train_losses = []
    val_losses = []
        
    for epoch in range(EPOCHS):
        print("Epoch: " + str(epoch + 1) + " out of " + str(EPOCHS))
        train_loss = model.train(train_loader, optimizer, loss_func, DEVICE)
        train_losses.append(train_loss)
        val_loss = model.validation(val_loader, loss_func, DEVICE)
        val_losses.append(val_loss)

    if show_results:
        save.model_save(model, MODEL_NAME)
        save.graph_loss(train_losses, val_losses)
        save.save_loss(train_losses, val_losses)
        #save.save_predictions(model.y_compare)

    return val_losses

def crossvalidation(dataset):
    """
    Outer loop of k-fold crossvalidation.
    """
    kfold = KFold(n_splits=N_FOLDS, shuffle=True)

    lrs = [.03]#[.001, .003, .01, .03]
    alphas = [.001, .003, .01, .03]
    #lrs = [x for x in range(0,.5,.05)]
    risks = []
    # iterate through flexibilities
    for (lr, alpha) in itertools.product(lrs, alphas):
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
            running_val_risk += feedforward(train_loader, val_loader, config)[-1]
        risks.append((config, running_val_risk / N_FOLDS))
    
    best_config = min(risks, key = lambda t: t[1])[0]
    
    # Plot THE curve
    alpha_dict = {x : 0.0 for x in alphas}
    for (config, risk) in risks:
        alpha_dict[config['alpha']] += risk/len(lrs)
    print(alpha_dict)
    
    if DEBUG:
        print("Optimal config: ", best_config)

    (train_loader, val_loader) = onetime_split(dataset)
    feedforward(train_loader, val_loader, best_config, True)


if __name__ == "__main__":
    if DEBUG:
        print("Operating on " + DEVICE)
    juliaDataset = load_data()
    crossvalidation(juliaDataset)

    #feedforward({'lr' : LEARNING_RATE})
    