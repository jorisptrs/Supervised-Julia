import torch.nn as nn
from torch.optim import Adam, SGD

import os
import multiprocessing
import data
from feedforward import CNN
import save

# Hyper hyper
import torch.optim as optim
from ray import tune


DATASET_SIZE = 100
BATCH_SIZE = 128
TEST_SET_PROP = 0.7

EPOCHS = 10
LEARNING_RATE = 0.005
L2_NORM_REG_PENALTY = 0.09

CORES = multiprocessing.cpu_count()
DEVICE = 'cpu' #cuda' if torch.cuda.is_available() else 'cpu'
MODEL_NAME = "cnn_fractal_model_v1.jmodel"
DATASET_PATH = os.path.join(os.path.realpath(__file__),'..','trainingData')

def feedforward(config):
    """
    Train a CNN to predict constants
    """
    julia_dataset = data.JuliaDataset()
    print("1")
    (training_loader, validation_loader) = julia_dataset.get_split_data()
    print("2")
    model = CNN()
    model.to(DEVICE)

    optimizer = SGD(model.parameters(), config['lr'], weight_decay=L2_NORM_REG_PENALTY)
    loss_func = nn.MSELoss().to(DEVICE)

    model.train(training_loader, validation_loader, optimizer, loss_func, DEVICE, EPOCHS)
    #total_validation_error = model.validation(validation_loader, loss_func, DEVICE, output=True)
    #print(total_validation_error)

    # save.model_save(model, MODEL_NAME)
    # save.graph_loss(model.losses, model.val_losses)
    # save.save_loss(model.losses, model.val_losses)
    # save.save_predictions(model.y_compare)

if __name__ == "__main__":
    analysis = tune.run(
        feedforward,
        config={'lr': tune.grid_search([.001, .01, .1])}
        )

    print("Best config: ", analysis.get_best_config(metric="loss"))