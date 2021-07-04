"""
Used for final testing dataset
"""

import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from feedforward import CNN
import data
import main
import save

ALL_DATA = -1
TEST_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)),'..','testData')


def final_exam(model):
    """
    Performs an intensive test on the trained model
    """
    test_ds = data.JuliaDataset(TEST_PATH, ALL_DATA, main.CORES, False)
    test_df = torch.utils.data.DataLoader(test_ds, shuffle=True, batch_size=len(test_ds), num_workers=main.CORES)
    _, y, yhat =  model.compute_error(test_df, nn.L1Loss().to(main.DEVICE), main.DEVICE)

    y = y[0].numpy()
    yhat = yhat[0].numpy()

    loss = np.power(y - yhat, 2)
    loss = np.sum(loss, axis=1)

    y = np.transpose(y)
    yhat = np.transpose(yhat)

    df = pd.DataFrame({
        "yreal" : y[0],
        "yimag" : y[1],
        "loss" : loss,
        "yhatreal": yhat[0],
        "yhatimag": yhat[1]
    })
    df.to_csv(os.path.join(TEST_PATH, "loss.csv"), index=False) 

def count_parameters(model):
    """
    Return the amount of parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    
    config = {'lr' : 0.001, 'alpha' : 0.01} # use best config
    model = CNN(config)
    save.model_load(model, os.path.join(main.MODEL_PATH, main.MODEL_NAME))

    # print(count_parameters(model))
    final_exam(model)