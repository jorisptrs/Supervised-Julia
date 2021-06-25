
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def save_predictions(pred_arr):
    y_true_real = []
    y_true_img = []
    y_pred_real = []
    y_pred_img = []
    for i in range(len(pred_arr)):
        y_true = pred_arr[i][0]
        y_pred = pred_arr[i][1]
        y_true_real.append(y_true[0])
        y_true_img.append(y_true[1])
        y_pred_real.append(y_pred[0])
        y_pred_img.append(y_pred[1])

    output = pd.DataFrame(list(zip(y_true_real, y_true_img, y_pred_real, y_pred_img)))
    output.columns = ['y_true_real', 'y_true_img', 'y_pred_real', 'y_pred_img']
    output.to_csv('predictions.csv')

def save_loss(loss_arr, val_arr):
    output = pd.DataFrame(list(zip(loss_arr)))
    output.columns = ['loss']
    output.to_csv('loss.dat')

def graph_loss(loss_arr, valLosses):
    plt.title("Training loss")
    plt.plot(loss_arr, label="train")
    #plt.plot(valLosses, label="validation")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

def model_save(model, path):
    torch.save(model.state_dict(), path)

def model_load(model, path):
    model.load_state_dict(torch.load(path))
