
import torch
import matplotlib.pyplot as plt
import pandas as pd


def save_loss(loss_arr, val_arr):
    output = pd.DataFrame(list(zip(loss_arr, val_arr)))
    output.columns = ['loss', 'val']
    output.to_csv('loss.dat')

def graph_loss(loss_arr, valLosses):
    plt.title("Training loss")
    plt.plot(loss_arr, label="train")
    plt.plot(valLosses, label="validation")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

def model_save(model, path):
    torch.save(model.state_dict(), path)

def model_load(model, path):
    model.load_state_dict(torch.load(path))
