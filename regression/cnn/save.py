
import torch
import matplotlib.pyplot as plt
import pandas as pd
import time
import numpy as np


class Graph:

    def __init__(self):
        self.folds = []
        self.train_losses = []
        self.val_losses = []
        self.epochs = []
        self.learning_rates = []
        self.alphas = []

    def append(self, fold, train_losses, val_losses, learning_rate, alpha):
        n = len(train_losses)
        self.folds += [fold + 1] * n
        self.epochs += list(range(1, n + 1))
        self.train_losses += train_losses
        self.val_losses += val_losses
        self.learning_rates += [learning_rate] * n
        self.alphas += [alpha] * n

    def save(self, path="data1.csv"):
        df = pd.DataFrame({
            "fold": self.folds,
            "epoch": self.epochs,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "learning_rate" : self.learning_rates,
            "alpha" : self.alphas
        })
        df.to_csv(path, index=False)


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
    output.to_csv('predictions.csv', index=False)

def save_loss(loss_arr, val_arr):
    output = pd.DataFrame(loss_arr)
    output.columns = ['loss']
    output.to_csv('loss.csv', index=False)

def graph_loss(loss_arr, val_losses):
    plt.title("Training loss")
    plt.plot(loss_arr, label="train")
    plt.plot(val_losses, label="validation")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    #plt.show()
    plt.savefig("training_fig" + str(time.time()) + ".png")

def model_save(model, path):
    torch.save(model.state_dict(), path)

def model_load(model, path):
    model.load_state_dict(torch.load(path))

def plot_4(data_set):
    # plotting example images
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))
    for ax, image, label in zip(axes, data_set.x[:4], data_set.y[:4]):
        print(image.shape)
        full_img = np.concatenate((np.flip(np.flip(image, 0), 1), image), axis=0)
        ax.set_axis_off()
        ax.imshow(full_img, cmap=plt.cm.gray_r)
        ax.set_title('y: {}'.format(label))
    plt.show()

def plot_at_idx(data_set, idx):
    fig, ax = plt.subplots(1,1)
    full_img = np.concatenate((np.flip(np.flip(data_set.x[idx], 0), 1), data_set.x[idx]), axis=0)
    ax.set_axis_off()
    plt.imshow(full_img, cmap=plt.cm.gray_r)
    ax.set_title('y: {}'.format(data_set.y[idx]))
    plt.show()
