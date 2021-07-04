
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import requests
import zipfile
import time
import os


class DataFrame:

    def __init__(self):
        self.folds = []
        self.train_losses = []
        self.val_losses = []
        self.epochs = []
        self.learning_rates = []
        self.alphas = []
        self.risks = []
        self.combinations = []

    def append_fold(self, comb, fold, train_losses, val_losses):
        n = len(train_losses)
        self.folds += [fold + 1] * n
        self.combinations += [comb + 1] * n
        self.epochs += list(range(1, n + 1))
        self.train_losses += train_losses
        self.val_losses += val_losses

    def append_risk(self, risk, learning_rate, alpha):
        self.learning_rates.append(learning_rate)
        self.alphas.append(alpha)
        self.risks.append(risk)

    def save(self, path="", index=False):
        df1 = pd.DataFrame({
            "combination" : self.combinations,
            "fold": self.folds,
            "epoch": self.epochs,
            "train_loss": self.train_losses,
            "val_loss": self.val_losses
        })
        df1.to_csv(os.path.join(path, "folds.csv"), index=index)
        df2 = pd.DataFrame({
            "combination" : list(range(1, len(self.risks) + 1)),
            "risk" : self.risks,
            "learning rate" : self.learning_rates,
            "alpha" : self.alphas
        })
        df2.to_csv(os.path.join(path, "risks.csv"), index=index)


class PredictionData:

    def __init__(self):
        self.y_actual_reals = []
        self.y_actual_imgs = []
        self.y_pred_reals = []
        self.y_pred_imgs = []

    def append(self, y_actual, y_pred):
        assert len(y_actual) == len(y_pred)
        for i in range(len(y_actual)):
            for j in range(len(y_actual[i])):
                self.add_row(y_actual[i][j][0].item(), y_actual[i][j][1].item(), y_pred[i][j][0].item(), y_pred[i][j][1].item())

    def add_row(self, y_actual_real, y_actual_img, y_pred_real, y_pred_img):
        self.y_actual_reals.append(y_actual_real)
        self.y_actual_imgs.append(y_actual_img)
        self.y_pred_reals.append(y_pred_real)
        self.y_pred_imgs.append(y_pred_img)

    def save(self, path="", index=False):
        df = pd.DataFrame({
            "y_actual_real" : self.y_actual_reals,
            "y_actual_img" : self.y_actual_imgs,
            "y_pred_real" : self.y_pred_reals,
            "y_pred_img" : self.y_pred_imgs
        })
        df.to_csv(os.path.join(path, "predictions.csv"), index=index)


def save_loss(train_losses, val_losses, path="", index=False):
    df = pd.DataFrame({
        "train_loss" : train_losses,
        "val_loss" : val_losses
    })
    df.to_csv(os.path.join(path, "loss.csv"), index=index)


def graph_loss(loss_arr, val_losses):
    plt.title("Training loss")
    plt.plot(loss_arr, label="train")
    plt.plot(val_losses, label="validation")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    # plt.show()
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


def unzip_data(path, zip_name, debug=True):
    """
    Unzip Data Folder
    """      
    data_path = os.path.join(path, zip_name)

    new_folder = zip_name.replace(".zip","")
    new_path = os.path.join(path, new_folder)

    if not os.path.exists(new_path):
        try:
            if debug:
                print('Unzipping...')
            with zipfile.ZipFile(data_path, 'r') as z:
                z.extractall(new_path)
                if debug:
                    print("Zipping Complete.")
        except:
            if debug:
                print("Zipping Failed.")

    return new_path
