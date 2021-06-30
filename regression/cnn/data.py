
import numpy as np
import torch
from sklearn import preprocessing

import os
import time
import multiprocessing

import save


class JuliaDataset(torch.utils.data.Dataset):
    """
    Our custom dataset.
    """
    
    def __init__(self, dataset_path, dataset_size, num_cores, debug=True):
        self.dataset_path = dataset_path
        self.dataset_size = dataset_size
        self.num_cores = num_cores
        self.debug = debug
        self.meta_data = dict()
        self.x = []
        self.y = []
        self.load_images()

    def __len__(self):
        """
        Returns the training set size
        """
        return self.dataset_size
    
    def __getitem__(self, idx):
        """
        Returns features and labels at index idx
        """
        return (self.x[idx], self.y[idx])

    def read_header(self, path, header_name="header.txt"):
        """
        Read the meta-data from the header file.
        """
        with open(os.path.join(path, header_name)) as f:
            for line in f.readlines():
                line = line.strip().split("=")
                self.meta_data[line[0]] = line[1]

    def reader(self, filename):
        """
        Return an array from a comma-separated file
        """
        return np.genfromtxt(filename, delimiter=",")

    def normalize(self, x):
        """
        Min-max scale all pixel values to a range of [0, 1]
        """
        old_shape = x.shape
        x = x.reshape(old_shape[0], -1)
        scaler = preprocessing.MinMaxScaler()
        #scaler = preprocessing.StandardScaler()
        x = scaler.fit_transform(x)
        x = x.reshape(old_shape)
        return x

    def load_images(self, pooling=True, download=True):
        """
        After ensuring that the dataset is locally present, load features X and labels Y into their
        corresponding fields, both as numpy arrays.
        """
        file_names = []

        if download and not os.path.exists(self.dataset_path):
            os.mkdir(self.dataset_path)
            save.download_data(self.dataset_path, debug=self.debug)

        self.read_header(self.dataset_path)

        if self.debug:
            print("Loading data into (V)RAM...")
            start_time = time.time()

        for index in range(self.dataset_size):
            file_names.append(os.path.join(self.dataset_path, "data" + str(index) + '.jset'))

        if pooling:
            pool = multiprocessing.Pool(self.num_cores)
            self.x = pool.map(self.reader, file_names)
        else:
            for index in range(self.dataset_size):
                tempX = self.reader(file_names[index])
                self.x.append(tempX)

        self.x = np.array(self.x, dtype=np.float)
        self.x = self.normalize(self.x)
        self.y = self.reader(os.path.join(self.dataset_path, "labels.txt"))
        self.y = self.y[:self.dataset_size]

        if self.debug:
            print("Data loaded in %s seconds." % (time.time() - start_time))
            print("X: " + str(self.x.shape))
            print("Y: " + str(self.y.shape) + "\n")        
