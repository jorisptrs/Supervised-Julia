import numpy as np
import torch.utils.data as tdata

import requests
import zipfile
import os
import time
import multiprocessing
from sklearn.preprocessing import StandardScaler

class JuliaDataset(tdata.Dataset):
    """
    Our custom dataset.
    """
    def __init__(self, num_cores, debug=True):
        self.num_cores = num_cores
        self.debug = debug
        self.num_images = 0
        self.meta_data = dict()
        self.x = []
        self.maxX = 0
        self.minX = 0
        self.y = []
        self.maxY = 0
        self.minY = 0

    def __len__(self):
        """
        Returns the training set size
        """
        return self.num_images
    
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
            
            self.maxX = int(self.meta_data["ITERATIONS"])
            self.maxY = float(self.meta_data["SAMPLE_RADIUS"])
            self.minY = -self.maxY

    def reader(self, filename):
        """
        Return an array from a comma-separated file
        """
        return np.genfromtxt(filename, delimiter=",")

    def normalize(self, x):
        """
        Normalize the features such that each pixel-value is between 0 and 1.
        Standardize to mean = 0 and std = 1.
        """
        old_shape = x.shape
        x = x.reshape(old_shape[0], -1)
        scaler = StandardScaler().fit(x)
        x = scaler.transform(x)
        x = x.reshape(old_shape)
        return x

    def load_images(self, path, num_images, pooling=True, download=True):
        """
        After ensuring that the dataset is locally present, load features X and labels Y into their
        corresponding fields, both as numpy arrays.
        """
        self.num_images = num_images
        file_names = []

        if download and not os.path.exists(path):
            os.mkdir(path)
            self.download_data(path)

        self.read_header(path)

        if self.debug:
            print("Loading data into RAM...")
            start_time = time.time()

        for index in range(num_images):
            file_names.append(os.path.join(path, "data" + str(index) + '.jset'))

        if pooling:
            pool = multiprocessing.Pool(self.num_cores)
            self.x = pool.map(self.reader, file_names)
        else:    
            for index in range(num_images):
                tempX = self.reader(file_names[index])
                self.x.append(tempX)

        self.x = np.array(self.x, dtype=np.float)
        self.x = self.normalize(self.x)
        self.y = self.reader(os.path.join(path, "labels.txt"))
        self.y = self.y[:num_images]

        if self.debug:
            print("Data loaded in %s seconds." % (time.time() - start_time))
            print("X: " + str(self.x.shape))
            print("Y: " + str(self.y.shape) + "\n")        

    def download_data(self, path, google_drive_id='13jpZFAuGekt3qZoikFs5VaD-0XUO8zdc'):
        """
        Download the dataset from google drive to data the 'data'-folder.
        based on https://github.com/ndrplz/google-drive-downloader
        """      
        url = "https://docs.google.com/uc?export=download"
        session = requests.Session()
        data_path = os.path.join(path, "data")

        if self.debug:
            print("Retrieving data from google drive...")  
        response = session.get(url, params={'id': google_drive_id}, stream=True)

        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                response = session.get(url, params={'id': google_drive_id, 'confirm': value}, stream=True)
                break
             
        with open(data_path, "wb") as f:
            for chunk in response.iter_content(32768):
                if chunk:
                    f.write(chunk)

        try:
            if self.debug:
                print('Unzipping...')
            with zipfile.ZipFile(data_path, 'r') as z:
                z.extractall(os.path.dirname(data_path))
                if self.debug:
                    print("Zipping Complete.")
        except:
            if self.debug:
                print("Zipping Failed.")
