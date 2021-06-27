import numpy as np
import torch.utils.data as tdata
import torch

import requests
import zipfile
import os
import time
import multiprocessing
from sklearn import preprocessing

BATCH_SIZE = 128
CORES = multiprocessing.cpu_count()
DATASET_SIZE = 100
DATASET_PATH = os.path.join(os.path.realpath(__file__),'..','trainingData')
TEST_SET_PROP = 0.7


class JuliaDataset(tdata.Dataset):
    """
    Our custom dataset.
    """
<<<<<<< HEAD
    def __init__(self, debug=True):
=======
    
    def __init__(self, num_cores, debug=True):
        self.num_cores = num_cores
>>>>>>> 1a40688884e00b1710211e459ded351c94213441
        self.debug = debug
        self.num_images = 0
        self.meta_data = dict()
        self.x = []
        self.maxX = 0
        self.minX = 0
        self.y = []
        self.maxY = 0
        self.minY = 0
        self.load_images(DATASET_PATH, DATASET_SIZE)

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
        Min-max scale all pixel values to a range of [0, 1]
        """
        old_shape = x.shape
        x = x.reshape(old_shape[0], -1)
        scaler = preprocessing.MinMaxScaler()
        #scaler = preprocessing.StandardScaler()
        x = scaler.fit_transform(x)
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
            print("Loading data into (V)RAM...")
            start_time = time.time()

        for index in range(num_images):
            file_names.append(os.path.join(path, "data" + str(index) + '.jset'))

        if pooling:
            pool = multiprocessing.Pool(CORES)
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

    def get_split_data(self):
        """
        Returns the loaded dataset, split into training- and validation sets.
        """
        # Split Data
        training_size = int(DATASET_SIZE * TEST_SET_PROP)
        validation_size = DATASET_SIZE - training_size
        training_set, validation_set = torch.utils.data.random_split(self, [training_size, validation_size])

        training_loader = torch.utils.data.DataLoader(training_set, shuffle=True, batch_size=BATCH_SIZE, num_workers=CORES)
        validation_loader = torch.utils.data.DataLoader(validation_set, shuffle=False, batch_size=len(validation_set), num_workers=CORES)

        return (training_loader, validation_loader)
