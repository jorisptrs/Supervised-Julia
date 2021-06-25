
import requests
import zipfile
import os
import numpy as np
import torch.utils.data as tdata
from multiprocessing import Pool
import time

# TODO bluescale images

def normalize(lower, upper, x):
    magnitude = upper - lower
    nX = (x - lower) / magnitude
    return nX

     

class JuliaDataset(tdata.Dataset):

    def __init__(self):
        self.num_images = 0
        self.image_vec_size = 0
        self.x = []
        self.y = []

        self.maxX = self.minX = None
       
    def reader(self, filename):
        return np.genfromtxt(filename, delimiter=",")


    def load_images(self, path, num_images, compress=True, compressed_width=26, pooling = False):
        if not os.path.exists(path):
            os.mkdir(path)
            self.download_data(path)

        with open(os.path.join(path, "header.txt")) as f:
            
            for line in f:
                args = line.split("=")
                if len(args) < 2:
                    continue
                name = args[0]
                value = args[1]

                if name == "ITERATIONS":
                    self.maxX = int(value)
                    self.minX = 0
                
                if name == "SAMPLE_RADIUS":
                    temp = float(value)
                    self.maxY = temp
                    self.minY = -temp

        print("Data loading ... ")
        start_time = time.time()

        if pooling:
            filesx = ["../trainingData/data" + str(index) + '.jset' for index in range(num_images)]
           
            pool = Pool(4)
            xs = pool.map(self.reader, filesx)
            self.x = np.array(xs, dtype=np.float)
        else:    
            for index in range(num_images):
                print(index)
                tempX = np.genfromtxt(os.path.join(path, "data" + str(index) + '.jset'), delimiter=",")
                tempX = self.compress(tempX, compressed_width) if compress else tempX
                self.x.append(tempX)
            self.y = np.array(self.y, dtype=np.float)
            self.x = np.array(self.x, dtype=np.float)

        print("--- %s seconds ---" % (time.time() - start_time))

        self.x = normalize(self.minX, self.maxX, self.x)
        self.y = np.genfromtxt(os.path.join(path, "labels.txt"), delimiter=",")
        self.y = self.y[:num_images]


        print("Data loaded")
        print(self.x.shape)
        print(self.y.shape)

        self.num_images = num_images
        self.image_vec_size = self.x.shape[1]

    def download_data(self, path, google_drive_id='13jpZFAuGekt3qZoikFs5VaD-0XUO8zdc', print_status=True):      
        URL = "https://docs.google.com/uc?export=download"
        chunk_size = 32768
        session = requests.Session()
        data_path = os.path.join(path, "data")

        if print_status:
            print("Retrieving data from google drive...")  
        response = session.get(URL, params={'id': google_drive_id}, stream=True)

        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                response = session.get(URL, params={'id': google_drive_id, 'confirm': value}, stream=True)
                break
             
        with open(data_path, "wb") as f:
            for chunk in response.iter_content(chunk_size):
                if chunk:
                    f.write(chunk)

        try:
            if print_status:
                print('Unzipping...')
            with zipfile.ZipFile(data_path, 'r') as z:
                z.extractall(os.path.dirname(data_path))
                if print_status:
                    print("Zipping Complete.")
        except:
            if print_status:
                print("Zipping Failed.")

    def compress(self, img, compressed_width):
        n = (int) (img.shape[0] / compressed_width)
        for i in range(compressed_width):
            for j in range(compressed_width):
                img[i, j] = np.mean(img[i*n:(i+1)*n, j*n:(j+1)*n])
        return img[:compressed_width,:compressed_width] 

    # Functions required to be implemented by torch Dataset object    

    def __len__(self):
        return self.num_images
    
    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])