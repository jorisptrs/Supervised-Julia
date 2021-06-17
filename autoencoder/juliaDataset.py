
import requests
import zipfile
import os
import numpy as np
import torch.utils.data as tdata

# TODO bluescale images

class JuliaDataset(tdata.Dataset):

    def __init__(self):
        self.num_images = 0
        self.image_vec_size = 0
        self.x = []
        self.y = []
       
    def load_images(self, path, num_images, compress=True, compressed_width=26):
        if not os.path.exists(path):
            os.mkdir(path)
            self.download_data(path)

        for index in range(num_images):
            with open(os.path.join(path, "data" + str(index) + '.jset'), 'r') as file:
                y_str = file.readline()[2:].rstrip()
                x_str_lines = file.readlines()

                self.y.append(np.fromstring(y_str, dtype=np.float32, sep=','))

                x_lines = []
                for line in x_str_lines:
                    x_lines.append(np.fromstring(line, dtype=np.float32, sep=','))

                training_example = np.array(x_lines)
                training_example_compressed = self.compress(training_example, compressed_width) if compress else training_example
                self.x.append(training_example_compressed)

        self.y = np.array(self.y)
        self.x = np.array(self.x)
        self.num_images = num_images
        self.image_vec_size = self.x.shape[1]

    def download_data(self, path, google_drive_id='1a-24ZjCuuvV-QnRcxuQPl_NKiqin-sXz', print_status=True):      
        if print_status:
            print("Retrieving data from google drive...")  

        URL = "https://docs.google.com/uc?export=download"
        chunk_size = 32768
        response = requests.Session().get(URL, params={'id': google_drive_id}, stream=True)
        data_path = os.path.join(path, "data")
             
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
        return img[:compressed_width,:compressed_width].reshape(compressed_width**2) 

    # Functions required to be implemented by torch Dataset object    

    def __len__(self):
        return self.num_images
    
    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])