
import numpy as np
import torch

class JuliaDataSet(torch.utils.data.Dataset):

    def __init__(self):
        self.num_images = 0
        self.image_size = 0
        self.x = []
        self.y_target = []
       
    def load_images(self, path, num_images):       
        for index in range(num_images):
            with open(path + str(index) + '.jset', 'r') as file:
                y_target_str = file.readline()[2:].rstrip()
                x_str_lines = file.readlines()

                self.y_target.append(np.fromstring(y_target_str, dtype=np.float32, sep=','))

                x_lines = []
                for line in x_str_lines:
                    x_lines.append(np.fromstring(line, dtype=np.float32, sep=','))

                self.x.append(self.compress(np.array(x_lines)))

        self.y_target = np.array(self.y_target)
        self.x = np.array(self.x)
        self.num_images = num_images
        self.image_size = self.x.shape[1]

    def compress(self, x):
        for i in range(26):
            for j in range(26):
                x[i, j] = np.mean(x[i*20:(i+1)*20, j*20:(j+1)*20])
        return x[:26,:26].reshape(676)
        
    def __len__(self):
        return self.num_images
    
    def __getitem__(self, idx):
        return (self.x[idx], self.y_target[idx])