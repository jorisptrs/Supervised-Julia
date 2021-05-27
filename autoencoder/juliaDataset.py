
import numpy as np
import torch.utils.data as tdata

class JuliaDataset(tdata.Dataset):

    def __init__(self):
        self.num_images = 0
        self.image_vec_size = 0
        self.x = []
        self.y = []
       
    def load_images(self, path, num_images, compress=True, compressed_width=26):       
        for index in range(num_images):
            with open(path + str(index) + '.jset', 'r') as file:
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