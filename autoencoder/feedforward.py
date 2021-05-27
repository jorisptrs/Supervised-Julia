import torch
import torch.nn as nn
import torch.nn.functional as F
import random
#from sklearn.model_selection import train_test_split

class Net(nn.Module):

    def __init__(self, w):
        self.w = w
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    # Replacement of sklearn implementation, 
    def train_test_split(dataset, split=0.60):
        train = list()
        train_size = split * len(dataset)
        dataset_copy = list(dataset)
        while len(train) < train_size:
            index = randrange(len(dataset_copy))
            train.append(dataset_copy.pop(index))
        return train, dataset_copy
    
    def train(self, x, y, device, test_prop = .1, epochs = 20):
        #random.shuffle(x)
        torch.utils.data.random_split(dataset, lengths)
        train_x, val_x, train_y, val_y = train_test_split(x, y, test_size = test_prop)
        
        train_x.reshape((-1,1,self.w,self.w))
        train_x.x = torch.Tensor(train_x)

        train_y.reshape((-1,2))
        train_y = torch.Tensor(train_y)

        val_x.reshape((-1,1,self.w,self.w))
        val_x.x = torch.Tensor(val_x)

        val_y.reshape((-1,2))
        val_y = torch.Tensor(val_y)

        (train_x.shape, train_y.shape), (val_x.shape, val_y.shape)
