import torch.optim as optim

from ray import tune
from ray.tune.examples.mnist_pytorch import get_data_loaders, ConvNet, train, test