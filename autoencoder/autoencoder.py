
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import Sigmoid

class Autoencoder(nn.Module):
    def __init__(self, data_dims, latent_dims, device):
        super(Autoencoder, self).__init__()

        self.device = device # No idea if the gpu thing actually works
        self.to(device)

        self.encoder = nn.Sequential(
            nn.Linear(data_dims, 512),
            nn.ReLU(True),
            nn.Linear(512, latent_dims)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dims, 512),
            nn.ReLU(True),
            nn.Linear(512, data_dims),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.decoder(self.encoder(x))

    def save(self, path="model"):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()

    def train(self, data, epochs=20):
        opt = torch.optim.Adam(self.parameters())
        for epoch in range(epochs):
            for x,_ in data:
                x = x.to(self.device) # GPU
                opt.zero_grad()
                x_hat = self(x)
                loss = ((x - x_hat)**2).sum()
                loss.backward()
                opt.step()


