
from juliaDataset import JuliaDataset
from autoencoder import Autoencoder

import torch
import matplotlib.pyplot as plt
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'



juliaDataset = JuliaDataset()
juliaDataset.load_images('trainingData/data', 4)

data_loader = torch.utils.data.DataLoader(juliaDataset, batch_size=128, shuffle=True)

autoencoder = Autoencoder(juliaDataset.image_vec_size, 2).to(device)
autoencoder.train(data_loader, device)

autoencoder.save()

# Test

plt.imshow(juliaDataset.x[0].reshape((26, 26)))
plt.show()

output = autoencoder.decoder(autoencoder.encoder(torch.from_numpy(juliaDataset.x)))
img = output.to('cpu').detach().numpy()

plt.imshow(img[0].reshape((26, 26)))
plt.show()


def plot_reconstructed(autoencoder, r0=(-5, 10), r1=(-10, 5), n=12):
    w = 26
    img = np.zeros((n*w, n*w))
    for i, y in enumerate(np.linspace(*r1, n)):
        for j, x in enumerate(np.linspace(*r0, n)):
            z = torch.Tensor([[x, y]]).to(device)
            x_hat = autoencoder.decoder(z)
            x_hat = x_hat.reshape(26, 26).to('cpu').detach().numpy()
            img[(n-1-i)*w:(n-1-i+1)*w, j*w:(j+1)*w] = x_hat
    plt.imshow(img, extent=[*r0, *r1])
    plt.show()

plot_reconstructed(autoencoder)