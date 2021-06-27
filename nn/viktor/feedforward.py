
import torch.nn as nn
from ray import tune


class CNN(nn.Module):
    """
    Custom CNN that extends Pytorch's nn module.
    """

    def __init__(self, config):
        super(CNN, self).__init__()
        self.float()

        self.cnn_layers = nn.Sequential(
            # 1st conv layer
            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 2nd conv layer
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.linear_layers = nn.Sequential(
            # Linear combination of conv outputs
            nn.Linear(2048, 2)
        )
            
    def forward(self, x):
        """
        Pass the unrolled images through all of the above defined layers.
        Return the prediction y_hat.
        """
        x = x.unsqueeze(dim=1)
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

    def batch(self, x, y, optimizer, loss_func):
        """
        On a batch of SGD (one image), make one step of gradient descent using BP.
        Return the loss.
        """
        yhat = self.forward(x)
        loss = loss_func(yhat, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    def train(self, train_set, val_set, optimizer, loss_func, device, epochs=20):
        """
        Train the network on the training set, while tracking the losses on the training-
        and validation sets.
        """     
        self.losses = []
        self.val_losses = []

        for epoch in range(epochs):
            running_loss = 0.0
            print("Epoch: " + str(epoch + 1) + " out of " + str(epochs))
            
            for (x, y) in train_set:
                x = x.to(device).float()
                y = y.to(device).float()
                running_loss += self.batch(x, y, optimizer, loss_func)
    
            tune.report(loss=running_loss)
            self.losses.append(running_loss / len(train_set))
            self.val_losses.append(self.validation(val_set, loss_func, device))

    def validation(self, validationLoader, loss_func, device, output=False):
        """
        Compute predictions on the lock-box validation set and print them if desired.
        Return the total loss.
        """
        self.y_compare = []
        loss = 0.0
    
        for (x, y) in validationLoader:
            x = x.to(device).float()
            y = y.to(device).float()

            yhat = self.forward(x)
            loss += loss_func(yhat, y)

            if output:
     
                for i, y_pred in enumerate(yhat):
                    y_true = y[i]
                    self.y_compare.append((y_true.tolist(), y_pred.tolist()))

                    #print("y^=" + str(y_pred[0].item()) + "," + str(y_pred[1].item()) + 
                    #" y=" + str(y_true[0].item()) + "," + str(y_true[1].item()))
            
        return loss.item()
