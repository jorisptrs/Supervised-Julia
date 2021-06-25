
# Pregrine Modules
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class CNN(nn.Module):

    def __init__(self, w):
        super(CNN, self).__init__()
        
        self.w = w
        self.optimizer = None
        self.losses = None
        self.valLosses = None
        self.validation_loader = None

        self.float()

        self.cnn_layers = nn.Sequential(

            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 2nd convolution layer
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(2048, 2)
        )

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def validation(self, validationLoader, device, loss_func, output=False):
        
        loss = 0.0
        # TODO this should not be a forcycle
        yhat = None
        y = None
        x = None
        for x,y in validationLoader:
            x = x.to(device)
            y = y.to(device)

            yhat = self.forward(x.float())
            loss += loss_func(yhat, y)
        
        if output:
            torch.set_printoptions(edgeitems=14)
            for i, pred in enumerate(yhat):
                yreal = y[i]
                #print(x[i])
                print("y^=" + str(pred[0].item()) + "," + str(pred[1].item()) + 
                " y=" + str(yreal[0].item()) + "," + str(yreal[1].item()))
            


        return loss.item()
            
    # Defining the forward pass    
    def forward(self, x):
        x = x.unsqueeze(dim=1)
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

    def batch(self, x, y, loss_func):
        yhat = self.forward(x)
        loss = loss_func(yhat, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
         
    def graph_loss(self):
        plt.title("Training loss")
        plt.plot(self.losses, label="train")
        plt.plot(self.valLosses, label="validation")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    def train(self, trainLoader, validationLoader, device, loss_func, epochs = 20):        

        self.losses = []
        self.valLosses = []
        for epoch in range(epochs):

            running_loss = 0.0

            print("Epochs: " + str(epoch + 1) + " out of " + str(epochs))

            n = 0
            for x, y in trainLoader:

                x = x.to(device)
                y = y.to(device)

                n += 1
                running_loss += self.batch(x.float(), y.float(), loss_func)
            
            self.losses.append(running_loss / n)
            self.valLosses.append(self.validation(validationLoader, device, loss_func))
            

        self.graph_loss()
        


                
                

            