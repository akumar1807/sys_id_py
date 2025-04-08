import torch
import torch.nn as nn
import numpy as np

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.l1 = nn.Linear(4,8) #Input Layer
        self.hl1 = nn.Linear(8, 8) # First hidden layer
        self.l2 = nn.Linear(8, 2) # Output layer

    def forward(self, x):
        x = nn.ReLU(self.l1(x))
        x = nn.ReLU(self.hl1(x))
        return self.l2(x)
    
    def initialise_weights(self):
        for layer in model.modules():
                if isinstance(layer, nn.Linear):
                        nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                        nn.init.zeros_(layer.bias)

model = NeuralNetwork()
print(model)