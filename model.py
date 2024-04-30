# import libraries
import torch
import torch.nn as nn


class NeuralNetworks(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetworks, self).__init__()
        # linear layer
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    #  how input data is processed through these layers to produce an output
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)  # Activation function
        out = self.l2(out)
        return out  # output


