import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, conv_layers, fc_layers):
        super(CNN, self).__init__()

        self.convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        for conv_layer in conv_layers:
            self.convs.append(nn.Conv1d(conv_layer['input_channels'], conv_layer['output_channels'], conv_layer['kernel_size']))
            self.pools.append(nn.MaxPool1d(conv_layer['pool_size']))

        self.flatten = nn.Flatten()
        
        self.fcs = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for fc_layer in fc_layers:
            self.fcs.append(nn.Linear(fc_layer['input_size'], fc_layer['output_size']))
            self.dropouts.append(nn.Dropout(fc_layer['dropout_rate']))

        self.out = nn.Linear(fc_layers[-1]['output_size'], 1)

    def forward(self, x):
        for conv, pool in zip(self.convs, self.pools):
            x = F.relu(conv(x))
            x = pool(x)

        x = self.flatten(x)

        for fc, dropout in zip(self.fcs, self.dropouts):
            x = F.relu(fc(x))
            x = dropout(x)

        x = self.out(x)
        return x

conv_layers = [
    {'input_channels': 7, 'output_channels': 64, 'kernel_size': 2, 'pool_size': 2},
    {'input_channels': 64, 'output_channels': 128, 'kernel_size': 2, 'pool_size': 2}
]

fc_layers = [
    {'input_size': 128 * 13, 'output_size': 100, 'dropout_rate': 0.25},
    {'input_size': 100, 'output_size': 50, 'dropout_rate': 0.25}
]

model = CNN(conv_layers, fc_layers)


