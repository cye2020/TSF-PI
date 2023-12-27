import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CNN(nn.Module):
    def __init__(self, conv_layers, fc_layers):
        super(CNN, self).__init__()
        
        self.convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.activations = nn.ModuleList()
        
        for conv_layer in conv_layers:
            input_channels, input_length = conv_layer['input_shape']
            output_channels = conv_layer['output_channels']
            kernel_size = conv_layer['kernel_size']
            pool_size = conv_layer['pool_size']
            activation = conv_layer['activation']
            
            self.convs.append(nn.Conv1d(input_channels, output_channels, kernel_size))
            self.pools.append(nn.MaxPool1d(pool_size))
            self.activations.append(nn.ReLU() if activation == 'relu' else nn.Linear())
            
        conv_output_length = ((input_length - kernel_size) / 1) + 1  # stride is assumed to be 1
        pool_output_length = conv_output_length // pool_size

        output_size = output_channels * int(pool_output_length)

        self.flatten = nn.Flatten()
        fc_layers[0]['input_size'] = output_size
        
        self.fcs = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.fc_activations = nn.ModuleList()
        
        for fc_layer in fc_layers:
            input_size = fc_layer['input_size']
            output_size = fc_layer['output_size']
            activation = None if 'activation' not in fc_layer else nn.ReLU() if fc_layer['activation'] == 'relu' else nn.Linear()
            dropout = nn.Dropout(fc_layer['dropout_rate']) if 'dropout_rate' in fc_layer else None
            
            self.fcs.append(nn.Linear(input_size, output_size))
            self.dropouts.append(dropout)
            self.fc_activations.append(activation)


    def forward(self, x):
        for conv, pool, activation in zip(self.convs, self.pools, self.activations):
            x = activation(conv(x))
            x = pool(x)

        x = self.flatten(x)

        for fc, dropout, activation in zip(self.fcs, self.dropouts, self.fc_activations):

            x = fc(x)
            
            if isinstance(activation, nn.Module):
                x = activation(x)
            
            if isinstance(dropout, nn.Module):
                x = dropout(x)

        return x



if __name__ =='__main__':
    # input_shape = (channel, length)
    conv_layers = [
        {'input_shape': (7, 30), 'output_channels': 64, 'kernel_size': 2, 'pool_size': 2, 'activation': 'relu'}
    ]

    fc_layers = [
        {'input_size': None, 'output_size': 100, 'activation': 'relu', 'dropout_rate': 0.25},
        {'input_size': 100, 'output_size': 30}
    ]


    model = CNN(conv_layers, fc_layers)



    summary(model, (7, 30), device='cuda' if torch.cuda.is_available() else 'cpu')