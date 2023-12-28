import torch
import torch.nn as nn
from torchinfo import summary


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CNN(nn.Module):
    def __init__(self, input_shape, conv_layers, fc_layers):
        super(CNN, self).__init__()
        
        modules = []
        input_length = input_shape[1]
        
        for conv_layer in conv_layers:
            input_channels = conv_layer['input_channels']
            output_channels = conv_layer['output_channels']
            kernel_size = conv_layer['kernel_size']
            pool_size = conv_layer['pool_size']
            activation = conv_layer['activation']
            stride = conv_layer['stride']
                    
            modules.append(nn.Conv1d(input_channels, output_channels, kernel_size, stride))
            modules.append(nn.ReLU() if activation == 'relu' else nn.Linear())
            modules.append(nn.MaxPool1d(pool_size))
                    
            conv_output_length = ((input_length - kernel_size) / stride) + 1
            pool_output_length = conv_output_length // pool_size

            output_size = output_channels * int(pool_output_length)
                    
            input_length = pool_output_length


        modules.append(nn.Flatten())
        fc_layers[0]['input_size'] = output_size
        
        for fc_layer in fc_layers:
            input_size = fc_layer['input_size']
            output_size = fc_layer['output_size']
            activation = None if 'activation' not in fc_layer else nn.ReLU() if fc_layer['activation'] == 'relu' else nn.Linear()
            dropout = fc_layer.get('dropout', 0)
            
            modules.append(nn.Linear(input_size, output_size))
            if activation is not None:
                modules.append(activation)
            if dropout > 0:
                modules.append(nn.Dropout(dropout))

        self.layers = nn.Sequential(*modules)

    def forward(self, x):
        return self.layers(x)




if __name__ =='__main__':
    input_shape = (7, 30)
    
    conv_layers = [
        {'input_channels': 7, 'output_channels': 64, 'kernel_size': 2, 'pool_size': 2, 'stride': 1 , 'activation': 'relu'},
        {'input_channels': 64, 'output_channels': 32, 'kernel_size': 2, 'pool_size': 2, 'stride': 1, 'activation': 'relu'},
    ]

    fc_layers = [
        {'input_size': None, 'output_size': 100, 'activation': 'relu', 'dropout': 0.25},
        {'input_size': 100, 'output_size': 30}
    ]


    model = CNN(input_shape, conv_layers, fc_layers)



    summary(model, (1, 7, 30), device='cuda' if torch.cuda.is_available() else 'cpu')