import torch
import torch.nn as nn
from torchinfo import summary

class RNN(nn.Module):
    def __init__(self, input_shape, lstm_layers, fc_layers):
        super(RNN, self).__init__()

        modules = []
        input_length = input_shape[1]
        
        output_size = 0
        for lstm_layer in lstm_layers:
            input_size = lstm_layer['input_size']
            hidden_size = lstm_layer['hidden_size']
            dropout = lstm_layer.get('dropout', 0)

            modules.append(nn.LSTM(input_size, hidden_size, dropout=dropout, num_layers=(1+(dropout>0)), bidirectional=True))

        output_size = hidden_size * 2  # Bi-directional LSTM doubles the output size

        modules.append(nn.Linear(output_size, fc_layers[0]['output_size']))
        activation = None if 'activation' not in fc_layers[0] else self.get_activation(fc_layers[0]['activation'])
        modules.append(nn.Dropout(dropout))
        
        modules.append(nn.Flatten())
        fc_layers[1]['input_size'] = fc_layers[0]['output_size'] * input_length

        for fc_layer in fc_layers[1:]:
            input_size = fc_layer['input_size']
            output_size = fc_layer['output_size']
            activation = None if 'activation' not in fc_layer else self.get_activation(fc_layer['activation'])
            dropout = fc_layer.get('dropout', 0)

            modules.append(nn.Linear(input_size, output_size))
            if activation is not None:
                modules.append(activation)
            if dropout > 0:
                modules.append(nn.Dropout(dropout))

        self.layers = nn.Sequential(*modules)

    def get_activation(self, activation):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        elif activation == 'tanh':
            return nn.Tanh()
        else:
            raise ValueError(f'Invalid activation: {activation}')

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.LSTM):
                x, _ = layer(x)
            else:
                x = layer(x)
        return x


if __name__ == '__main__':
    input_shape = (7, 30)
    
    # LSTM layers
    lstm_layers = [
        {
            'input_size': 7,  # input_shape=(30,7)으로부터
            'hidden_size': 128,
            'dropout': 0.25
        },
        {
            'input_size': 256,  # 128 * 2 (Bi-directional LSTM)
            'hidden_size': 64,
            'dropout': 0.25
        },
        # {
        #     'input_size': 128,  # 64 * 2 (Bi-directional LSTM)
        #     'hidden_size': 32,
        #     'dropout': 0.25
        # },
        # {
        #     'input_size': 64,  # 64 * 2 (Bi-directional LSTM)
        #     'hidden_size': 16,
        #     'dropout': 0.25
        # }
    ]

    # Fully connected layers
    fc_layers = [
        {
            'input_size': None,
            'output_size': 60,
            'dropout': 0.25,
            'activation': 'tanh'
        },
        {
            'input_size': 60,
            'output_size': 1,
        }
    ]

    model = RNN(input_shape, lstm_layers, fc_layers)

    summary(model, (1, 30, 7), device='cuda' if torch.cuda.is_available() else 'cpu')