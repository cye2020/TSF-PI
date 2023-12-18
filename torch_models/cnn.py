import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, input_shape=(1, 7, 30),
                conv_configs=[{'filters': 64, 'kernel_size': 2, 'dropout': 0}],
                pool_size=2,
                dense_configs=[{'neurons': 100, 'dropout': 0.25}, {'neurons': 30, 'dropout': 0}]):
        super(CNN, self).__init__()

        # Create a list to store the layers
        layers = []

        in_size = input_shape[1]
        output_height = input_shape[2]
        
        for config in conv_configs:
            layers.append(nn.Conv1d(in_channels=in_size, out_channels=config['filters'], kernel_size=config['kernel_size']))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(kernel_size=pool_size))
            if config['dropout'] > 0:
                layers.append(nn.Dropout(config['dropout']))
            in_size = config['filters']
            output_height = ((output_height - config['kernel_size']) + 1) // pool_size
        
        in_size = in_size * output_height
        # Flatten the output
        layers.append(nn.Flatten())
        for config in dense_configs:
            layers.append(nn.Linear(in_features=in_size, out_features=config['neurons']))
            layers.append(nn.ReLU())
            if config['dropout'] > 0:
                layers.append(nn.Dropout(config['dropout']))
            in_size = config['neurons']
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# 사용 예시
if __name__ == '__main__':
    conv_configs = [
        {'filters': 64, 'kernel_size': 2, 'dropout': 0},
        {'filters': 32, 'kernel_size': 2, 'dropout': 0}
    ]

    dense_configs = [
        {'neurons': 100, 'dropout': 0.25},
        {'neurons': 50, 'dropout': 0.25},
        {'neurons': 30, 'dropout': 0}
    ]

    input_shape = (1, 5, 30)  # PyTorch의 입력 형식 (배치 크기, 채널 수, 시간 단계)

    model = CNN(input_shape=input_shape, conv_configs=conv_configs, dense_configs=dense_configs)
    print(model)




