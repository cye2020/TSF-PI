import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_shape=(30, 7), units=[128, 64], dropout=0.25):
        super(RNN, self).__init__()

        self.layers = nn.ModuleList()
        for unit in units:
            self.layers.append(nn.LSTM(input_size=input_shape[1], hidden_size=unit, bidirectional=True, batch_first=True))
            self.layers.append(nn.Dropout(p=dropout))
            input_shape = (input_shape[0], unit * 2)  # Update input_shape for the next layer

        self.layers.extend([nn.Linear(in_features=60, out_features=1), nn.Tanh(), nn.Dropout(p=dropout), nn.Flatten()])

    def forward(self, x):
        for layer in self.layers:
            x, _ = layer(x)
        return x


# 사용 예시
if __name__ == '__main__':
    model = RNN(units=[128, 64])
    print(model)