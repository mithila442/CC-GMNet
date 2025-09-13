import torch
import torch.nn as nn

class TimeSeriesCNN(nn.Module):
    def __init__(self, input_channels=5, output_size=50):
        super(TimeSeriesCNN, self).__init__()
        self.output_size = output_size
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, output_size)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        b, n, t, c = x.shape
        x = x.view(b * n, t, c)
        x = x.transpose(1, 2)
        x = self.net(x)
        x = x.view(b, n, self.output_size)
        return x
