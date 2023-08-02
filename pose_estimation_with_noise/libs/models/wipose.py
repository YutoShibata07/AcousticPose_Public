import torch.nn as nn
import torchvision
import torch.nn.functional as F
from .wisppn_resnet import get_wisppn
import torch

class Wipose_LSTM(nn.Module):
    def __init__(self, in_cha=7, out_cha=4, debug=False):
        super().__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv1d(in_cha, 64, kernel_size=16, stride=1, padding = 'same'),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.SiLU(),
        )
        self.cnn2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=16, stride=1, padding = 'same'),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.SiLU(),
        )
        self.cnn3 = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=8, padding = 'same'),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.SiLU(),
        )
        self.cnn4 = nn.Sequential(
            nn.Conv1d(64, 1, kernel_size=4, padding = 'same'),
            nn.Flatten(),
            nn.BatchNorm1d(2400),
            nn.Dropout(0.2),
            nn.SiLU(),
        )
        self.cnn = nn.Sequential(
            self.cnn1,
            self.cnn2,
            self.cnn3,
            self.cnn4,
        )
        self.hidden_size = 544
        self.lstm = nn.LSTM(input_size = 2400, hidden_size = self.hidden_size, dropout = 0.1)
        self.seq_len = 12
        self.fc1 = nn.Sequential(
            nn.Linear(self.hidden_size, out_cha),
        )
        self.debug = debug

    def forward(self, x, pos=None):
        x = torch.fft.fft(x, dim = -1)
        x = torch.abs(x)
        # print('2:',x.shape)
        # assert False
        x = [self.cnn(x[:,i,:,:]).unsqueeze(1) for i in range(self.seq_len)] #input[bs, 4, freq_bins] -> output[bs, 1, 932]
        x = torch.cat(x, dim = 1) #[bs, 5, 932]
        x = x.permute(1,0,2)
        x, _ = self.lstm(x)
        x = x.permute(1,0,2)
        x = self.fc1(x)
        return x