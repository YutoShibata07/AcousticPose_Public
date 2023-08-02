import torch.nn as nn
import torchvision
import torch.nn.functional as F
from .wisppn_resnet import get_wisppn
import torch
from .functions import get_downsampling_block1, get_downsampling_block2_1d, get_downsampling_block1_1d, get_downsampling_block2
class Speech2pose(nn.Module):
    def __init__(self, in_cha = 4, out_cha = 4):
        super().__init__()
        self.cha_1 = 64
        self.down_k = 4
        self.k = 3
        self.down_s = 2
        self.s = 1
        self.cha_2 = 128
        self.cha_3 = 256
        self.cha_4 = 256
        self.new_freq_dim = 16
        self.first_block = get_downsampling_block1(in_cha, self.cha_1, k = self.k, s = 1, p = (1, 2))
        self.second_block = get_downsampling_block1(self.cha_1, self.cha_2, k = self.k, s = 1, p = (1,2))
        self.third_block = get_downsampling_block1(self.cha_2, self.cha_3, k = self.k, s = 1, p = (1,2))
        self.forth_block = get_downsampling_block1(self.cha_3, self.cha_4, k = self.k, s = 1, p=None)
        self.fifth_block = get_downsampling_block2_1d(self.cha_4 * self.new_freq_dim, self.cha_4, k = 2, s = self.s, p=None)
        self.sixth_block = get_downsampling_block2_1d(self.cha_4, self.cha_4, k = 2, s = self.s)
        self.seventh_block = get_downsampling_block1_1d(self.cha_4, self.cha_4, k = 2, s = self.s)
        self.eight_block = get_downsampling_block2_1d(self.cha_4, self.cha_4, k = self.k, s = self.s)
        self.ninth_block = get_downsampling_block2_1d(self.cha_4, self.cha_4, k = self.k, s = self.s)
        self.tenth_block = get_downsampling_block2(self.cha_4, self.cha_4, k = self.k, s = self.s, p = None)
        self.upsampling = nn.Upsample(scale_factor=2, mode='nearest')
        self.fifth_block2 = get_downsampling_block2_1d(self.cha_4, self.cha_4, k = self.k, s = self.s, p = None)
        self.fifth_block3 = get_downsampling_block2_1d(self.cha_4, self.cha_4, k = self.k, s = self.s, p = None)
        self.fifth_block4 = get_downsampling_block2_1d(self.cha_4, self.cha_4, k = self.k, s = self.s, p = None)
        self.fifth_block5 = get_downsampling_block2_1d(self.cha_4, self.cha_4, k = self.k, s = self.s, p = None)
        self.bottleneck = nn.Conv1d(self.cha_4, out_cha, kernel_size=1, stride=1)
        self.dropout = nn.Dropout(0.2)
        
        
    def forward(self, mel): #mel -> (bs, in_cha = 4, time, freq)
        seq_len = mel.shape[2]
        out = self.first_block(mel)
        out = self.second_block(out)
        out = self.third_block(out)
        out = self.forth_block(out)
        out = out.reshape(out.shape[0], -1, seq_len, 1) #[bs, seq_len, 1, channels * f']
        out_4 = out.squeeze(-1) #[bs, channel * freq, seq_len] #[256, 4096, 12]
        out_5 = self.fifth_block(out_4) #[256, 256, 12]
        out_6 = self.sixth_block(out_5) #[256, 256, 6]
        out_7 = self.seventh_block(out_6) #[256, 256, 3]
        out_6 = out_6 + self.upsampling(out_7) #[256, 256, 6]
        out_5 = out_5 + self.upsampling(out_6) #[256, 256, 12]
        out_5 = self.fifth_block2(out_5)
        out_5 = self.fifth_block3(out_5)
        out_5 = self.fifth_block4(out_5)
        out_5 = self.fifth_block5(out_5) #[bs,seq_len, channels]
        output = self.bottleneck(out_5) #[bs, 21 * 3, seq_len]
        output = output.permute(0, 2, 1) #[bs, seq_len, 21 * 3]
        return output


