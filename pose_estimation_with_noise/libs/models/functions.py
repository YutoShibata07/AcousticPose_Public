import torch.nn as nn
import torchvision
import torch.nn.functional as F
from .wisppn_resnet import get_wisppn
import torch


def get_downsampling_block1(in_cha = 4, out_cha = 4, k = 2, s = 2, p = 2):
    if p != None:
        model = nn.Sequential(
            nn.Conv2d(in_cha, out_cha, k, s, padding='same'),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_cha),
            nn.Conv2d(out_cha, out_cha, k, s, padding='same'),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_cha),
            nn.MaxPool2d(p, stride=p),
        )
    else:
        model = nn.Sequential(
            nn.Conv2d(in_cha, out_cha, k, s, padding='same'),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_cha),
            nn.Conv2d(out_cha, out_cha, k, s, padding='same'),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_cha),
        )
    return model
def get_downsampling_block2(in_cha = 4, out_cha = 4, k = 2, s = 2, p = 2):
    if p != None:
        model = nn.Sequential(
            nn.Conv2d(in_cha, out_cha, k, s, padding='same'),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_cha),
            nn.MaxPool2d(2, stride=2),
        )
    else:
        model = nn.Sequential(
            nn.Conv2d(in_cha, out_cha, k, s, padding='same'),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_cha),
        )
    return model

def get_downsampling_block1_1d(in_cha = 4, out_cha = 4, k = 2, s = 2, p = 2):
    if p != None:
        model = nn.Sequential(
            nn.Conv1d(in_cha, out_cha, k, s, padding='same'),
            nn.LeakyReLU(),
            nn.BatchNorm1d(out_cha),
            nn.Conv1d(out_cha, out_cha, k, s, padding='same'),
            nn.LeakyReLU(),
            nn.BatchNorm1d(out_cha),
            nn.MaxPool1d(p, stride=p),
        )
    else:
        model = nn.Sequential(
            nn.Conv1d(in_cha, out_cha, k, s, padding='same'),
            nn.LeakyReLU(),
            nn.BatchNorm1d(out_cha),
            nn.Conv1d(out_cha, out_cha, k, s, padding='same'),
            nn.LeakyReLU(),
            nn.BatchNorm1d(out_cha),
        )
    return model
def get_downsampling_block2_1d(in_cha = 4, out_cha = 4, k = 2, s = 2, p = 2):
    if p != None:
        model = nn.Sequential(
            nn.Conv1d(in_cha, out_cha, k, s, padding='same'),
            nn.LeakyReLU(),
            nn.BatchNorm1d(out_cha),
            nn.MaxPool1d(2, stride=2),
        )
    else:
        model = nn.Sequential(
            nn.Conv1d(in_cha, out_cha, k, s, padding='same'),
            nn.LeakyReLU(),
            nn.BatchNorm1d(out_cha),
        )
    return model

