from logging import getLogger
from typing import List

logger = getLogger(__name__)

def get_mean() -> List[float]:
    # in a classroom
    mean = [
        -23.539913, 
        -25.119139, 
        -26.711174, 
        -22.011652, 
        0.0044820304, 
        0.00080580305, 
        -0.040803947
        ]
    

    logger.info(f"mean value: {mean}")
    return mean


def get_std() -> List[float]:
    # in a classroom
    std = [
        12.967929, 
        10.571223, 
        10.273731, 
        11.8457985, 
        0.022749536, 
        0.019968906, 
        0.037085325
        ]

    logger.info(f"std value: {std}")
    return std

def get_raw_mean() -> List[float]:
    mean = [
        -1.99718730e-05, -1.60138708e-08, -5.86972669e-08, -1.24919584e-07
    ]

    logger.info(f"mean value: {mean}")
    return mean


def get_raw_std() -> List[float]:
    std = [
        0.00657787, 0.00519175, 0.00472543, 0.00824625
    ]

    logger.info(f"std value: {std}")
    return std

def get_logmel_mean() -> List[float]:
    mean = [
        -23.539913, 
        -25.119139, 
        -26.711174, 
        -22.011652, 
        ]

    logger.info(f"mean value: {mean}")
    return mean


def get_logmel_std() -> List[float]:
    std = [
        12.967929, 
        10.571223, 
        10.273731, 
        11.8457985, 
        ]
    return std

def get_wo_norm_mean() -> List[float]:
    mean = [
        -16.566772,
        -13.443462,
        -18.72402,
        -15.498938,
        -17.582392,
        -17.18739,
        -15.774
    ]
    return mean

def get_wo_norm_std() -> List[float]:
    std = [
        15.025663, 
        12.787112, 
        17.318056, 
        14.863319, 
        16.7786, 
        15.965399, 
        14.254401
    ]
    return std

if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    import tqdm

    csv_path = "./csv/pose_regression_kfold_subject_1/train.csv"
    csv = pd.read_csv(csv_path)
    csv = csv[csv["sound_length"] == 2400]
    csv = csv[csv['preprocess'] == 'intensity'].reset_index()
    # mean = np.array(get_mean())
    # std = np.array(get_std())
    # data = [(np.load(path) - mean) / std for path in tqdm.tqdm(csv["sound_path"])]
    data = [np.load(path) for path in  tqdm.tqdm(csv["sound_path"])]
    data = np.array(data)
    data = data.reshape(-1, 7, 128)
    print("mean: [", end="")
    for i in range(7):
        if i:
            print(", ", end="")
        print(np.mean(data[:, :, i]), end="")
    print("]")
    print("std: [", end="")
    for i in range(7):
        if i:
            print(", ", end="")
        print(np.std(data[:, :, i]), end="")
    print("]")
