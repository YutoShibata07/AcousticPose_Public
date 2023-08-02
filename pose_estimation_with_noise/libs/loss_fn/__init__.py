from logging import getLogger
from typing import Optional

import torch.nn as nn

# from ..dataset_csv import DATASET_CSVS
# from .class_weight import get_class_weight

__all__ = ["get_criterion"]
logger = getLogger(__name__)


def get_criterion(
    adver_ratio:float = 0
) -> nn.Module:
    if adver_ratio > 0:
        criterion = dict()
        criterion['model'] = nn.MSELoss()
        criterion['D'] = nn.CrossEntropyLoss()
        criterion['ratio'] = adver_ratio
    else:
        criterion = nn.MSELoss()
    return criterion
