from logging import getLogger

import torch.nn as nn
import torchvision
import torch.nn.functional as F
from .wisppn_resnet import get_wisppn
import torch
from .speech2pose import Speech2pose
from .speech2pose_ad import Speech2pose_P, Speech2Pose_D
from .wipose import Wipose_LSTM

__all__ = ["get_model"]

model_names = [
    "speech2pose",
    "wipose_lstm",
    "wisppn",
    "speech2pose_ad",
]
logger = getLogger(__name__)


def get_model(
    name: str, n_classes: int, input_feature: str, pretrained: bool = True
) -> nn.Module:
    name = name.lower()
    if name not in model_names:
        message = (
            "There is no model appropriate to your choice. "
            "You have to choose %s as a model." % (", ").join(model_names)
        )
        logger.error(message)
        raise ValueError(message)

    logger.info("{} will be used as a model.".format(name))

    try:
        model = getattr(torchvision.models, name)(pretrained=pretrained)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features=in_features, out_features=n_classes, bias=True)
    except:
        in_cha = 7
        if input_feature == "logmel":
            in_cha = 4
        elif input_feature == "intensity":
            in_cha = 3
        elif input_feature == 'raw':
            in_cha = 4
        out_cha = 21 * 3
        if name == "wisppn":
            model = get_wisppn(in_cha=in_cha, out_cha=21 * 3)
        elif name == 'speech2pose':
            model = Speech2pose(in_cha=in_cha, out_cha=21 * 3)
        elif name == 'wipose_lstm':
            model = Wipose_LSTM(in_cha = in_cha, out_cha = out_cha)
        elif name == 'speech2pose_ad':
            model = dict()
            model['model'] = Speech2pose_P(in_cha = in_cha, out_cha= 21 * 3)
            model['D'] = Speech2Pose_D(in_cha = 256, out_cha = 8)
        else:
            message = (
                "There is no model appropriate to your choice. "
                "You have to choose %s as a model." % (", ").join(model_names)
            )
            logger.error(message)
            raise ValueError(message)

    return model

