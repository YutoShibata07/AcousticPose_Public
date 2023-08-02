import argparse
import csv
import datetime
import os
from logging import DEBUG, INFO, basicConfig, getLogger
import numpy as np

import pandas as pd
import torch
from torchvision.transforms import Compose, Normalize, ToTensor

from libs.config import get_config
from libs.dataset import get_dataloader
from libs.device import get_device
from libs.dataset_csv import DATASET_CSVS
from libs.joint_list import joint2list
from libs.mean_std import get_mean, get_std
from libs.metric import calc_rmse_mae_acc


logger = getLogger(__name__)


def get_arguments() -> argparse.Namespace:
    """parse all the arguments from command line inteface return a list of
    parsed arguments."""

    parser = argparse.ArgumentParser(
        description="""
                    train a network for sound pose estimation
                    with Sound Pose Dataset
                    """
    )
    parser.add_argument("config", type=str, help="path of a config file")
    parser.add_argument("mode", type=str, help="validation or test")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="""path to the trained model. If you do not specify, the trained model,
            'best_acc1_model.prm' in result directory will be used.""",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Add --debug option if you want to see debug-level logs.",
    )

    return parser.parse_args()


def main() -> None:
    args = get_arguments()

    # configuration
    config = get_config(args.config)
    result_path = os.path.dirname(args.config)
    image_dir = os.path.join(result_path, "images")
    os.makedirs(image_dir, exist_ok=True)

    if args.mode not in ["validation", "test"]:
        message = "args.mode is invalid. ['validation', 'test']"
        logger.error(message)
        raise ValueError(message)

    # setting logger configuration
    logname = os.path.join(
        result_path, f"{datetime.datetime.now():%Y-%m-%d}_{args.mode}.log"
    )
    basicConfig(
        level=DEBUG if args.debug else INFO,
        format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=logname,
    )

    # get mean position
    csv_file = getattr(
        DATASET_CSVS[config.dataset_name],
        "val" if args.mode == "validation" else "test",
    )
    data = pd.read_csv(csv_file).mean()
    data = torch.Tensor(joint2list(data))

    transform = Compose([ToTensor(), Normalize(mean=get_mean(), std=get_std())])

    loader = get_dataloader(
        config.dataset_name,
        config.sound_length,
        "val" if args.mode == "validation" else "test",
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    gts = []
    preds = []

    for i, sample in enumerate(loader):
        gt = sample["targets"]
        pred = [data]

        gts += list(gt)
        preds += list(pred)

    rmse, mae, acc = calc_rmse_mae_acc(gts, preds)

    # train and validate model
    logger.info(
        f"---------- Start evaluation for {args.mode} data with mean evaluation ----------"
    )
    logger.info(
        "RMSE: {:.2f}\tMAE: {:.2f}\tAcc: {:.2f}".format(
            rmse["all"], mae["all"], acc["all"]
        )
    )
    logger.info(
        "arm RMSE: {:.2f}\tarm MAE: {:.2f}\tAcc: {:.2f}".format(
            rmse["arm"], mae["arm"], acc["arm"]
        )
    )
    logger.info(
        "leg RMSE: {:.2f}\tleg MAE: {:.2f}\tAcc: {:.2f}".format(
            rmse["leg"], mae["leg"], acc["leg"]
        )
    )
    logger.info(
        "body RMSE: {:.2f}\tbody MAE: {:.2f}\tAcc: {:.2f}".format(
            rmse["body"], mae["body"], acc["body"]
        )
    )

    df = pd.DataFrame(
        {"rmse": [rmse], "mae": [mae], "acc": [acc]},
        columns=["rmse", "mae", "acc"],
        index=None,
    )

    df.to_csv(os.path.join(result_path, "{}_log.csv").format(args.mode), index=False)

    logger.info("Done.")


if __name__ == "__main__":
    main()
