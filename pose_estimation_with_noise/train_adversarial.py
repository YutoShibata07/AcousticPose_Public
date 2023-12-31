import argparse
import datetime
import os
import time
from logging import DEBUG, INFO, basicConfig, getLogger

import torch
import torch.optim as optim
import wandb
from torchvision.transforms import (
    ColorJitter,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    ToTensor,
)

from libs.checkpoint import resume, save_checkpoint
from libs.class_id_map import get_cls2id_map
from libs.config import get_config
from libs.dataset import get_dataloader
from libs.device import get_device
from libs.helper_adversarial import evaluate, train
from libs.logger_ad import TrainLogger
from libs.loss_fn import get_criterion
from libs.models import get_model
from libs.seed import set_seed

logger = getLogger(__name__)


def get_arguments() -> argparse.Namespace:
    """parse all the arguments from command line inteface return a list of
    parsed arguments."""

    parser = argparse.ArgumentParser(
        description="""
        train a network for sound pose estimation with Sound Pose Dataset.
        """
    )
    parser.add_argument("config", type=str, help="path of a config file")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Add --resume option if you start training from checkpoint.",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Add --use_wandb option if you want to use wandb.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Add --debug option if you want to see debug-level logs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed",
    )

    return parser.parse_args()


def main() -> None:
    args = get_arguments()

    # save log files in the directory which contains config file.
    result_path = os.path.dirname(args.config)
    experiment_name = os.path.basename(result_path)

    # setting logger configuration
    logname = os.path.join(result_path, f"{datetime.datetime.now():%Y-%m-%d}_train.log")
    basicConfig(
        level=DEBUG if args.debug else INFO,
        format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=logname,
    )

    # fix seed
    set_seed()
    # configuration
    config = get_config(args.config)

    # cpu or cuda
    device = get_device(allow_only_gpu=False)

    train_loader = get_dataloader(
        config.dataset_name,
        config.sound_length,
        config.input_feature,
        "train",
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = get_dataloader(
        config.dataset_name,
        config.sound_length,
        config.input_feature,
        "val",
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    # the number of classes
    n_classes = len(get_cls2id_map())

    # define a model
    model = get_model(
        config.model,
        n_classes,
        config.input_feature,
        pretrained=config.pretrained,
    )

    # send the model to cuda/cpu
    model['model'].to(device)
    model['D'].to(device)
    optimizer = dict()
    optimizer['model'] = optim.Adam(model['model'].parameters(), lr=config.learning_rate)
    optimizer['D'] = optim.Adam(model['D'].parameters(), lr=0.001)
    # keep training and validation log
    begin_epoch = 0
    best_rmse = float("inf")

    # resume if you want
    if args.resume:
        resume_path = os.path.join(result_path, "checkpoint.pth")
        begin_epoch, model, optimizer, best_rmse = resume(resume_path, model, optimizer)

    log_path = os.path.join(result_path, "log.csv")
    train_logger = TrainLogger(log_path, resume=args.resume)

    # criterion for loss
    criterion = get_criterion(config.ratio)

    # Weights and biases
    if args.use_wandb:
        wandb.init(
            name=experiment_name,
            config=config,
            project="sound_pose_estimation",
            job_type="training",
            # dirs="./wandb_result/",
        )
        # Magic
        wandb.watch(model['model'], log="all")

    # train and validate model
    logger.info("Start training.")

    for epoch in range(begin_epoch, config.max_epoch):
        # training
        start = time.time()
        # train_loss, train_acc1, train_f1s = train(
        #     train_loader, model, criterion, optimizer, epoch, device
        # )
        train_loss, train_d_loss, train_rmse, train_mae, train_acc, train_testee_acc = train(
            train_loader, model, criterion, optimizer, epoch, device, gan=config.gan, smooth_loss=config.smooth_loss
        )
        train_time = int(time.time() - start)

        # validation
        start = time.time()
        # val_loss, val_acc1, val_f1s, c_matrix = evaluate(
        #     val_loader, model, criterion, device
        # )
        val_loss, val_d_loss, val_rmse, val_mae, val_acc, val_testee_acc = evaluate(
            val_loader, model, criterion, device
        )
        val_time = int(time.time() - start)

        # save a model if top1 acc is higher than ever
        if best_rmse > val_rmse["all"]:
            best_rmse = val_rmse["all"]
            torch.save(
                model['model'].state_dict(),
                os.path.join(result_path, "best_model.prm"),
            )
            torch.save(
                model['D'].state_dict(),
                os.path.join(result_path, "best_D_model.prm"),
            )

        # save checkpoint every epoch
        save_checkpoint(result_path, epoch, model['model'], optimizer['model'], best_rmse)

        # write logs to dataframe and csv file
        train_logger.update(
            epoch,
            optimizer['model'].param_groups[0]["lr"],
            train_time,
            train_loss,
            train_d_loss,
            train_rmse,
            train_mae,
            train_acc,
            train_testee_acc,
            val_time,
            val_loss,
            val_d_loss,
            val_rmse,
            val_mae,
            val_acc,
            val_testee_acc,
        )

        # save logs to wandb
        if args.use_wandb:
            wandb.log(
                {
                    "lr": optimizer['model'].param_groups[0]["lr"],
                    "train_time[sec]": train_time,
                    "train_loss_m": train_loss,
                    "train_d_loss": train_d_loss,
                    "train_rmse": train_rmse,
                    "train_mae": train_mae,
                    "train_acc": train_acc,
                    "train_testee_acc":train_testee_acc,
                    "val_time[sec]": val_time,
                    "val_loss_m": val_loss,
                    "val_d_loss":val_d_loss,
                    "val_rmse": val_rmse,
                    "val_mae": val_mae,
                    "val_acc": val_acc,
                    "val_testee_acc":val_testee_acc,
                },
                step=epoch,
            )

    # save models
    torch.save(model['model'].state_dict(), os.path.join(result_path, "final_model.prm"))
    torch.save(model['D'].state_dict(), os.path.join(result_path, "final_D_model.prm"))
    # delete checkpoint
    os.remove(os.path.join(result_path, "checkpoint.pth"))

    logger.info("Done")


if __name__ == "__main__":
    main()