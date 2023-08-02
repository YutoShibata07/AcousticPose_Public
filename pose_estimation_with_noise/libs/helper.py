import os, time
from logging import getLogger
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# from sklearn.metrics import confusion_matrix, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.utils.data import DataLoader

from .meter import AverageMeter, ProgressMeter
from .metric import calc_accuracy, calc_rmse_mae_acc
from .loss_fn import mixup
from .joint_list import list2joint, joint2list

__all__ = ["train", "evaluate"]

logger = getLogger(__name__)


def do_one_iteration(
    sample: Dict[str, Any],
    model: nn.Module,
    criterion: Any,
    device: str,
    iter_type: str,
    optimizer: Optional[optim.Optimizer] = None,
    do_mixup: bool = False,
    smooth_loss: bool = False
) -> Tuple[int, float, float, np.ndarray, np.ndarray]:

    if iter_type not in ["train", "evaluate"]:
        message = "iter_type must be either 'train' or 'evaluate'."
        logger.error(message)
        raise ValueError(message)

    if iter_type == "train" and optimizer is None:
        message = "optimizer must be set during training."
        logger.error(message)
        raise ValueError(message)

    x = sample["sound"].to(device)
    t = sample["targets"].to(device)

    batch_size = x.shape[0]

    if (do_mixup) & (np.random.rand() < 0.5):
        mixed_x, y_a, y_b, lam = mixup.mixup_data(x, t)
        output = model(mixed_x)
        loss = mixup.mixup_criterion(criterion, output, y_a, y_b, lam)
    else:
        output = model(x)
        if smooth_loss:
            t_diff = t[:,:-1,:] - t[:,1:,:]
            output_diff = output[:,:-1,:] - output[:,1:,:]
            loss = criterion(output, t) + criterion(t_diff, output_diff)
        else:
            loss = criterion(output, t) 

    # measure accuracy and record loss
    # accs = calc_accuracy(output, t, topk=(1,))
    # acc1 = accs[0]

    if iter_type == "train" and optimizer is not None:
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # keep predicted results and gts for calculate F1 Score
    gt = t.to("cpu").numpy()
    pred = output.to("cpu").detach().numpy()
    
    gt = gt.reshape(-1, gt.shape[-1])
    pred = pred.reshape(-1, pred.shape[-1])

    # return batch_size, loss.item(), acc1, gt, pred
    return batch_size, loss.item(), gt, pred


def train(
    loader: DataLoader,
    model: nn.Module,
    criterion: Any,
    optimizer: optim.Optimizer,
    epoch: int,
    device: str,
    interval_of_progress: int = 50,
    aug:str = '',
    smooth_loss:bool = False
) -> Tuple[float, float, float]:

    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    # top1 = AverageMeter("Acc@1", ":6.2f")

    progress = ProgressMeter(
        len(loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch),
    )

    # keep predicted results and gts for calculate F1 Score
    gts = []
    preds = []

    # switch to train mode
    model.train()

    end = time.time()
    for i, sample in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # batch_size, loss, acc1, gt, pred = do_one_iteration(
        #     sample, model, criterion, device, "train", optimizer, do_mixup=True
        # )
        if aug == 'mixup':
            batch_size, loss, gt, pred = do_one_iteration(
                sample, model, criterion, device, "train", optimizer, do_mixup=True
            )
        else:
            batch_size, loss, gt, pred = do_one_iteration(
                sample, model, criterion, device, "train", optimizer, do_mixup=False, smooth_loss=smooth_loss
            )

        losses.update(loss, batch_size)
        # top1.update(acc1, batch_size)

        # save the ground truths and predictions in lists
        gts += list(gt)
        preds += list(pred)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # show progress bar per 50 iteration
        if i != 0 and i % interval_of_progress == 0:
            progress.display(i)

    # calculate F1 Score
    # f1s = f1_score(gts, preds, average="macro")
    rmse, mae, acc = calc_rmse_mae_acc(gts, preds)

    # return losses.get_average(), top1.get_average(), f1s
    return losses.get_average(), rmse, mae, acc


def evaluate(
    loader: DataLoader,
    model: nn.Module,
    criterion: Any,
    device: str,
    mode: str = "train",
    image_dir: str = None,
    output_type:str = 'both',
) -> Tuple[float, float, float, np.ndarray]:
    losses = AverageMeter("Loss", ":.4e")
    # top1 = AverageMeter("Acc@1", ":6.2f")

    # keep predicted results and gts for calculate F1 Score
    gts = []
    preds = []

    # calculate confusion matrix
    # n_classes = loader.dataset.get_n_classes()
    # c_matrix = np.zeros((n_classes, n_classes), dtype=np.int32)

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for sample in loader:
            # batch_size, loss, acc1, gt, pred = do_one_iteration(
            #     sample, model, criterion, device, "evaluate", do_mixup=False
            # )
            batch_size, loss, gt, pred = do_one_iteration(
                sample, model, criterion, device, "evaluate", do_mixup=False
            )

            losses.update(loss, batch_size)
            # top1.update(acc1, batch_size)

            # keep predicted results and gts for calculate F1 Score
            gts += list(gt)
            preds += list(pred)

            # c_matrix += confusion_matrix(
            #     gt,
            #     pred,
            #     labels=[i for i in range(n_classes)],
            # )

    # f1s = f1_score(gts, preds, average="macro")

    rmse, mae, acc = calc_rmse_mae_acc(gts, preds, mode=mode, image_dir=image_dir, output_type=output_type)
    # return losses.get_average(), top1.get_average(), f1s, c_matrix
    return losses.get_average(), rmse, mae, acc
