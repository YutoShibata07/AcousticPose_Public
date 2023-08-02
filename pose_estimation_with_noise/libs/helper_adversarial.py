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
    model: Any,
    criterion: Any,
    device: str,
    iter_type: str,
    optimizer: Optional[optim.Optimizer] = None,
    do_mixup: bool = False,
    epoch: int = 8,
    gan:str = 'normal',
    smooth_loss:bool = False
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
    testee = sample['testee'].to(device)

    batch_size = x.shape[0]

    if (do_mixup) & (np.random.rand() < 0.5):
        mixed_x, y_a, y_b, lam = mixup.mixup_data(x, t)
        output = model(mixed_x)
        loss = mixup.mixup_criterion(criterion, output, y_a, y_b, lam)
    elif iter_type == "train" and optimizer is not None:
        output, d_in = model["model"](x)
        d_output = model["D"](d_in)
        # d_output = d_output.reshape(-1, d_output.shape[-1])
        # testee = testee.reshape(-1,)
        # testee = testee.long()
        d_loss = criterion["D"](d_output, testee)

        optimizer["model"].zero_grad()
        optimizer["D"].zero_grad()
        d_loss.backward()
        # optimizer["model"].step()
        # Discriminatorの学習 ー体格差を見分けるモデルー
        optimizer['D'].step()
        
        output, d_in = model["model"](x)
        d_output = model["D"](d_in)
        # d_output = d_output.reshape(-1, d_output.shape[-1])
        std_ = torch.std(nn.Softmax(dim=-1)(d_output), dim = -1)
        std_ = torch.mean(std_)
        # 体格差が検出可能=Discriminatorのロスが小さい場合はパラメタの更新を大きくして個人差を無くさせる
        if epoch < 35:
            if gan == 'std':
                m_loss = criterion["model"](output, t) + criterion["ratio"] * std_
            elif gan == 'normal':
                m_loss = criterion["model"](output, t) - criterion["D"](d_output, testee) * criterion['ratio'] #+ criterion["ratio"] *  std_
            else:
                m_loss = criterion["model"](output, t)
        else:
            m_loss = criterion["model"](output, t)
        if smooth_loss == True:
            t_diff = t[:,:-1,:] - t[:,1:,:]
            output_diff = output[:,:-1,:] - output[:,1:,:]
            m_loss = m_loss + criterion["model"](t_diff, output_diff)
        optimizer["model"].zero_grad()
        optimizer["D"].zero_grad()
        m_loss.backward()
        optimizer["model"].step()
        loss = {
            "model": m_loss.item(),
            "D": d_loss.item(),
        }
        gt = t.to("cpu").numpy()
        pred = output.to("cpu").detach().numpy()
        gt = gt.reshape(-1, gt.shape[-1])
        pred = pred.reshape(-1, pred.shape[-1])
        d_pred = d_output.to('cpu').detach().numpy()
        testee = testee.to('cpu').detach().numpy()
        return batch_size, loss, gt, pred, testee, d_pred
    else:
        output, d_in = model['model'](x)
        loss = criterion['model'](output, t)
        d_output = model["D"](d_in)
        d_output = d_output.reshape(-1, d_output.shape[-1])
        testee = testee.reshape(-1,)
        testee = testee.long()
        d_loss = criterion["D"](d_output, testee)
        loss = {
            "model": loss.item(),
            "D": d_loss.item(),
        }
        gt = t.to("cpu").numpy()
        pred = output.to("cpu").detach().numpy()
        gt = gt.reshape(-1, gt.shape[-1])
        pred = pred.reshape(-1, pred.shape[-1])
        d_pred = d_output.to('cpu').detach().numpy()
        testee = testee.to('cpu').detach().numpy()
        d_pred = d_pred.reshape(-1, d_pred.shape[-1])
        testee = testee.reshape(-1, testee.shape[-1])
        return batch_size, loss, gt, pred, testee, d_pred
    # measure accuracy and record loss
    # accs = calc_accuracy(output, t, topk=(1,))
    # acc1 = accs[0]
    # keep predicted results and gts for calculate F1 Score
    gt = t.to("cpu").numpy()
    pred = output.to("cpu").detach().numpy()

    # return batch_size, loss.item(), acc1, gt, pred
    return batch_size, loss, gt, pred


def train(
    loader: DataLoader,
    model: nn.Module,
    criterion: Any,
    optimizer: optim.Optimizer,
    epoch: int,
    device: str,
    interval_of_progress: int = 50,
    gan:str = 'normal',
    smooth_loss:bool = False
) -> Tuple[float, float, float]:

    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    d_losses = AverageMeter('D_Loss', ":.4e")
    d_acc = AverageMeter('D_ACC', ":.4e")
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
    model['model'].train()
    model['D'].train()

    end = time.time()
    for i, sample in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # batch_size, loss, acc1, gt, pred = do_one_iteration(
        #     sample, model, criterion, device, "train", optimizer, do_mixup=True
        # )
        batch_size, loss, gt, pred, testee, d_pred = do_one_iteration(
            sample, model, criterion, device, "train", optimizer, epoch = epoch, gan = gan, smooth_loss = smooth_loss
        )
        acc = (testee == d_pred.argmax(-1)).mean()
        losses.update(loss['model'], batch_size)
        d_losses.update(loss['D'], batch_size)
        d_acc.update(acc, batch_size)
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
    return losses.get_average(), d_losses.get_average(), rmse, mae, acc, d_acc.get_average()


def evaluate(
    loader: DataLoader,
    model: Dict,
    criterion: Dict,
    device: str,
    mode: str = "train",
    image_dir: str = None,
    output_type = 'both',
) -> Tuple[float, float, float, np.ndarray]:
    losses = AverageMeter("Loss", ":.4e")
    d_losses = AverageMeter('D_Loss', ":.4e")
    d_acc = AverageMeter('D_ACC', ":.4e")
    # top1 = AverageMeter("Acc@1", ":6.2f")

    # keep predicted results and gts for calculate F1 Score
    gts = []
    preds = []

    # calculate confusion matrix
    # n_classes = loader.dataset.get_n_classes()
    # c_matrix = np.zeros((n_classes, n_classes), dtype=np.int32)

    # switch to evaluate mode
    model['model'].eval()
    model['D'].eval()

    with torch.no_grad():
        for sample in loader:
            # batch_size, loss, acc1, gt, pred = do_one_iteration(
            #     sample, model, criterion, device, "evaluate", do_mixup=False
            # )
            batch_size, loss, gt, pred, testee, d_pred = do_one_iteration(
                sample, model, criterion, device, "evaluate", do_mixup=False
            )

            acc = (testee == d_pred.argmax(-1)).mean()
            losses.update(loss['model'], batch_size)
            d_losses.update(loss['D'], batch_size)
            d_acc.update(acc, batch_size)
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
    return losses.get_average(), d_losses.get_average(), rmse, mae, acc, d_acc.get_average()