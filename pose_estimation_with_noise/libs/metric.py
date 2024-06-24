from typing import Any, Dict, List, Tuple

import torch
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_squared_error, mean_absolute_error

from libs.joint_list import (
    get_joint_names,
    get_leg,
    get_arm,
    get_body,
    joint2list,
    list2joint,
)


def calc_accuracy(
    output: torch.Tensor, target: torch.Tensor, topk: Tuple[int] = (1,)
) -> List[float]:
    """Computes the accuracy over the k top predictions.

    Args:
        output: (N, C). model output.
        target: (N, C). ground truth.
        topk: if you set (1, 5), top 1 and top 5 accuracy are calcuated.
    Return:
        res: List of calculated top k accuracy
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1)
            correct_k = correct_k.float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res


def make_pose_image(
    idx: int,
    joints: Dict[str, Any],
    sub_joints: Dict[str, Any] = None,
    output_type = 'both',
) -> None:
    # joints = sub_joints
    # sub_joints = None

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    lim = 10
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(0, 2 * lim)
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    ax.axes.zaxis.set_ticklabels([])

    for lr in ["Left", "Right"]:
        for parts in [get_leg(lr), get_arm(lr)]:
            points = {
                "_x": [],
                "_y": [],
                "_z": [],
            }
            for name in parts:
                for dim in ["_x", "_y", "_z"]:
                    points[dim].append(joints[name + dim])
            if (output_type == 'both') | (output_type=='gt'):
                ax.plot(points["_x"], -np.array(points["_z"]), points["_y"], color="red")
    points = {
        "_x": [],
        "_y": [],
        "_z": [],
    }
    for name in get_body():
        for dim in ["_x", "_y", "_z"]:
            points[dim].append(joints[name + dim])
    if (output_type == 'both') | (output_type == 'gt'):
        ax.plot(
            points["_x"],
            -np.array(points["_z"]),
            points["_y"],
            color="red",
            label="Ground Truth",
        )
    if not sub_joints is None:
        for lr in ["Left", "Right"]:
            for parts in [get_leg(lr), get_arm(lr)]:
                points = {
                    "_x": [],
                    "_y": [],
                    "_z": [],
                }
                for name in parts:
                    for dim in ["_x", "_y", "_z"]:
                        points[dim].append(sub_joints[name + dim])
                if (output_type == 'both') | (output_type == 'prediction'):
                    ax.plot(
                        points["_x"], -np.array(points["_z"]), points["_y"], color="blue"
                    )
        points = {
            "_x": [],
            "_y": [],
            "_z": [],
        }
        for name in get_body():
            for dim in ["_x", "_y", "_z"]:
                points[dim].append(sub_joints[name + dim])
        if (output_type == 'both') | (output_type == 'prediction'):
            ax.plot(
                points["_x"],
                -np.array(points["_z"]),
                points["_y"],
                color="blue",
                label="Predict",
            )
        if output_type == 'both':
            plt.legend(fontsize=9)

    fig.canvas.draw()
    data = fig.canvas.tostring_rgb()
    w, h = fig.canvas.get_width_height()
    c = len(data) // (w * h)
    plt.close()

    img = np.frombuffer(data, dtype=np.uint8).reshape(h, w, c)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return {"idx": idx, "img": img}


def make_time_score(
    gt_lists: Dict[str, List[Any]],
    pred_lists: Dict[str, List[Any]],
    mode: str,
    image_dir: str,
) -> None:

    for part in ["all", "arm", "leg", "body"]:

        rmse = []
        mae = []

        for gt, pred in zip(gt_lists[part], pred_lists[part]):
            rmse.append(np.sqrt(mean_squared_error(gt, pred)))
            mae.append(mean_absolute_error(gt, pred))

        fig = plt.figure(figsize=(10, 10))

        ax1 = fig.add_subplot(2, 1, 1)
        ax1.scatter(range(len(rmse)), rmse)
        ax1.set_ylabel("rmse (cm)")
        ax1.set_ylim(0, 5)

        ax2 = fig.add_subplot(2, 1, 2)
        ax2.scatter(range(len(mae)), mae)
        ax2.set_xlabel("time (-)")
        ax2.set_ylabel("mae (cm)")
        ax2.set_ylim(0, 5)

        path = os.path.join(image_dir, "%s_%s.png" % (mode, part))
        fig.savefig(path)
        plt.close()


def calc_dis(
    gt_dict: Dict[str, float], pred_dict: Dict[str, float], joint: str
) -> float:
    dis = 0.0
    for dim in ["_x", "_y", "_z"]:
        dis += (gt_dict[joint + dim] - pred_dict[joint + dim]) ** 2
    return np.sqrt(dis)


def calc_acc(corrects: Dict[str, List[float]], part: str) -> float:

    acc = []

    for i in range(4):
        if part == "all":
            ac = np.mean(
                np.array(
                    [np.mean(corrects[joint][:, i]) for joint in get_joint_names()]
                )
            )
        elif part == "arm":
            ac = np.mean(
                np.array(
                    [
                        np.mean(corrects[joint][:, i])
                        for joint in get_arm("Right") + get_arm("Left")
                    ]
                )
            )
        elif part == "leg":
            ac = np.mean(
                np.array(
                    [
                        np.mean(corrects[joint][:, i])
                        for joint in get_leg("Right") + get_leg("Left")
                    ]
                )
            )
        elif part == "body":
            ac = np.mean(
                np.array([np.mean(corrects[joint][:, i]) for joint in get_body()])
            )

        acc.append(ac)

    return acc


def gt_pred_process(idx, gt, pred):
    gt_dict = list2joint(gt)
    pred_dict = list2joint(pred)

    out = {
        "idx": idx,
        "gt_dicts": gt_dict,
        "pred_dicts": pred_dict,
        "corrects": {},
        "gt_lists": {},
        "pred_lists": {},
    }

    threshold = 0.0
    for dim in ["_x", "_y", "_z"]:
        threshold += (gt_dict["Neck" + dim] - gt_dict["Head" + dim]) ** 2
    threshold = np.sqrt(threshold) * 0.5
    for joint in get_joint_names():
        out["corrects"][joint] = []
        for i in range(1, 5):
            cor = float(calc_dis(gt_dict, pred_dict, joint) < (threshold * i))
            out["corrects"][joint].append(cor)

    for part in ["all", "arm", "leg", "body"]:
        gt_list = joint2list(gt_dict, part)
        pred_list = joint2list(pred_dict, part)
        out["gt_lists"][part] = gt_list
        out["pred_lists"][part] = pred_list

    return out


def calc_rmse_mae_acc(gts, preds, mode=None, image_dir: str = None, output_type:str = 'both'):

    lists = Parallel(n_jobs=24)(
        [delayed(gt_pred_process)(i, gts[i], preds[i]) for i in range(len(gts))]
    )
    lists = sorted(lists, key=lambda x: x["idx"])
    # lists = [gt_pred_process(i, gts[i], preds[i]) for i in range(len(gts))]
    corrects = {
        name: [data["corrects"][name] for data in lists] for name in get_joint_names()
    }
    corrects = {k: np.array(v) for k, v in corrects.items()}
    gt_lists = {}
    pred_lists = {}
    for part in ["all", "arm", "leg", "body"]:
        gt_lists[part] = [data["gt_lists"][part] for data in lists]
        pred_lists[part] = [data["pred_lists"][part] for data in lists]

    if not image_dir is None:
        fps = 20
        w = 500
        h = 500
        codec = cv2.VideoWriter_fourcc(*"mp4v")
        video = cv2.VideoWriter(
            os.path.join(image_dir, mode + ".mp4"), codec, fps, (w, h)
        )

        imgs = Parallel(n_jobs=16)(
            [
                delayed(make_pose_image)(
                    data["idx"], data["gt_dicts"], data["pred_dicts"], output_type = output_type
                )
                for data in lists
            ]
        )
        imgs = sorted(imgs, key=lambda x: x["idx"])

        for img in imgs:
            video.write(img["img"])
        video.release()

    if not image_dir is None:
        make_time_score(gt_lists, pred_lists, mode, image_dir)

    rmse = {}
    mae = {}
    acc = {}

    for part in ["all", "arm", "leg", "body"]:
        rmse[part] = np.sqrt(
            mean_squared_error(
                np.concatenate(gt_lists[part]), np.concatenate(pred_lists[part])
            )
        )
        mae[part] = mean_absolute_error(
            np.concatenate(gt_lists[part]), np.concatenate(pred_lists[part])
        )
        acc[part] = calc_acc(corrects, part)

    return rmse, mae, acc
