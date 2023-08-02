import argparse
import glob
import os
import sys
from typing import Dict, List, Tuple, Union

import pandas as pd
from pandas.core.frame import DataFrame

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from libs.class_id_map import get_cls2id_map
from libs.dataset_csv import DATASET_CSVS
from libs.joint_list import get_joints


def get_arguments() -> argparse.Namespace:
    """parse all the arguments from command line inteface return a list of
    parsed arguments."""

    parser = argparse.ArgumentParser(
        description="make csv files for Sound Pose dataset"
    )
    parser.add_argument(
        "--sound_csv_path",
        type=str,
        default="./dataset/sound.csv",
        help="path to class label csv",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="action_segmentation",
        help="a directory where csv files will be saved",
    )
    parser.add_argument(
        "--subject_name",
        type=str,
        default="kimura",
        help="a directory where csv files will be saved",
    )
    parser.add_argument(
        "--preprocess",
        type=str,
        default="raw",
        help="a directory where csv files will be saved",
    )

    return parser.parse_args()


def split_data(
    data: Dict[str, Dict[str, List[Union[int, str]]]],
    data_paths: List[str],
    sound_length_list: List[int],
    cls_name: str,
    cls_id: int,
) -> None:

    elements = list(zip(data_paths, sound_length_list))
    for i, (path, sound_length) in enumerate(elements):
        if i % 5 == 4:
            # for test
            data["test"]["sound_path"].append(path)
            data["test"]["label"].append(cls_name)
            data["test"]["class_id"].append(cls_id)
            data["test"]["sound_length"].append(sound_length)
        elif i % 5 == 3:
            # for validation
            data["val"]["sound_path"].append(path)
            data["val"]["label"].append(cls_name)
            data["val"]["class_id"].append(cls_id)
            data["val"]["sound_length"].append(sound_length)
        else:
            # for training
            data["train"]["sound_path"].append(path)
            data["train"]["label"].append(cls_name)
            data["train"]["class_id"].append(cls_id)
            data["train"]["sound_length"].append(sound_length)


def split_posedata(sound_data) -> Tuple[DataFrame]:
    columns = ["sound_path", "testee", "sound_length"] + get_joints()
    csv_data = {
        "train": {name: [] for name in columns},
        "val": {name: [] for name in columns},
        "test": {name: [] for name in columns},
    }

    data = sound_data[columns]
    data = data.set_index("sound_path").reset_index()

    for sound_length in data["sound_length"].unique():
        _data = data[data["sound_length"] == sound_length].reset_index()
        sec_frame = 48000 // sound_length
        for idx in range(len(_data)):
            for column in columns:
                sec = idx // sec_frame
                if sec % 3 == 0:
                    csv_data["test"][column].append(_data[column][idx])
                elif sec % 3 == 1:
                    csv_data["val"][column].append(_data[column][idx])
                else:
                    csv_data["train"][column].append(_data[column][idx])

    train_df = pd.DataFrame(
        csv_data["train"],
        columns=columns,
    )

    val_df = pd.DataFrame(
        csv_data["val"],
        columns=columns,
    )

    test_df = pd.DataFrame(
        csv_data["test"],
        columns=columns,
    )

    return train_df, val_df, test_df

def split_posedata_timeseries_name(sound_data, name:str) -> Tuple[DataFrame]:
    columns = ["sound_path", "testee", "sound_length"] + get_joints()
    csv_data = {
        "train": {name: [] for name in columns},
        "val": {name: [] for name in columns},
        "test": {name: [] for name in columns},
    }

    data = sound_data[columns]
    data = data[data.testee == name].reset_index(drop=True)
    data = data.set_index("sound_path").reset_index()

    for sound_length in data["sound_length"].unique():
        _data = data[data["sound_length"] == sound_length].reset_index()
        testee_list = _data['testee'].unique()
        for testee in testee_list:
            testee_tmp = _data[_data.testee == testee].reset_index(drop=True)
            print(testee,':',testee_tmp.shape[0])
            test_start = (testee_tmp.shape[0]//5 ) * 4
            for idx in range(testee_tmp.shape[0]):
                for column in columns:
                    if idx < test_start:
                        csv_data["train"][column].append(testee_tmp[column][idx])
                    else:
                        csv_data["test"][column].append(testee_tmp[column][idx])
    train_df = pd.DataFrame(
        csv_data["train"],
        columns=columns,
    )
    test_df = pd.DataFrame(
        csv_data["test"],
        columns=columns,
    )
    val_df = test_df.copy()
    print('train shape:', train_df.shape)
    print('val shape:', val_df.shape)
    print('test shape:', test_df.shape)

    return train_df, val_df, test_df

def split_posedata_kfold_name(sound_data, name) -> Tuple[DataFrame]:
    columns = ["sound_path", "testee", "sound_length", "label"] + get_joints()

    data = sound_data[columns]
    data = data.set_index("sound_path")
    train_df = data[
        (data.testee != name)
    ].reset_index()
    val_df = data[
        (data.testee == name)
    ].reset_index()
    test_df = data[
        (data.testee == name)
    ].reset_index()
    return train_df, val_df, test_df


def create_subject_3_demo(sound_data) -> Tuple[DataFrame]:
    columns = ["sound_path", "testee", "sound_length", "label"] + get_joints()
    data = sound_data[columns]
    data = data.set_index("sound_path")
    train_df = data.reset_index()
    val_df = data.reset_index()
    test_df = data.reset_index()
    return train_df, val_df, test_df

def split_posedata_dark(sound_data) -> Tuple[DataFrame]:
    columns = ["sound_path", "testee", "sound_length", "label"] + get_joints()

    data = sound_data[columns]
    data = data.set_index("sound_path")
    train_df = data.reset_index()
    val_df = data.reset_index()
    test_df = data.reset_index()
    return train_df, val_df, test_df


def update_df(train_df, val_df, test_df, path, preprocess):
    is_file = os.path.isfile(path)
    if is_file:
        old_train_df = pd.read_csv(path)
        old_val_df = pd.read_csv(path.replace('train', 'val'))
        old_test_df = pd.read_csv(path.replace('train', 'test'))
        if preprocess in old_train_df['preprocess'].unique():
            return old_train_df, old_val_df, old_test_df
        train_df = pd.concat([train_df, old_train_df], axis = 0)
        train_df = train_df.reset_index(drop=True)
        val_df = pd.concat([val_df, old_val_df], axis = 0)
        val_df = val_df.reset_index(drop=True)
        test_df = pd.concat([test_df, old_test_df], axis = 0)
        test_df = val_df.reset_index(drop=True)
    return train_df, val_df, test_df
        


def main() -> None:
    args = get_arguments()

    if args.dataset_name not in DATASET_CSVS:
        if (args.dataset_name != 'pose_regression_timeseries') & (args.dataset_name != 'pose_regression_kfold'):
            message = f"dataset_name should be selected from {list(DATASET_CSVS.keys())}."
            raise ValueError(message)

    sound_data = pd.read_csv(args.sound_csv_path)

    sound_data.sound_path = sound_data.sound_path.map(os.path.abspath)


    if args.dataset_name == "pose_regression":
        train_df, val_df, test_df = split_posedata(sound_data)
    
    elif args.dataset_name == 'pose_regression_timeseries':
        train_df, val_df, test_df = split_posedata_timeseries_name(sound_data, args.subject_name)
        
    elif args.dataset_name == 'pose_regression_kfold':
        train_df, val_df, test_df = split_posedata_kfold_name(sound_data, args.subject_name)
        
    elif args.dataset_name == "pose_regression_subject_3_dark":
        train_df, val_df, test_df = split_posedata_dark(sound_data)
    
    elif args.dataset_name == "pose_regression_subject_8_dark":
        train_df, val_df, test_df = split_posedata_dark(sound_data)
        
    elif args.dataset_name == 'pose_regression_subject_3_demo':
        train_df, val_df, test_df = create_subject_3_demo(sound_data)

    train_df['preprocess'] = args.preprocess
    val_df['preprocess'] = args.preprocess
    test_df['preprocess'] = args.preprocess

    
    # kfoldの場合は名前もつける
    if (args.dataset_name == 'pose_regression_timeseries') | (args.dataset_name == 'pose_regression_kfold'):
        path = os.path.join("./csv", args.dataset_name + '_' + args.subject_name,"train.csv")
        train_df, val_df, test_df = update_df(train_df, val_df, test_df, path, preprocess=args.preprocess)
        print('train size:', train_df.shape)
        print('val size:', val_df.shape)
        print('test size:', test_df.shape)
        os.makedirs(
            os.path.join("./csv", args.dataset_name + '_' + args.subject_name),
            exist_ok=True,
        )
        train_df.to_csv(
            os.path.join("./csv", args.dataset_name + '_' + args.subject_name,"train.csv"),
            index=None,
        )
        val_df.to_csv(
            os.path.join("./csv",args.dataset_name+ '_' + args.subject_name,"val.csv"),
            index=None,
        )
        test_df.to_csv(
            os.path.join("./csv",args.dataset_name+ '_' + args.subject_name,"test.csv"),
            index=None,
        )
    else:
        # 保存ディレクトリがなければ，作成
        os.makedirs(
            os.path.dirname(getattr(DATASET_CSVS[args.dataset_name], "train")),
            exist_ok=True,
        )
        path = os.path.join(getattr(DATASET_CSVS[args.dataset_name],"train"))
        train_df, val_df, test_df = update_df(train_df, val_df, test_df, path, preprocess=args.preprocess)
        # 保存
        train_df.to_csv(
            os.path.join(getattr(DATASET_CSVS[args.dataset_name],"train")),
            index=None,
        )
        val_df.to_csv(
            os.path.join(getattr(DATASET_CSVS[args.dataset_name],"val")),
            index=None,
        )
        test_df.to_csv(
            os.path.join(getattr(DATASET_CSVS[args.dataset_name],"test")),
            index=None,
        )

    print("Finished making %s csv files." % args.dataset_name)


if __name__ == "__main__":
    main()