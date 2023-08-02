from logging import getLogger

import argparse
import glob
import os
import sys
from typing import Dict, List, Union
import tqdm

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import warnings
warnings.simplefilter('ignore')

logger = getLogger(__name__)

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from libs.sound_length_list import get_sound_length_list
from libs.joint_list import get_joints, get_joint_names


def get_arguments() -> argparse.Namespace:
    """parse all the arguments from command line inteface return a list of
    parsed arguments."""

    parser = argparse.ArgumentParser(
        description="make sound dataset for sound pose estimation"
    )
    parser.add_argument(
        "--sound_dir",
        type=str,
        default="./data/sound/",
        help="path to a sound dirctory",
    )
    parser.add_argument(
        "--annotation_dir",
        type=str,
        default="./data/annotations/",
        help="a path where annotations are",
    )
    parser.add_argument(
        "--position_data_path",
        type=str,
        default="./data/poses",
        help="a path where position csv file",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./dataset",
        help="a directory where sound dataset will be saved",
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=48000,
        help="sampling rate of sound data",
    )
    parser.add_argument(
        "--processed_method",
        type=str,
        default="original",
        help="a path where processed mic data",
    )
    parser.add_argument(
        "--denoise_method",
        type=str,
        default="",
        help="description of denoise method ex. wpe, inv_conv_subtract",
    )
    return parser.parse_args()


class LogMelIntensityExtractor:
    def __init__(self, fs, nfft, norm_intensity):

        self.nfft = nfft
        self.melW = librosa.filters.mel(
            sr=fs,
            n_fft=nfft,
        )
        self.norm_intensity = norm_intensity

    def logmel(self, sig):
        S = (
            np.abs(
                librosa.stft(
                    y=sig,
                    n_fft=self.nfft,
                    center=False,
                )
            )
            ** 2
        )
        S_mel = np.dot(self.melW, S).T
        S_logmel = librosa.power_to_db(S_mel, ref=1.0, amin=1e-10, top_db=None)
        S_logmel = np.expand_dims(S_logmel, axis=0)

        return S_logmel

    def intensity(self, sig):

        ref = sig[0]
        x = sig[1]
        y = sig[2]
        z = sig[3]

        Pref = librosa.stft(
            y=ref,
            n_fft=self.nfft,
            center=False,
        )
        Px = librosa.stft(
            y=x,
            n_fft=self.nfft,
            center=False,
        )
        Py = librosa.stft(
            y=y,
            n_fft=self.nfft,
            center=False,
        )
        Pz = librosa.stft(
            y=z,
            n_fft=self.nfft,
            center=False,
        )

        I1 = np.real(np.conj(Pref) * Px)
        I2 = np.real(np.conj(Pref) * Py)
        I3 = np.real(np.conj(Pref) * Pz)
        normal = np.sqrt(I1 ** 2 + I2 ** 2 + I3 ** 2)
        if self.norm_intensity==True:
            I1 = np.dot(self.melW, I1 / normal).T
            I2 = np.dot(self.melW, I2 / normal).T
            I3 = np.dot(self.melW, I3 / normal).T
        else:
            I1 = np.dot(self.melW, I1).T
            I2 = np.dot(self.melW, I2).T
            I3 = np.dot(self.melW, I3).T
        intensity = np.array([I1, I2, I3])

        return intensity

    def transform(self, audio):

        channel_num = audio.shape[0]
        feature_logmel = []
        for n in range(0, channel_num):
            feature_logmel.append(self.logmel(audio[n]))
        feature_intensity = self.intensity(sig=audio)

        feature_logmel = np.concatenate(feature_logmel, axis=0)
        feature = np.concatenate([feature_logmel, feature_intensity], axis=0)
        feature = feature.reshape(7, -1).T

        return feature


def read_annotation(path: str):
    anno = {}
    with open(path) as f:
        for line in f.readlines():
            start, end, pose = line.split(" ")
            pose = pose.split("\n")[0]
            anno[pose] = (float(start), float(end))

    return anno


def joint_norm(joint_locations: Dict[str, float]):
    shift = {}
    magnification = 0
    for dim in ["_x", "_y", "_z"]:
        shift[dim[-1]] = joint_locations["Hips" + dim]

        magnification += (
            joint_locations["Spine" + dim] - joint_locations["Hips" + dim]
        ) ** 2
    magnification = np.sqrt(magnification)
    for k, v in joint_locations.items():
        joint_locations[k] = (v - shift[k[-1]]) / magnification
    for joint in get_joint_names():
        shift["y"] = min(shift["y"], joint_locations[joint + "_y"])
    for joint in get_joint_names():
        joint_locations[joint + "_y"] = joint_locations[joint + "_y"] - shift["y"]


def main() -> None:
    args = get_arguments()

    # 保存ディレクトリがなければ，作成
    os.makedirs(args.save_dir, exist_ok=True)
    if args.processed_method == "original":
        dataset_name = "sound"
    elif args.processed_method == "raw":
        dataset_name = "sound_raw"
    elif args.processed_method == 'intensity_wo_norm':
        dataset_name = 'intensity_wo_norm'
    else:
        dataset_name = "sound_with_intensity"
    dataset_dir = os.path.join(args.save_dir, dataset_name)

    # すでにデータセットが存在しているなら終了
    if os.path.exists(dataset_dir):
        print("Sound dataset exists.")
        return
    else:
        os.mkdir(dataset_dir)

    sound_length_list = get_sound_length_list()
    joint_names = get_joints()

    columns = ["sound_path", "label", "testee", "sound_length"] + joint_names

    data: Dict[str, List[Union[int, str]]] = {name: [] for name in columns}
    DEBUG = False
    anno_path_list = glob.glob(os.path.join(args.annotation_dir, "*"))
    data_for_train = ["002", '005', "009", "012", "015", "018", "021", "004", "007", "010", "013", "016", "020", "022"]
    anno_path_list = [anno_path for anno_path in anno_path_list if str(anno_path).split('/')[-1].split('.txt')[0] in data_for_train]
    testee_id_map = {"002":"subject_1", "004":"subject_2", "005":"subject_2", "007":'subject_3', "009":'subject_4',"010":'subject_4', "012":"subject_5", "013":'subject_5', "015":"subject_6", "016":"subject_6","018":"subject_7", "020":"subject_8", "021":"subject_8"}
    print(len(anno_path_list))
    for anno_idx, anno_path in enumerate(anno_path_list):
        print("anno path:", anno_path)
        data_name = anno_path.split("/")[-1].split(".")[0]
        if args.denoise_method != '':
            mic_path = os.path.join(args.sound_dir, data_name + "_" + args.denoise_method + ".wav")
        else:
            mic_path = os.path.join(args.sound_dir, data_name + "_cut.WAV")
        position_path = os.path.join(args.position_data_path, "position_" + data_name + ".csv")
        if "demo" in data_name:
            continue
        print(
            "%s/%s Making dataset from %s"
            % (anno_idx + 1, len(anno_path_list), data_name)
        )
        for sound_length in sound_length_list:
            print(sound_length)

            # Read OptiTrack
            position_csv = pd.read_csv(position_path)
            position_size = sound_length / 400
            sound, _ = sf.read(mic_path)
            sample_num = len(sound) // sound_length  # データセットのフレーム数
            anno = read_annotation(anno_path)
            testee = testee_id_map[data_name.split("_")[0]]
            if args.processed_method == 'intensity_wo_norm':
                norm_intensity = False
            else:
                norm_intensity = True
            feature_extractor = LogMelIntensityExtractor(args.sr, sound_length, norm_intensity=norm_intensity)

            for frame in tqdm.tqdm(range(sample_num)):  # データセットの各フレームに対して処理を実行
                if DEBUG:
                    if frame > 50:break
                sound_ = sound[sound_length * frame : sound_length * (frame + 1)]
                second = frame / args.sr * sound_length  # 秒数timestampを取得

                for pose, (start, end) in anno.items():
                    if pose == "no_people":
                        continue
                    if second < start:
                        continue
                    if end < second:
                        continue

                    if args.processed_method == "original":
                        processed_file_name = "%s_%s_%s.npy"
                        sound_ = np.concatenate(
                            [
                                librosa.power_to_db(
                                    librosa.feature.melspectrogram(
                                        sound_[:, i],
                                        sr=args.sr,
                                        n_fft=sound_length,
                                        center=False,
                                    ),
                                    ref=np.max,
                                )
                                for i in range(4)
                            ],
                            axis=1,
                        )
                    elif args.processed_method == "raw":
                        processed_file_name = "%s_%s_%s_raw.npy"
                        sound_ = sound_
                        if len(sound_) != sound_length:continue
                    elif args.processed_method == 'intensity_wo_norm':
                        processed_file_name = "%s_%s_%s_with_intensity_wo_norm.npy"
                        sound_ = sound_.T  # [4, sample_num]
                        sound_ = feature_extractor.transform(sound_)
                    else:
                        processed_file_name = "%s_%s_%s_with_intensity.npy"
                        sound_ = sound_.T  # [4, sample_num]
                        sound_ = feature_extractor.transform(sound_)

                    joint_locations = {}
                    for joint_name in joint_names:
                        joint_location = position_csv.iloc[
                            int(np.round(frame * position_size)) : int(np.round((frame + 1) * position_size))
                        ][joint_name]
                        joint_location = np.mean(joint_location, axis=0)
                        joint_locations[joint_name] = joint_location
                    joint_norm(joint_locations)
                    for joint_name in joint_names:
                        data[joint_name].append(joint_locations[joint_name])

                    data["label"].append(pose)
                    sound_path = os.path.join(
                        dataset_dir,
                        processed_file_name % (data_name, frame, sound_length),
                    )
                    data["sound_path"].append(sound_path)
                    data["testee"].append(testee)
                    data["sound_length"].append(sound_length)
                    np.save(sound_path, sound_.astype("float32"))

    # list を DataFrame に変換
    df = pd.DataFrame(
        data,
        columns=columns,
    )

    # 保存
    if args.processed_method == "original":
        csv_file_name = "sound.csv"
    elif args.processed_method == "raw":
        csv_file_name = 'sound_raw.csv'
    elif args.processed_method == 'intensity_wo_norm':
        csv_file_name = 'sound_intensity_wo_norm.csv'
    else:
        csv_file_name = "sound_intensity.csv"
    df.to_csv(os.path.join(args.save_dir, csv_file_name), index=None)

    print("Finished making sound dataset.")


if __name__ == "__main__":
    main()
