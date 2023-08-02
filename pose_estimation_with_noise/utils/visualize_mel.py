import argparse
import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from libs.models import get_model


def get_arguments() -> argparse.Namespace:
    """parse all the arguments from command line inteface return a list of
    parsed arguments."""

    parser = argparse.ArgumentParser(description="Model visualization.")
    parser.add_argument(
        "model",
        type=str,
        choices=["resnet18", "resnet34", "resnet50"],
        help="name of the model you want to visualize.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./imgs",
        help="a directory where images will be saved",
    )

    return parser.parse_args()


def main() -> None:
    # args = get_arguments()
    testee_id = [4, 11, 17]
    chanel_id = 0
    for i in range(3):
        mel_file = glob.glob(f"./dataset/{testee_id[i]}_*.npy")
        mel_file = np.random.choice(mel_file, 1)[0]
        exp_id = mel_file.split("_")[1].split(".")[0]
        sound = np.load(mel_file)[:, chanel_id].reshape(-1, 1)
        # 音信号の数とフーリエ変換の幅によって変える必要あり
        fig, ax = plt.subplots()
        sound_dB = librosa.power_to_db(sound, ref=np.max)
        img = librosa.display.specshow(
            sound_dB, x_axis="time", y_axis="mel", sr=48000, ax=ax
        )
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        ax.set(title=f"Mel-frequency spectrogram:{testee_id[i]}_{exp_id}")
        plt.show()


if __name__ == "__main__":
    main()
