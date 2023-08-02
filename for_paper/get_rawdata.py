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


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


def get_arguments() -> argparse.Namespace:
    """parse all the arguments from command line inteface return a list of
    parsed arguments."""

    parser = argparse.ArgumentParser(
        description="make sound dataset for sound pose estimation"
    )
    parser.add_argument(
        "--sound_dir",
        type=str,
        # default="./data/mic_cut/",
        default="~/datasets/soundpose/for_paper/",
        help="path to a sound dirctory",
    )

    return parser.parse_args()


def main() -> None:
    args = get_arguments()
    for sound_path in glob.glob(os.path.join(args.sound_dir, "*")):
        data_name = sound_path.split("/")[-1].split(".")[0]
        print("Making dataset from ", data_name)

        sound, _ = sf.read(sound_path)
        np.save(f"./{data_name}.npy", sound.astype("float32"))

        break


if __name__ == "__main__":
    main()
