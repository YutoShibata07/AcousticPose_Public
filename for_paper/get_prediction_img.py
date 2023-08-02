import pandas as pd
import numpy as np
import glob 
import argparse
import cv2
import os


def get_arguments() -> argparse.Namespace:
    """parse all the arguments from command line inteface return a list of
    parsed arguments."""

    parser = argparse.ArgumentParser(
        description="make csv files for Sound Pose dataset"
    )
    parser.add_argument(
        "--img_path",
        type=str,
        default="../result/speech2pose_logmel_gan/images/validation.mp4",
        help="path to class label csv",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./for_paper/images",
        help="path to class label csv",
    )
    parser.add_argument(
        '--model',
        type=str,
        default='ours',
    )
    parser.add_argument(
        '--start_time',
        type=int,
        default=49,
    )
    parser.add_argument(
        '--video_title',
        type=str,
        default='kawashima_shihuku',
    )

    return parser.parse_args()
def main():
    args = get_arguments()
    cap = cv2.VideoCapture(args.img_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print('total frame num:', total_frame)
    max_frame = 10
    img_count = 0
    start_frame = args.start_time * fps
    os.makedirs(
        os.path.join(args.save_path, args.video_title, args.model, "start_" + str(args.start_time)),
        exist_ok=True,
    )
    print('fps:', fps)
    # 0.5秒に一枚欲しい
    hop_length = int(fps)
    for i in range(0, int(total_frame)):
        ret, frame = cap.read()
        if i < start_frame:
            continue
        if img_count > max_frame:
            break
        if i %  hop_length == 0:
            tmp_save_path = os.path.join(args.save_path, args.video_title, args.model, "start_" + str(args.start_time) ,f'{i}.jpg')
            cv2.imwrite(tmp_save_path, frame)
            img_count += 1
    return 

if __name__ == '__main__':
    main()