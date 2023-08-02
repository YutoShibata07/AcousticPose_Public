"""
Code for reproduce figures used in CVPR2022 submission
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import glob
import cv2
from PIL import Image
import pandas as pd
import argparse

# from data_collection.generate_tsp import normal_tsp
# from utils.make_sound_dataset import LogMelIntensityExtractor

def get_arguments() -> argparse.Namespace:
    """parse all the arguments from command line inteface return a list of
    parsed arguments."""

    parser = argparse.ArgumentParser(
        description="make csv files for Sound Pose dataset"
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



def draw_and_save_template(data, dst_imgname, color):
    plt.figure(figsize=(3.14, 1.5))
    plt.rcParams["figure.autolayout"] = False
    plt.plot(data, color)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.axis("off")
    plt.savefig(os.path.expanduser(dst_imgname))


# def draw_tsp_signal(dst_imgname):
#     n = 8  # 実験に使ったのは12だと思うのですが，図を描く時に信号が見えやすいように変えています
#     normal_tsp_sig, normal_tsp_inv_sig = normal_tsp(n, repeat=1)
#     data = normal_tsp_sig[50:240]
#     draw_and_save_template(data, dst_imgname, "b")


def draw_images_template(dir_name, ext, crop, src_img, vconcat=False, start = 0):
    dir_name = os.path.expanduser(dir_name)
    files = glob.glob(dir_name + "*." + ext)
    margin = [10, 20]
    size = [400, 400]
    img = np.asarray(Image.new("RGB", (1, size[1]), (256, 256, 256)))
    img_margin_h = np.asarray(Image.new("RGB", (margin[0], size[1]), (256, 256, 256)))
    file_df = pd.DataFrame({'file':files})
    file_df['time'] = file_df['file'].apply(lambda x:int(x.split('/')[-1].split('.')[0]))
    file_df = file_df.sort_values(by='time')
    files = file_df['file'].values
    for file in files:
        print(file)
        tmp0 = np.asarray(Image.open(file))
        tmp1 = tmp0[crop[1] : crop[1] + crop[3], crop[0] : crop[0] + crop[2], :]
        tmp2 = cv2.resize(tmp1, dsize=(size[0], size[1]))
        if img.shape[1] > size[1]:
            img = cv2.hconcat([img, img_margin_h])
            img = cv2.hconcat([img, tmp2])
        else:
            img = cv2.hconcat([img, tmp2])
    if vconcat:
        img_margin_v = np.asarray(
            Image.new("RGB", (src_img.shape[1], margin[1]), (256, 256, 256))
        )
        ans = cv2.vconcat([src_img, img_margin_v])
        ans = cv2.vconcat([ans, img])
    else:
        ans = img
    return ans


if __name__ == "__main__":
    args = get_arguments()
    """Qualitative results image"""
    img = draw_images_template(
        f"./for_paper/images/{args.video_title}/RGB/start_{248}/",
        "jpg",
        crop=[200, 10, 400, 470],
        src_img=None,
        vconcat=False,
    )
    img = draw_images_template(
        f"./for_paper/images/{args.video_title}/gt/start_{str(args.start_time)}/",
        "jpg",
        crop=[10, 10, 450, 450],
        src_img=img,
        vconcat=True,
    )
    img = draw_images_template(
        f"./for_paper/images/{args.video_title}/speech2pose/start_{str(args.start_time)}/",
        "jpg",
        crop=[10, 10, 450, 450],
        src_img=img,
        vconcat=True,
    )
    img = draw_images_template(
        f"./for_paper/images/{args.video_title}/wipose/start_{str(args.start_time)}/",
        "jpg",
        crop=[10, 10, 450, 450],
        src_img=img,
        vconcat=True,
    )
    img = draw_images_template(
        f"./for_paper/images/{args.video_title}/ours/start_{str(args.start_time)}/",
        "jpg",
        crop=[10, 10, 450, 450],
        src_img=img,
        vconcat=True,
    )
    # img = draw_images_template(
    #     "~/datasets/soundpose/for_paper/qualitative_0/resnet/",
    #     "png",
    #     crop=[10, 10, 450, 450],
    #     src_img=img,
    #     vconcat=True,
    # )
    # img = draw_images_template(
    #     "~/datasets/soundpose/for_paper/qualitative_0/ours/",
    #     "png",
    #     crop=[10, 10, 450, 450],
    #     src_img=img,
    #     vconcat=True,
    # )
    # このまま必要な分だけstackしてください
    Image.fromarray(img).save(
        os.path.expanduser(f"./for_paper/images/{args.video_title}_{str(args.start_time)}.png")
    )
    '''
    """ TSP """
    draw_tsp_signal("~/datasets/soundpose/for_paper/tsp.png")

    """ Received audio signal """
    data = np.load(
        os.path.expanduser("~/datasets/soundpose/for_paper/kawashima_demo.npy")
    )
    draw_and_save_template(
        data[0:200, 0], "~/datasets/soundpose/for_paper/received_ch0.png", "r"
    )
    draw_and_save_template(
        data[0:200, 1], "~/datasets/soundpose/for_paper/received_ch1.png", "g"
    )
    draw_and_save_template(
        data[0:200, 2], "~/datasets/soundpose/for_paper/received_ch2.png", "y"
    )
    draw_and_save_template(
        data[0:200, 3], "~/datasets/soundpose/for_paper/received_ch3.png", "b"
    )

    """ Audio features (結局図には使ってない) """
    data = np.load(
        os.path.expanduser("~/datasets/soundpose/for_paper/kawashima_demo.npy")
    )
    audio = data[0:20000, :].transpose()
    sr = 48000  # sampling rate of the incoming signal
    sound_length = 4800  # number of FF components, sound_length_list = [400, 800, 1600, 2400, 4800]
    feature_extractor = LogMelIntensityExtractor(sr, sound_length)
    feature = feature_extractor.transform(audio)
    channel_num = audio.shape[0]
    feature_logmel = []
    for n in range(0, channel_num):
        feature_logmel.append(feature_extractor.logmel(audio[n]))
    feature_logmel = np.concatenate(feature_logmel, axis=0)
    feature_intensity = feature_extractor.intensity(audio)
    """
    '''
