import os
import glob
import argparse
import pandas as pd


def get_arguments() -> argparse.Namespace:
    """parse all the arguments from command line inteface return a list of
    parsed arguments."""

    parser = argparse.ArgumentParser(
        description="""
        preprocess OptiTrack csv.
        """
    )
    parser.add_argument("--opti_dir", type=str, help="path of a config file")

    return parser.parse_args()


def main():
    args = get_arguments()

    for path in glob.glob(os.path.join(args.opti_dir, "*")):

        columns = []

        with open(path, mode="r") as f:
            lines = f.readlines()
            noise = []
            for i in range(len(lines[0].split(",")[:-1])):
                if lines[0].split(",")[i].replace(" ", "") == "":
                    noise.append(i)
                columns.append(
                    lines[0].split(",")[i].replace(" ", "")
                    + "_"
                    + lines[1].split(",")[i].replace(" ", "")
                )
            data = {name: [] for i, name in enumerate(columns) if not i in noise}
            for line in lines[2:]:
                for i in range(len(columns)):
                    if not i in noise:
                        data[columns[i]].append(line.split(",")[i])

        columns = [name for i, name in enumerate(columns) if not i in noise]

        df = pd.DataFrame(data, columns=columns)
        df.to_csv(path)


if __name__ == "__main__":
    main()