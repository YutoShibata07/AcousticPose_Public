import dataclasses
from logging import getLogger

logger = getLogger(__name__)

__all__ = ["DATASET_CSVS"]


@dataclasses.dataclass(frozen=True)
class DatasetCSV:
    train: str
    val: str
    test: str


DATASET_CSVS = {
    # paths from `src` directory
    "action_segmentation": DatasetCSV(
        train="./csv/train.csv",
        val="./csv/val.csv",
        test="./csv/test.csv",
    ),
    "action_segmentation_by_testee": DatasetCSV(
        train="./csv/by_testee/train.csv",
        val="./csv/by_testee/val.csv",
        test="./csv/by_testee/test.csv",
    ),
    "pose_regression": DatasetCSV(
        train="./csv/pose/train.csv",
        val="./csv/pose/val.csv",
        test="./csv/pose/test.csv",
    ),
}