from typing import Dict


def get_cls2id_map() -> Dict[str, int]:
    cls2id_map = {"no_people": 0, "bow": 1, "sit": 2, "stand": 3}

    return cls2id_map


def get_id2cls_map() -> Dict[int, str]:
    cls2id_map = get_cls2id_map()
    return {val: key for key, val in cls2id_map.items()}
