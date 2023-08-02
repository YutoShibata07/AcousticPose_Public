from typing import Any, Dict, List


def get_joint_names() -> List[str]:
    joint_names = [
        "Hips",
        "Spine",
        "Spine1",
        "Neck",
        "Head",
        "LeftShoulder",
        "LeftArm",
        "LeftForeArm",
        "LeftHand",
        "RightShoulder",
        "RightArm",
        "RightForeArm",
        "RightHand",
        "LeftUpLeg",
        "LeftLeg",
        "LeftFoot",
        "LeftToeBase",
        "RightUpLeg",
        "RightLeg",
        "RightFoot",
        "RightToeBase",
    ]

    return joint_names


def get_joints() -> List[str]:
    joints = []
    for name in get_joint_names():
        for dim in ["_x", "_y", "_z"]:
            joints.append(name + dim)
    return joints


def get_leg(head: str = "") -> List[str]:
    joint_names = ["ToeBase", "Foot", "Leg", "UpLeg", "Hips"]

    joint_names = [
        head + name if name != joint_names[-1] else name for name in joint_names
    ]

    return joint_names


def get_arm(head: str = "") -> List[str]:
    joint_names = ["Hand", "ForeArm", "Arm", "Shoulder", "Spine1"]

    joint_names = [
        head + name if name != joint_names[-1] else name for name in joint_names
    ]

    return joint_names


def get_body() -> List[str]:
    joint_names = ["Hips", "Spine", "Spine1", "Neck", "Head"]

    return joint_names


def joint2list(joint: Dict[str, Any], part: str = "all") -> List[Any]:
    targets = []
    if part == "all":
        joint_names = get_joint_names()
    elif part == "arm":
        joint_names = get_arm("Right")[:-1] + get_arm("Left")[:-1]
    elif part == "leg":
        joint_names = get_leg("Right")[:-1] + get_leg("Left")[:-1]
    elif part == "body":
        joint_names = get_body()

    for name in joint_names:
        x = joint[name + "_x"]
        y = joint[name + "_y"]
        z = joint[name + "_z"]
        targets.extend([x, y, z])
    return targets


def list2joint(joint_list: List[Any]) -> Dict[str, Any]:
    joint = {}
    for i, name in enumerate(get_joints()):
        joint[name] = joint_list[i]
    return joint
