from typing import List

MOVE_GROUP_ARM = "arm"
MOVE_GROUP_GRIPPER = "gripper"

OPEN_GRIPPER_JOINT_POSITIONS = [0.0]
CLOSED_GRIPPER_JOINT_POSITIONS = [-0.03]

def joint_names() -> List[str]:
    return [
        "column_joint",
        "shoulder_joint",
        "forearm_joint",
        "wrist_joint",
    ]

def base_link_name() -> str:
    return "base_link"

def end_effector_name() -> str:
    return "tool0"

def gripper_joint_names() -> List[str]:
    return [
        "left_finger_joint",
    ]
