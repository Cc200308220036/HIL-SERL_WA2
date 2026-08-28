from enum import Enum


class ArmGroup(Enum):
    """
    用于 movej / movel / speedj / speedl 等
    """
    LEFT = 1
    RIGHT = 2
    DUAL = 3
    NECK = 4
    WAIST = 8
    LIFT = 16


def as_arm_group(arm) -> ArmGroup:
    if isinstance(arm, ArmGroup):
        return arm
    return ArmGroup(arm)


class RobotModel(Enum):
    WA1 = "wa1"
    WA2 = "wa2"


def as_robot_model(model) -> RobotModel:
    if isinstance(model, RobotModel):
        return model
    if isinstance(model, str):
        return RobotModel(model.lower())
    raise TypeError("model must be RobotModel or str")


ARM_SERVICE_NAMES = {
            ArmGroup.LEFT:  "left_arm",
            ArmGroup.RIGHT: "right_arm",
            ArmGroup.DUAL:  "dual_arm",
            ArmGroup.NECK:  "neck",
            ArmGroup.WAIST: "waist",
            ArmGroup.LIFT:  "lifting",
        }

ARM_DOF = {
            ArmGroup.LEFT:  ("left_arm", 7),
            ArmGroup.RIGHT: ("right_arm", 7),
            ArmGroup.DUAL:  ("dual_arm", 14),
            ArmGroup.NECK:  ("neck", 2),
            ArmGroup.WAIST: ("waist", 2),
            ArmGroup.LIFT:  ("lifting", 1),
        }

JOINT_DOF_BY_MODEL = {
            RobotModel.WA1: {
                ArmGroup.LEFT: 7,
                ArmGroup.RIGHT: 7,
                ArmGroup.DUAL: 14,
                ArmGroup.NECK: 2,
                ArmGroup.WAIST: 2,
                ArmGroup.LIFT: 1,
            },
            RobotModel.WA2: {
                ArmGroup.LEFT: 8,
                ArmGroup.RIGHT: 8,
                ArmGroup.DUAL: 16,
                ArmGroup.NECK: 2,
                ArmGroup.WAIST: 4,
            },
        }

JOINT_LAYOUT_BY_MODEL = {
            RobotModel.WA1: (
                (ArmGroup.LEFT, 7),
                (ArmGroup.RIGHT, 7),
                (ArmGroup.NECK, 2),
                (ArmGroup.WAIST, 2),
                (ArmGroup.LIFT, 1),
            ),
            RobotModel.WA2: (
                (ArmGroup.LEFT, 8),
                (ArmGroup.RIGHT, 8),
                (ArmGroup.NECK, 2),
                (ArmGroup.WAIST, 4),
            ),
        }

TCP_POSE_DOF = {
            ArmGroup.LEFT: 7,
            ArmGroup.RIGHT: 7,
            ArmGroup.DUAL: 14,
        }

TCP_SPEED_DOF = {
            ArmGroup.LEFT: 6,
            ArmGroup.RIGHT: 6,
            ArmGroup.DUAL: 12,
        }

L_GROUP = (ArmGroup.LEFT, ArmGroup.RIGHT, ArmGroup.DUAL)

class HandType(Enum):
    LEFT = 1
    RIGHT = 2


class CmdState(Enum):
    """
    对应 UplimbState.cmd_num
    """
    STOPPED = 0
    MOVEJ = 1
    MOVEJ_PATH = 2
    MOVEL_NULLSPACE = 3
    MOVEL = 4
    MOVEL_PATH = 5
    SPEEDJ = 6
    SPEEDL = 7
    SPEED_STOP = 8
    MOVECSVFILE = 9
    MOVEFOURIER = 10
    MOVEJ_SPLINE = 11
    TEACH = 12
    SERVOJ = 13
    SERVOL = 14
