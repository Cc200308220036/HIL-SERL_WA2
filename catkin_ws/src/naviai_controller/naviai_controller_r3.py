#!/usr/bin/env python3
"""
Single-file version of the naviai_controller package.

This file keeps the high-level NaviController API, but inlines the original
ArmController, HandController, enum definitions, and small pose utilities so it
can be used like navi_controller_r3.py without importing the naviai_controller
Python package.
"""

import threading
import time
from enum import Enum
from threading import Lock
from typing import List, Optional, Sequence, Union

import numpy as np
import rospy
from geometry_msgs.msg import Point, Pose, Quaternion, WrenchStamped
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import JointState
from std_srvs.srv import SetBool, SetBoolRequest

try:
    from upperlimb.msg import (
        Joints,
        Pose as UplimbPose,
        SpeedJ,
        SpeedL,
        TcpSpeed,
        UplimbState,
    )
    from upperlimb.srv import (
        MoveJ,
        MoveJByPath,
        MoveJByPathRequest,
        MoveJRequest,
        MoveL,
        MoveLRequest,
        Servo,
        ServoRequest,
    )
except ImportError:
    from zj_humanoid.upperlimb.msg import (
        Joints,
        Pose as UplimbPose,
        SpeedJ,
        SpeedL,
        TcpSpeed,
        UplimbState,
    )
    from zj_humanoid.upperlimb.srv import (
        MoveJ,
        MoveJByPath,
        MoveJByPathRequest,
        MoveJRequest,
        MoveL,
        MoveLRequest,
        Servo,
        ServoRequest,
    )

try:
    from zj_humanoid.hand.msg import PressureSensor
    from zj_humanoid.hand.srv import HandJoint, HandJointRequest
except ImportError:
    from hand.msg import PressureSensor
    from hand.srv import HandJoint, HandJointRequest


class ArmGroup(Enum):
    """Body part selector for movej / movel / speedj / speedl."""

    LEFT = 1
    RIGHT = 2
    DUAL = 3
    NECK = 4
    WAIST = 8
    LIFT = 16


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
    ArmGroup.LEFT: "left_arm",
    ArmGroup.RIGHT: "right_arm",
    ArmGroup.DUAL: "dual_arm",
    ArmGroup.NECK: "neck",
    ArmGroup.WAIST: "waist",
    ArmGroup.LIFT: "lifting",
}

ARM_DOF = {
    ArmGroup.LEFT: ("left_arm", 7),
    ArmGroup.RIGHT: ("right_arm", 7),
    ArmGroup.DUAL: ("dual_arm", 14),
    ArmGroup.NECK: ("neck", 2),
    ArmGroup.WAIST: ("waist", 2),
    ArmGroup.LIFT: ("lifting", 1),
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

LINEAR_ARM_GROUPS = (ArmGroup.LEFT, ArmGroup.RIGHT, ArmGroup.DUAL)
WHOLE_BODY_GROUPS = (
    ArmGroup.LEFT,
    ArmGroup.RIGHT,
    ArmGroup.NECK,
    ArmGroup.WAIST,
    ArmGroup.LIFT,
)


class HandType(Enum):
    LEFT = 1
    RIGHT = 2


class CmdState(Enum):
    """Values matching UplimbState.cmd_num."""

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


def as_arm_group(arm: Union[ArmGroup, int]) -> ArmGroup:
    if isinstance(arm, ArmGroup):
        return arm
    return ArmGroup(arm)


def as_hand_type(hand: Union[HandType, int]) -> HandType:
    if isinstance(hand, HandType):
        return hand
    return HandType(hand)


def check_length(data: Sequence[float], length: int, name: str) -> bool:
    if len(data) != length:
        raise ValueError(f"{name} length must be {length}, but got {len(data)}")
    return True


def rt_to_matrix(rt: Sequence[float]) -> np.ndarray:
    """[x, y, z, qx, qy, qz, qw] -> 4x4 matrix."""
    check_length(rt, 7, "rt")
    matrix = np.eye(4)
    matrix[:3, :3] = R.from_quat(rt[3:]).as_matrix()
    matrix[:3, 3] = rt[:3]
    return matrix


def matrix_to_rt(matrix: np.ndarray) -> List[float]:
    """4x4 matrix -> [x, y, z, qx, qy, qz, qw]."""
    if matrix.shape != (4, 4):
        raise ValueError("matrix must be 4x4")
    position = matrix[:3, 3]
    quaternion = R.from_matrix(matrix[:3, :3]).as_quat()
    return list(position) + list(quaternion)


def list_to_pose(rt: Sequence[float]) -> Pose:
    """[x, y, z, qx, qy, qz, qw] -> geometry_msgs/Pose."""
    check_length(rt, 7, "rt")
    pose = Pose()
    pose.position = Point(*rt[:3])
    pose.orientation = Quaternion(*rt[3:])
    return pose


def pose_to_rt(pose: Pose) -> List[float]:
    """geometry_msgs/Pose -> [x, y, z, qx, qy, qz, qw]."""
    return [
        pose.position.x,
        pose.position.y,
        pose.position.z,
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
    ]


class ArmController:
    def __init__(self, prefix: str, model="wa1"):
        self.prefix = prefix
        self.model = as_robot_model(model)
        self.joint_dof = JOINT_DOF_BY_MODEL[self.model]
        self.joint_layout = JOINT_LAYOUT_BY_MODEL[self.model]
        self._lock = Lock()

        self._joint_states = None
        self._left_joints = None
        self._right_joints = None
        self._neck_joints = None
        self._waist_joints = None
        self._lift_joint = None
        self._left_tcp = None
        self._right_tcp = None
        self._tcp_speed = None
        self._cmd_state = None

        self._init_publishers()
        time.sleep(1)
        self._init_subscribers()
        time.sleep(1)
        rospy.loginfo(f"ArmController initialized: {self.model.value}")

    def _init_subscribers(self) -> None:
        rospy.Subscriber(f"{self.prefix}/joint_states", JointState, self._joint_states_cb)
        rospy.Subscriber(f"{self.prefix}/uplimb_state", UplimbState, self._cmd_state_cb)
        rospy.Subscriber(f"{self.prefix}/tcp_pose/left_arm", UplimbPose, self._left_tcp_cb)
        rospy.Subscriber(f"{self.prefix}/tcp_pose/right_arm", UplimbPose, self._right_tcp_cb)
        rospy.Subscriber(f"{self.prefix}/tcp_speed/dual_arm", TcpSpeed, self._tcp_speed_cb)

    def _init_publishers(self) -> None:
        self._speedl_pubs = {
            ArmGroup.LEFT: rospy.Publisher(f"{self.prefix}/speedl/{ARM_SERVICE_NAMES[ArmGroup.LEFT]}", SpeedL, queue_size=1),
            ArmGroup.RIGHT: rospy.Publisher(f"{self.prefix}/speedl/{ARM_SERVICE_NAMES[ArmGroup.RIGHT]}", SpeedL, queue_size=1),
            ArmGroup.DUAL: rospy.Publisher(f"{self.prefix}/speedl/{ARM_SERVICE_NAMES[ArmGroup.DUAL]}", SpeedL, queue_size=1),
        }
        self._speedj_pubs = {
            group: rospy.Publisher(
                f"{self.prefix}/speedj/{ARM_SERVICE_NAMES[group]}",
                SpeedJ,
                queue_size=1,
            )
            for group in self.joint_dof
        }
        self._servoj_dual_pub = rospy.Publisher(f"{self.prefix}/servoj/dual_arm", Joints, queue_size=1)

    def _joint_states_cb(self, msg: JointState) -> None:
        with self._lock:
            self._joint_states = msg
            joints = list(msg.position)
            expected_len = sum(length for _, length in self.joint_layout)
            if len(joints) < expected_len:
                return

            slices = {}
            start = 0
            for group, length in self.joint_layout:
                slices[group] = joints[start:start + length]
                start += length

            self._left_joints = slices.get(ArmGroup.LEFT)
            self._right_joints = slices.get(ArmGroup.RIGHT)
            self._neck_joints = slices.get(ArmGroup.NECK)
            self._waist_joints = slices.get(ArmGroup.WAIST)
            self._lift_joint = slices.get(ArmGroup.LIFT)

    def _left_tcp_cb(self, msg: UplimbPose) -> None:
        with self._lock:
            self._left_tcp = [
                msg.position.x,
                msg.position.y,
                msg.position.z,
                msg.quaternion.x,
                msg.quaternion.y,
                msg.quaternion.z,
                msg.quaternion.w,
            ]

    def _right_tcp_cb(self, msg: UplimbPose) -> None:
        with self._lock:
            self._right_tcp = [
                msg.position.x,
                msg.position.y,
                msg.position.z,
                msg.quaternion.x,
                msg.quaternion.y,
                msg.quaternion.z,
                msg.quaternion.w,
            ]

    def _tcp_speed_cb(self, msg: TcpSpeed) -> None:
        with self._lock:
            self._tcp_speed = msg

    def _cmd_state_cb(self, msg: UplimbState) -> None:
        with self._lock:
            self._cmd_state = msg

    def _joint_service(self, arm: ArmGroup) -> str:
        if arm not in self.joint_dof:
            raise ValueError(f"{self.model.value} does not support {arm.name}")
        return ARM_SERVICE_NAMES[arm]

    def _joint_dof(self, arm: ArmGroup) -> int:
        if arm not in self.joint_dof:
            raise ValueError(f"{self.model.value} does not support {arm.name}")
        return self.joint_dof[arm]

    def _whole_body_expected_len(self, arm_mask: int) -> int:
        if not isinstance(arm_mask, int):
            raise TypeError("arm_mask must be int (8421 code)")

        supported_groups = tuple(group for group in self.joint_dof if group != ArmGroup.DUAL)
        supported_bits = sum(group.value for group in supported_groups)
        unsupported_bits = arm_mask & ~supported_bits
        if unsupported_bits:
            unsupported_names = [
                group.name
                for group in ArmGroup
                if group != ArmGroup.DUAL and unsupported_bits & group.value
            ]
            detail = ", ".join(unsupported_names) if unsupported_names else str(unsupported_bits)
            raise ValueError(f"{self.model.value} does not support arm_mask bits: {unsupported_bits} ({detail})")

        expected_len = 0
        for group in supported_groups:
            if arm_mask & group.value:
                expected_len += self._joint_dof(group)

        if expected_len == 0:
            raise ValueError("arm_mask selects no body part")
        return expected_len

    def get_joints(self, arm: Union[ArmGroup, int]):
        arm = as_arm_group(arm)
        with self._lock:
            if arm == ArmGroup.LEFT:
                return None if self._left_joints is None else self._left_joints.copy()
            if arm == ArmGroup.RIGHT:
                return None if self._right_joints is None else self._right_joints.copy()
            if arm == ArmGroup.DUAL:
                if self._left_joints is None or self._right_joints is None:
                    return None
                return self._left_joints.copy() + self._right_joints.copy()
            if arm == ArmGroup.NECK:
                return None if self._neck_joints is None else self._neck_joints.copy()
            if arm == ArmGroup.WAIST:
                return None if self._waist_joints is None else self._waist_joints.copy()
            if arm == ArmGroup.LIFT:
                return None if self._lift_joint is None else self._lift_joint.copy()
        return None

    def get_tcp_rt(self, arm: Union[ArmGroup, int]):
        arm = as_arm_group(arm)
        with self._lock:
            if arm == ArmGroup.LEFT:
                return None if self._left_tcp is None else self._left_tcp.copy()
            if arm == ArmGroup.RIGHT:
                return None if self._right_tcp is None else self._right_tcp.copy()
        return None

    def get_tcp_matrix(self, arm: Union[ArmGroup, int]):
        tcp_rt = self.get_tcp_rt(arm)
        if tcp_rt is None:
            return None
        return rt_to_matrix(tcp_rt)

    def get_tcp_pose(self, arm: Union[ArmGroup, int]):
        tcp_rt = self.get_tcp_rt(arm)
        if tcp_rt is None:
            return None
        return list_to_pose(tcp_rt)

    def get_tcp_speed(self, arm: Union[ArmGroup, int]):
        arm = as_arm_group(arm)
        with self._lock:
            if self._tcp_speed is None:
                return None
            if arm == ArmGroup.LEFT:
                return list(self._tcp_speed.left_arm)
            if arm == ArmGroup.RIGHT:
                return list(self._tcp_speed.right_arm)
        return None

    def get_robot_state(self):
        with self._lock:
            return self._cmd_state

    def is_robot_moving(self) -> bool:
        with self._lock:
            if self._cmd_state is None:
                return False
            return self._cmd_state.cmd_num != CmdState.STOPPED.value

    def movej(
        self,
        joints: Sequence[float],
        arm: Union[ArmGroup, int],
        v: float = 0.3,
        acc: float = 0.5,
        t: Optional[float] = None,
        is_async: bool = False,
    ) -> bool:
        arm = as_arm_group(arm)
        service_suffix = self._joint_service(arm)
        dof = self._joint_dof(arm)
        check_length(joints, dof, "joints")

        request = MoveJRequest()
        request.joints = list(joints)
        request.v = v
        request.acc = acc
        request.is_async = is_async
        if t is not None:
            request.t = t

        return self._call_service(f"{self.prefix}/movej/{service_suffix}", MoveJ, request, "MoveJ")

    def movejh(
        self,
        arm_mask: int,
        joints: Sequence[float],
        v: float = 0.3,
        acc: float = 0.5,
        is_async: bool = False,
    ) -> bool:
        expected_len = self._whole_body_expected_len(arm_mask)
        check_length(joints, expected_len, "joints")

        request = MoveJRequest()
        request.arm_type = arm_mask
        request.joints = list(joints)
        request.v = v
        request.acc = acc
        request.is_async = is_async

        return self._call_service(f"{self.prefix}/movej/whole_body", MoveJ, request, "MoveJWholeBody")

    def movel(
        self,
        pose: Sequence[float],
        arm: Union[ArmGroup, int],
        v: float = 0.1,
        acc: float = 0.1,
        is_async: bool = False,
    ) -> bool:
        arm = as_arm_group(arm)
        if arm not in LINEAR_ARM_GROUPS:
            raise ValueError(f"Unsupported arm group: {arm}")

        service_suffix = ARM_SERVICE_NAMES[arm]
        check_length(pose, TCP_POSE_DOF[arm], f"{arm} pose")
        pose_list = []
        if arm == ArmGroup.DUAL:
            pose_list.append(list_to_pose(pose[:7]))
            pose_list.append(list_to_pose(pose[7:]))
        else:
            pose_list.append(list_to_pose(pose))

        request = MoveLRequest()
        request.pose = pose_list
        request.v = v
        request.acc = acc
        request.is_async = is_async

        return self._call_service(f"{self.prefix}/movel/{service_suffix}", MoveL, request, "MoveL")

    def movel_relative_base(
        self,
        delta_xyz: Sequence[float],
        arm: Union[ArmGroup, int],
        v: float = 0.1,
        acc: float = 0.1,
        is_async: bool = False,
    ) -> bool:
        arm = as_arm_group(arm)
        if arm not in LINEAR_ARM_GROUPS:
            raise ValueError(f"Unsupported arm group: {arm}")

        if arm == ArmGroup.DUAL:
            check_length(delta_xyz, 6, "delta_xyz")
            left_pose = self.get_tcp_rt(ArmGroup.LEFT)
            right_pose = self.get_tcp_rt(ArmGroup.RIGHT)
            if left_pose is None or right_pose is None:
                raise ValueError("left or right tcp pose is not ready")
            left_pose[0:3] = [left_pose[i] + float(delta_xyz[i]) for i in range(3)]
            right_pose[0:3] = [right_pose[i] + float(delta_xyz[i + 3]) for i in range(3)]
            pose = left_pose + right_pose
        else:
            check_length(delta_xyz, 3, "delta_xyz")
            pose = self.get_tcp_rt(arm)
            if pose is None:
                raise ValueError("tcp pose is not ready")
            pose[0:3] = [pose[i] + float(delta_xyz[i]) for i in range(3)]

        return self.movel(pose, arm, v, acc, is_async)

    def movel_relative_eef(
        self,
        transform: Union[np.ndarray, Sequence[Sequence[float]]],
        arm: Union[ArmGroup, int],
        v: float = 0.1,
        acc: float = 0.1,
        is_async: bool = False,
    ) -> bool:
        arm = as_arm_group(arm)
        transform_matrix = np.asarray(transform, dtype=np.float64)
        if transform_matrix.shape != (4, 4):
            if transform_matrix.size == 16:
                transform_matrix = transform_matrix.reshape(4, 4)
            else:
                raise ValueError(f"transform must be 4x4 or 16 floats, got shape {tuple(transform_matrix.shape)}")

        tcp_rt = self.get_tcp_rt(arm)
        if tcp_rt is None:
            raise ValueError("tcp pose is not ready")
        target_tcp_rt = matrix_to_rt(rt_to_matrix(tcp_rt) @ transform_matrix)
        return self.movel(target_tcp_rt, arm, v, acc, is_async)

    def enable_speedj(self, enable: bool = True) -> bool:
        return self._call_setbool_service(f"{self.prefix}/enable_speedj", enable, "SpeedJ")

    def enable_speedl(self, enable: bool = True) -> bool:
        return self._call_setbool_service(f"{self.prefix}/enable_speedl", enable, "SpeedL")

    def speedj(
        self,
        joint_speed: Sequence[float],
        arm: Union[ArmGroup, int],
        acc: float = 0.05,
    ) -> None:
        arm = as_arm_group(arm)
        check_length(joint_speed, self._joint_dof(arm), "joint_speed")

        msg = SpeedJ()
        msg.joint_speed = list(joint_speed)
        msg.acc = acc
        self._speedj_pubs[arm].publish(msg)

    def speedl(
        self,
        tcp_speed: Sequence[float],
        arm: Union[ArmGroup, int],
        acc: float = 0.05,
    ) -> None:
        arm = as_arm_group(arm)
        if arm not in LINEAR_ARM_GROUPS:
            raise ValueError(f"Unsupported arm group: {arm}")
        length = TCP_SPEED_DOF[arm]
        check_length(tcp_speed, length, "tcp_speed")

        msg = SpeedL()
        msg.tcp_speed = list(tcp_speed)
        msg.acc = acc
        self._speedl_pubs[arm].publish(msg)

    def stop_speedj(self, arm: Union[ArmGroup, int]) -> None:
        arm = as_arm_group(arm)
        self.speedj([0.0] * self._joint_dof(arm), arm, acc=0.05)

    def stop_speedl(self, arm: Union[ArmGroup, int]) -> None:
        arm = as_arm_group(arm)
        if arm not in LINEAR_ARM_GROUPS:
            raise ValueError(f"Unsupported arm group: {arm}")
        length = TCP_SPEED_DOF[arm]
        self.speedl([0.0] * length, arm, acc=0.05)

    def set_servo_params(self, time_sec: float, gain: int) -> bool:
        request = ServoRequest(time=time_sec, gain=gain, arm_type=ArmGroup.DUAL.value)
        return self._call_service(f"{self.prefix}/set_servo_params", Servo, request, "ServoJ")

    def clear_servo_params(self) -> bool:
        return self._call_service(f"{self.prefix}/clear_servo_params", Servo, ServoRequest(), "ServoJ")

    def servoj_dual_arm(self, joints: Sequence[float]) -> None:
        check_length(joints, self._joint_dof(ArmGroup.DUAL), "servoj dual_arm joints")
        self._servoj_dual_pub.publish(Joints(list(joints)))

    def movej_by_path(
        self,
        path: Sequence[Sequence[float]],
        arm: Union[ArmGroup, int],
        total_time: Optional[float] = None,
        timestamps: Optional[Sequence[float]] = None,
        is_async: bool = False,
    ) -> bool:
        arm = as_arm_group(arm)
        if arm == ArmGroup.LIFT:
            raise ValueError("LIFT does not support movej_by_path")
        if len(path) < 2:
            raise ValueError("path length must be greater than 1")
        if total_time is None and timestamps is None:
            raise ValueError("total_time and timestamps cannot both be None")

        service_suffix = self._joint_service(arm)
        dof = self._joint_dof(arm)
        request = MoveJByPathRequest()
        for index, joints in enumerate(path):
            check_length(joints, dof, f"path[{index}]")
            request.path.append(Joints(list(joints)))
        if total_time is not None:
            request.time = total_time
        if total_time is None and timestamps is not None:
            if len(timestamps) != len(path):
                raise ValueError("timestamps length must be equal to path length")
            request.timestamp = list(timestamps)
        request.is_async = is_async

        return self._call_service(
            f"{self.prefix}/movej_by_path/{service_suffix}",
            MoveJByPath,
            request,
            "MoveJByPath",
        )

    def movejh_by_path(
        self,
        path: Sequence[Sequence[float]],
        arm_mask: int,
        total_time: Optional[float] = None,
        timestamps: Optional[Sequence[float]] = None,
        is_async: bool = False,
    ) -> bool:
        if not isinstance(arm_mask, int):
            raise TypeError("arm_mask must be int (8421 code)")
        if len(path) < 2:
            raise ValueError("path length must be greater than 1")
        if total_time is None and timestamps is None:
            raise ValueError("total_time and timestamps cannot both be None")

        expected_len = self._whole_body_expected_len(arm_mask)

        request = MoveJByPathRequest()
        request.arm_type = arm_mask
        for index, joints in enumerate(path):
            check_length(joints, expected_len, f"path[{index}]")
            request.path.append(Joints(list(joints)))
        if total_time is not None:
            request.time = total_time
        if total_time is None and timestamps is not None:
            if len(timestamps) != len(path):
                raise ValueError("timestamps length must be equal to path length")
            request.timestamp = list(timestamps)
        request.is_async = is_async

        return self._call_service(
            f"{self.prefix}/movej_by_path/whole_body",
            MoveJByPath,
            request,
            "MoveJWholeBodyByPath",
        )

    def _call_setbool_service(self, service_name: str, enable: bool, tag: str) -> bool:
        try:
            rospy.wait_for_service(service_name, timeout=5.0)
            client = rospy.ServiceProxy(service_name, SetBool)
            response = client(SetBoolRequest(data=enable))
            if response.success:
                state = "enabled" if enable else "disabled"
                rospy.loginfo(f"{tag} {state}")
            else:
                rospy.logerr(f"{tag} failed: {response.message}")
            return response.success
        except rospy.ROSException as exc:
            rospy.logerr(f"{tag} service not available: {exc}")
            return False
        except rospy.ServiceException as exc:
            rospy.logerr(f"{tag} service call failed: {exc}")
            return False

    def _call_service(self, service_name: str, service_type, request, tag: str) -> bool:
        try:
            rospy.wait_for_service(service_name, timeout=5.0)
            client = rospy.ServiceProxy(service_name, service_type)
            response = client(request)
            if hasattr(response, "success"):
                return response.success
            return bool(response)
        except rospy.ROSException as exc:
            rospy.logerr(f"{tag} service not available: {exc}")
            return False
        except rospy.ServiceException as exc:
            rospy.logerr(f"{tag} service call failed: {exc}")
            return False


class HandController:
    def __init__(self, prefix: str):
        self.prefix = prefix
        self.left_hand_joints = None
        self.right_hand_joints = None
        self.left_hand_pressures = None
        self.right_hand_pressures = None
        self.left_force = None
        self.right_force = None
        self._init_subscribers()
        rospy.loginfo("HandController initialized")

    def _init_subscribers(self) -> None:
        rospy.Subscriber(f"{self.prefix}/joint_states", JointState, self._joint_states_cb)
        rospy.Subscriber(f"{self.prefix}/finger_pressures/left", PressureSensor, self._left_pressure_cb)
        rospy.Subscriber(f"{self.prefix}/finger_pressures/right", PressureSensor, self._right_pressure_cb)
        rospy.Subscriber("/wrist_force_control/left_arm_compensated_force", WrenchStamped, self._left_force_cb)
        rospy.Subscriber("/wrist_force_control/right_arm_compensated_force", WrenchStamped, self._right_force_cb)

    def _joint_states_cb(self, msg: JointState) -> None:
        try:
            if len(msg.position) != 12:
                rospy.logwarn("hand joint_states length != 12")
                return
            self.left_hand_joints = list(msg.position[:6])
            self.right_hand_joints = list(msg.position[6:])
        except Exception as exc:
            rospy.logwarn(f"hand joint_states callback error: {exc}")

    def _left_pressure_cb(self, msg: PressureSensor) -> None:
        self.left_hand_pressures = list(msg.pressure)

    def _right_pressure_cb(self, msg: PressureSensor) -> None:
        self.right_hand_pressures = list(msg.pressure)

    def _left_force_cb(self, msg: WrenchStamped) -> None:
        self.left_force = msg.wrench.force

    def _right_force_cb(self, msg: WrenchStamped) -> None:
        self.right_force = msg.wrench.force

    def get_joints(self, hand: Union[HandType, int]):
        hand = as_hand_type(hand)
        if hand == HandType.LEFT:
            return self.left_hand_joints
        if hand == HandType.RIGHT:
            return self.right_hand_joints
        return None

    def get_pressures(self, hand: Union[HandType, int]):
        hand = as_hand_type(hand)
        if hand == HandType.LEFT:
            return self.left_hand_pressures
        if hand == HandType.RIGHT:
            return self.right_hand_pressures
        return None

    def get_force(self, hand: Union[HandType, int]):
        hand = as_hand_type(hand)
        if hand == HandType.LEFT:
            return self.left_force
        if hand == HandType.RIGHT:
            return self.right_force
        return None

    def grasp(self, hand: Union[HandType, int], joints: Sequence[float]) -> bool:
        return self.set_joints(hand, joints)

    def release(self, hand: Union[HandType, int]) -> bool:
        return self.set_joints(hand, [0.0] * 6)

    def set_joints(self, hand: Union[HandType, int], joints: Sequence[float]) -> bool:
        hand = as_hand_type(hand)
        check_length(joints, 6, "hand joints")

        side = "left" if hand == HandType.LEFT else "right"
        service = f"{self.prefix}/joint_switch/{side}"
        try:
            rospy.wait_for_service(service, timeout=5.0)
            client = rospy.ServiceProxy(service, HandJoint)
            response = client(HandJointRequest(q=list(joints)))
            return response.success
        except rospy.ROSException as exc:
            rospy.logerr(f"HandJoint service not available: {exc}")
            return False
        except rospy.ServiceException as exc:
            rospy.logerr(f"HandJoint service call failed: {exc}")
            return False


class NaviController:
    def __init__(self, auto_spin: bool = True, model: Union[RobotModel, str] = RobotModel.WA1):
        self.arm = ArmController("/zj_humanoid/upperlimb", model=model)
        self.hand = HandController("/zj_humanoid/hand")

        self._spin_thread = None
        if auto_spin:
            self._spin_thread = threading.Thread(target=rospy.spin, daemon=True)
            self._spin_thread.start()

    def get_tcp_rt(self, arm: Union[ArmGroup, int]):
        return self.arm.get_tcp_rt(arm)

    def get_tcp_matrix(self, arm: Union[ArmGroup, int]):
        return self.arm.get_tcp_matrix(arm)

    def get_joints(self, arm: Union[ArmGroup, int]):
        return self.arm.get_joints(arm)

    def get_tcp_speed(self, arm: Union[ArmGroup, int]):
        return self.arm.get_tcp_speed(arm)

    def get_robot_state(self):
        return self.arm.get_robot_state()

    def is_robot_moving(self) -> bool:
        return self.arm.is_robot_moving()

    def movej(
        self,
        joints: List[float],
        arm: Union[ArmGroup, int],
        v: float = 0.3,
        acc: float = 0.5,
        t: Optional[float] = None,
        is_async: bool = False,
    ) -> bool:
        return self.arm.movej(joints=joints, arm=arm, v=v, acc=acc, t=t, is_async=is_async)

    def movejh(
        self,
        joints: List[float],
        mask: int,
        v: float = 0.3,
        acc: float = 0.5,
        is_async: bool = False,
    ) -> bool:
        return self.arm.movejh(arm_mask=mask, joints=joints, v=v, acc=acc, is_async=is_async)

    def movej_by_path(
        self,
        path: List[List[float]],
        arm: Union[ArmGroup, int],
        total_time: Optional[float] = None,
        timestamps: Optional[List[float]] = None,
        is_async: bool = False,
    ) -> bool:
        return self.arm.movej_by_path(
            path=path,
            arm=arm,
            total_time=total_time,
            timestamps=timestamps,
            is_async=is_async,
        )

    def movejh_by_path(
        self,
        path: List[List[float]],
        arm_mask: int,
        total_time: Optional[float] = None,
        timestamps: Optional[List[float]] = None,
        is_async: bool = False,
    ) -> bool:
        return self.arm.movejh_by_path(
            path=path,
            arm_mask=arm_mask,
            total_time=total_time,
            timestamps=timestamps,
            is_async=is_async,
        )

    def movel(
        self,
        pose: List[float],
        arm: Union[ArmGroup, int],
        v: float = 0.1,
        acc: float = 0.1,
        is_async: bool = False,
    ) -> bool:
        return self.arm.movel(pose=pose, arm=arm, v=v, acc=acc, is_async=is_async)

    def movel_relative_base(
        self,
        delta_xyz: List[float],
        arm: Union[ArmGroup, int],
        v: float = 0.1,
        acc: float = 0.1,
        is_async: bool = False,
    ) -> bool:
        return self.arm.movel_relative_base(delta_xyz, arm, v, acc, is_async)

    def movel_relative_eef(
        self,
        transform: Union[np.ndarray, Sequence[Sequence[float]]],
        arm: Union[ArmGroup, int],
        v: float = 0.1,
        acc: float = 0.1,
        is_async: bool = False,
    ) -> bool:
        return self.arm.movel_relative_eef(transform, arm, v, acc, is_async)

    def enable_speedj(self, enable: bool = True) -> bool:
        return self.arm.enable_speedj(enable)

    def stop_speedj(self, arm: Union[ArmGroup, int]):
        return self.arm.stop_speedj(arm)

    def speedj(
        self,
        joint_speed: List[float],
        arm: Union[ArmGroup, int],
        acc: float = 0.05,
    ):
        return self.arm.speedj(joint_speed, arm, acc)

    def enable_speedl(self, enable: bool = True) -> bool:
        return self.arm.enable_speedl(enable)

    def stop_speedl(self, arm: Union[ArmGroup, int]):
        return self.arm.stop_speedl(arm)

    def speedl(
        self,
        tcp_speed: List[float],
        arm: Union[ArmGroup, int],
        acc: float = 0.05,
    ):
        return self.arm.speedl(tcp_speed, arm, acc)

    def set_servo_params(self, time_sec: float, gain: int) -> bool:
        return self.arm.set_servo_params(time_sec, gain)

    def clear_servo_params(self) -> bool:
        return self.arm.clear_servo_params()

    def servoj_dual_arm(self, joints: List[float]) -> None:
        return self.arm.servoj_dual_arm(joints)

    def get_hand_joints(self, hand: Union[HandType, int]):
        return self.hand.get_joints(hand)

    def get_hand_pressures(self, hand: Union[HandType, int]):
        return self.hand.get_pressures(hand)

    def grasp_hand(self, hand: Union[HandType, int], joints: Sequence[float]):
        return self.hand.grasp(hand, joints)

    def release_hand(self, hand: Union[HandType, int]):
        return self.hand.release(hand)

    def get_hand_force(self, hand: Union[HandType, int]):
        return self.hand.get_force(hand)


def main() -> None:
    rospy.init_node("navi_controller_single_test")
    controller = NaviController()
    rospy.sleep(1.0)
    print("left joints:", controller.get_joints(ArmGroup.LEFT))
    print("right joints:", controller.get_joints(ArmGroup.RIGHT))
    print("left tcp:", controller.get_tcp_rt(ArmGroup.LEFT))
    print("right tcp:", controller.get_tcp_rt(ArmGroup.RIGHT))


if __name__ == "__main__":
    main()
