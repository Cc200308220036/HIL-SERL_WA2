import numpy as np
from typing import Sequence
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import Pose, Point, Quaternion



def rt_to_matrix(rt: Sequence[float]) -> np.ndarray:
    """
    [x, y, z, qx, qy, qz, qw] -> 4x4 SE(3) matrix
    """
    if len(rt) != 7:
        raise ValueError("rt must be [x,y,z,qx,qy,qz,qw]")

    T = np.eye(4)
    T[:3, :3] = R.from_quat(rt[3:]).as_matrix()
    T[:3, 3] = rt[:3]
    return T


def matrix_to_rt(T: np.ndarray) -> list:
    """
    4x4 SE(3) matrix -> [x, y, z, qx, qy, qz, qw]
    """
    if T.shape != (4, 4):
        raise ValueError("T must be 4x4 matrix")

    pos = T[:3, 3]
    quat = R.from_matrix(T[:3, :3]).as_quat()
    return list(pos) + list(quat)


def list_to_pose(rt: Sequence[float]) -> Pose:
    """
    [x, y, z, qx, qy, qz, qw] -> geometry_msgs/Pose
    """
    if len(rt) != 7:
        raise ValueError("rt must be [x,y,z,qx,qy,qz,qw]")

    pose = Pose()
    pose.position = Point(*rt[:3])
    pose.orientation = Quaternion(*rt[3:])
    return pose


def pose_to_rt(pose: Pose) -> list:
    """
    geometry_msgs/Pose -> [x, y, z, qx, qy, qz, qw]
    """
    return [
        pose.position.x,
        pose.position.y,
        pose.position.z,
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
    ]


def check_length(data: Sequence[float], length: int, name: str) -> bool:
    """
    Check if the length of the data is equal to the length.
    """
    if len(data) != length:
        raise ValueError(f"{name} length must be {length}, but got {len(data)}")
    return True