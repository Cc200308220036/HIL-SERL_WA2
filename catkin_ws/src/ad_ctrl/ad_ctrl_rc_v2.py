#!/usr/bin/env python3
"""导纳控制 v2。

本文件在保留 v1 控制算法的基础上，提供面向调用方的单文件控制接口。
调用方只需要设置模式、期望速度和期望力，不需要直接操作 ROS topic。

示例:
    rospy.init_node("ad_ctrl_client")
    controller = AdmittanceController()
    controller.set_mode(AdmittanceMode.SINGLE, active_arm=ArmGroup.RIGHT)
    controller.set_velocity(ArmGroup.RIGHT, [0.01, 0, 0, 0, 0, 0])
    controller.set_target_force(ArmGroup.RIGHT, [-6, 0, 0])
    controller.start()
    rospy.sleep(3.0)
    controller.stop()
"""

import copy
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence

import numpy as np
import rospy
from geometry_msgs.msg import WrenchStamped
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from std_srvs.srv import SetBool, SetBoolRequest
from upperlimb.msg import Pose, SpeedL

from naviai_controller import ArmGroup


@dataclass
class AdmittanceConfig:
    """导纳控制配置。

    属性:
        frequency: 控制循环频率，单位为 Hz。
        force_deadband: 三维力死区，单位为 N。
        torque_deadband: 三维力矩死区，单位为 N·m。
        sensor_force_ratio: 力输入缩放系数。
        sensor_torque_ratio: 力矩输入缩放系数。
        left_mass/right_mass: 左右臂导纳虚拟质量。
        left_damping/right_damping: 左右臂导纳阻尼系数。
        speed_ratio: 输出速度缩放系数。
        max_force_norm: 力输入安全上限，None 表示不额外限制。
        max_linear_speed/max_angular_speed: TCP 线速度和角速度上限。
        state_timeout: 力传感器和关节状态允许的最大数据延迟。
        robot_model: 机器人型号，用于确定 joint_states 中的腰部索引。
        waist_joint_indices: 可选的腰部 yaw/pitch 索引；用于覆盖型号默认值。
    """

    frequency: float = 125.0
    force_deadband: float = 6.0
    torque_deadband: float = 0.2
    sensor_force_ratio: float = 0.1
    sensor_torque_ratio: float = 0.1
    left_mass: float = 3.0
    right_mass: float = 3.0
    left_damping: float = 40.0
    right_damping: float = 40.0
    speed_ratio: float = 1.0
    speed_acceleration: float = 0.3
    resample_time: float = 0.01
    max_force_norm: Optional[float] = None
    max_linear_speed: Optional[float] = None
    max_angular_speed: Optional[float] = None
    state_timeout: Optional[float] = 0.2
    robot_model: str = "wa1"
    waist_joint_indices: Optional[Sequence[int]] = None
    left_force_length: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    right_force_length: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    left_weight: float = 0.0
    right_weight: float = 0.0
    namespace: str = "/zj_humanoid/upperlimb"
    force_topic_prefix: str = "/wrist_force_control"

    def __post_init__(self) -> None:
        """校验并标准化配置参数。"""
        if self.frequency <= 0:
            raise ValueError("frequency must be greater than zero")
        if self.resample_time <= 0:
            raise ValueError("resample_time must be greater than zero")
        if self.left_mass <= 0 or self.right_mass <= 0:
            raise ValueError("virtual mass must be greater than zero")
        if self.left_damping < 0 or self.right_damping < 0:
            raise ValueError("damping must not be negative")
        for name, value in (
            ("max_force_norm", self.max_force_norm),
            ("max_linear_speed", self.max_linear_speed),
            ("max_angular_speed", self.max_angular_speed),
            ("state_timeout", self.state_timeout),
        ):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be greater than zero or None")

        self.robot_model = str(self.robot_model).lower()
        if self.robot_model not in ("wa1", "wa2"):
            raise ValueError("robot_model must be 'wa1' or 'wa2'")

        # joint_states 排列：
        # WA1=[左7, 右7, 颈2, 腰2, 升降1]
        # WA2=[左8, 右8, 颈2, 腰4]
        # 默认使用腰部的前两个关节作为 yaw、pitch；若实际 SDK 排列不同，
        # 调用方可通过 waist_joint_indices 显式覆盖。
        if self.waist_joint_indices is None:
            self.waist_joint_indices = (16, 17) if self.robot_model == "wa1" else (18, 19)
        else:
            indices = tuple(self.waist_joint_indices)
            if len(indices) != 2 or any(
                not isinstance(index, (int, np.integer)) or index < 0
                for index in indices
            ):
                raise ValueError(
                    "waist_joint_indices must contain two non-negative integers"
                )
            self.waist_joint_indices = (int(indices[0]), int(indices[1]))
        self.left_force_length = _as_vector(
            self.left_force_length, 3, "left_force_length"
        )
        self.right_force_length = _as_vector(
            self.right_force_length, 3, "right_force_length"
        )


class AdmittanceMode(Enum):
    """导纳运行模式。"""

    SINGLE = "single"
    DUAL = "dual"
    MASTER_SLAVE = "master_slave"


def _as_vector(
    value: Sequence[float], size: int, name: str
) -> np.ndarray:
    """将输入转换为指定长度的浮点向量。

    参数:
        value: 输入序列。
        size: 期望的向量长度。
        name: 参数名称，用于生成错误信息。

    返回:
        np.ndarray: 独立的浮点向量副本。

    异常:
        ValueError: 输入长度不正确时抛出。
    """
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.size != size:
        raise ValueError(f"{name} must contain {size} values")
    return vector.copy()


class _QuinticResampler:
    """为六维速度指令提供平滑的五次时间插值。"""

    def __init__(self, duration: float) -> None:
        """初始化插值器。

        参数:
            duration: 每次指令切换的插值时长，单位为秒。
        """
        self._duration = duration
        self._start = np.zeros(6, dtype=np.float64)
        self._target = np.zeros(6, dtype=np.float64)
        self._start_time = time.monotonic()
        self._lock = threading.RLock()

    def set_target(self, target: Sequence[float]) -> None:
        """设置新的六维速度目标。

        参数:
            target: 六维笛卡尔速度，前三维为线速度，后三维为角速度。
        """
        target_vector = _as_vector(target, 6, "velocity")
        with self._lock:
            self._start = self.value()
            self._target = target_vector
            self._start_time = time.monotonic()

    def value(self) -> np.ndarray:
        """获取当前插值速度。

        返回:
            np.ndarray: 当前插值时刻的六维速度。
        """
        with self._lock:
            elapsed = time.monotonic() - self._start_time
            if elapsed >= self._duration:
                return self._target.copy()
            ratio = max(0.0, elapsed / self._duration)
            scale = 6 * ratio**5 - 15 * ratio**4 + 10 * ratio**3
            return self._start + scale * (self._target - self._start)


class _AdmittanceModel:
    """六维导纳离散计算器。"""

    def __init__(self, mass: float, damping: float, delta_t: float) -> None:
        """初始化导纳计算器。

        参数:
            mass: 虚拟质量。
            damping: 虚拟阻尼。
            delta_t: 离散控制周期，单位为秒。
        """
        self._mass = mass
        self._damping = damping
        self._delta_t = delta_t
        self._last_velocity = np.zeros(6, dtype=np.float64)

    def reset(self) -> None:
        """清零上一周期速度状态。"""
        self._last_velocity.fill(0.0)

    def step(
        self,
        desired_velocity: Sequence[float],
        external_force: Sequence[float],
        waist_rotation: np.ndarray,
    ) -> np.ndarray:
        """计算一个控制周期的导纳速度。

        参数:
            desired_velocity: 六维期望速度。
            external_force: 六维外力/外力矩。
            waist_rotation: 当前腰部坐标系旋转矩阵。

        返回:
            np.ndarray: 六维导纳输出速度。
        """
        desired = _as_vector(desired_velocity, 6, "desired_velocity")
        force = _as_vector(external_force, 6, "external_force")
        rotation_6d = np.zeros((6, 6), dtype=np.float64)
        rotation_6d[:3, :3] = waist_rotation
        rotation_6d[3:, 3:] = waist_rotation
        desired_in_waist = rotation_6d @ desired
        acceleration = (
            force - self._damping * (self._last_velocity - desired_in_waist)
        ) / self._mass
        self._last_velocity += acceleration * self._delta_t
        return np.linalg.inv(rotation_6d) @ self._last_velocity


class AdmittanceController:
    """导纳控制高层接口。

    该类封装力传感器订阅、腰部姿态更新、导纳计算、速度发布和安全停止。
    调用方不需要直接操作导纳相关的 ROS topic。

    示例:
        controller = AdmittanceController()
        controller.set_mode(AdmittanceMode.SINGLE, ArmGroup.RIGHT)
        controller.set_velocity(ArmGroup.RIGHT, [0.01, 0, 0, 0, 0, 0])
        controller.set_target_force(ArmGroup.RIGHT, [-6, 0, 0])
        controller.start()
        controller.stop()
    """

    def __init__(self, config: Optional[AdmittanceConfig] = None) -> None:
        """初始化导纳控制器，但不自动启动运动。

        参数:
            config: 导纳配置；未提供时使用 v1 的默认参数。

        异常:
            RuntimeError: ROS 节点尚未初始化时抛出。
        """
        if not rospy.core.is_initialized():
            raise RuntimeError("rospy node must be initialized before construction")

        self.config = config or AdmittanceConfig()
        self._lock = threading.RLock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._mode = AdmittanceMode.SINGLE
        self._active_arm = ArmGroup.LEFT
        self._force: Dict[ArmGroup, Optional[np.ndarray]] = {
            ArmGroup.LEFT: None,
            ArmGroup.RIGHT: None,
        }
        self._force_update = {
            ArmGroup.LEFT: False,
            ArmGroup.RIGHT: False,
        }
        self._force_stamp: Dict[ArmGroup, Optional[float]] = {
            ArmGroup.LEFT: None,
            ArmGroup.RIGHT: None,
        }
        self._waist_rotation = np.eye(3, dtype=np.float64)
        self._joint_state_stamp: Optional[float] = None
        self._tcp_rotation = {
            ArmGroup.LEFT: np.eye(3, dtype=np.float64),
            ArmGroup.RIGHT: np.eye(3, dtype=np.float64),
        }
        self._desired_force = {
            ArmGroup.LEFT: np.zeros(3, dtype=np.float64),
            ArmGroup.RIGHT: np.zeros(3, dtype=np.float64),
        }
        self._desired_velocity = {
            ArmGroup.LEFT: _QuinticResampler(self.config.resample_time),
            ArmGroup.RIGHT: _QuinticResampler(self.config.resample_time),
        }
        delta_t = 1.0 / self.config.frequency
        self._models = {
            ArmGroup.LEFT: _AdmittanceModel(
                self.config.left_mass, self.config.left_damping, delta_t
            ),
            ArmGroup.RIGHT: _AdmittanceModel(
                self.config.right_mass, self.config.right_damping, delta_t
            ),
        }

        self._speed_publisher = rospy.Publisher(
            f"{self.config.namespace}/speedl/dual_arm",
            SpeedL,
            queue_size=10,
        )
        self._enable_speedl_service = f"{self.config.namespace}/enable_speedl"
        self._subscribers = [
            rospy.Subscriber(
                f"{self.config.force_topic_prefix}/left_arm_compensated_force",
                WrenchStamped,
                self._left_force_callback,
                queue_size=10,
                tcp_nodelay=True,
            ),
            rospy.Subscriber(
                f"{self.config.force_topic_prefix}/right_arm_compensated_force",
                WrenchStamped,
                self._right_force_callback,
                queue_size=10,
                tcp_nodelay=True,
            ),
            rospy.Subscriber(
                f"{self.config.namespace}/joint_states",
                JointState,
                self._joint_state_callback,
                queue_size=10,
            ),
            rospy.Subscriber(
                f"{self.config.namespace}/tcp_pose/left_arm",
                Pose,
                self._left_tcp_callback,
                queue_size=10,
            ),
            rospy.Subscriber(
                f"{self.config.namespace}/tcp_pose/right_arm",
                Pose,
                self._right_tcp_callback,
                queue_size=10,
            ),
        ]

    def set_mode(
        self,
        mode: AdmittanceMode,
        active_arm: ArmGroup = ArmGroup.LEFT,
    ) -> None:
        """设置导纳模式。

        参数:
            mode: 单臂、双臂或主从模式。
            active_arm: 单臂模式的控制臂，或主从模式的主臂。

        异常:
            ValueError: 模式或手臂类型不正确时抛出。
        """
        if not isinstance(mode, AdmittanceMode):
            raise ValueError("mode must be an AdmittanceMode")
        self._validate_arm(active_arm)
        with self._lock:
            if self._running:
                raise RuntimeError("cannot change mode while controller is running")
            self._mode = mode
            self._active_arm = active_arm

    def set_velocity(self, arm: ArmGroup, velocity: Sequence[float]) -> None:
        """设置指定手臂的六维期望速度。

        参数:
            arm: 左臂或右臂。
            velocity: `[vx, vy, vz, wx, wy, wz]`，单位分别为 m/s 和 rad/s。
        """
        self._validate_arm(arm)
        self._desired_velocity[arm].set_target(velocity)

    def set_target_force(self, arm: ArmGroup, force: Sequence[float]) -> None:
        """设置指定手臂的三维期望力。

        参数:
            arm: 左臂或右臂。
            force: `[fx, fy, fz]`，单位为 N。
        """
        self._validate_arm(arm)
        with self._lock:
            self._desired_force[arm] = _as_vector(force, 3, "force")

    def get_force(self, arm: ArmGroup) -> Optional[np.ndarray]:
        """获取最近一次收到的六维力/力矩数据。

        参数:
            arm: 左臂或右臂。

        返回:
            Optional[np.ndarray]: `[fx, fy, fz, tx, ty, tz]`；尚未收到数据时为 None。
        """
        self._validate_arm(arm)
        with self._lock:
            value = self._force[arm]
            return None if value is None else value.copy()

    def wait_until_ready(self, timeout: float = 5.0) -> bool:
        """等待所需手臂的力传感器数据就绪。

        参数:
            timeout: 最大等待时间，单位为秒。

        返回:
            bool: 在超时前收到所需数据则返回 True。
        """
        deadline = time.monotonic() + timeout
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            with self._lock:
                now = time.monotonic()
                force_ready = all(
                    self._force[arm] is not None
                    and self._force_stamp[arm] is not None
                    and (
                        self.config.state_timeout is None
                        or now - self._force_stamp[arm] <= self.config.state_timeout
                    )
                    for arm in self._required_arms()
                )
                joint_state_ready = (
                    self._joint_state_stamp is not None
                    and (
                        self.config.state_timeout is None
                        or now - self._joint_state_stamp <= self.config.state_timeout
                    )
                )
                if force_ready and joint_state_ready:
                    return True
            rospy.sleep(0.01)
        return False

    def start(self, wait_for_force: bool = True, timeout: float = 5.0) -> None:
        """启动导纳控制循环。

        参数:
            wait_for_force: 是否先等待力传感器数据。
            timeout: 等待力传感器的最大时间，单位为秒。

        异常:
            RuntimeError: 传感器未就绪或底层 speedl 未能使能时抛出。
        """
        with self._lock:
            if self._running:
                return
        if wait_for_force and not self.wait_until_ready(timeout):
            raise RuntimeError("required force sensor data is not ready")
        if not self._enable_speedl(True):
            raise RuntimeError("failed to enable speedl")
        with self._lock:
            for model in self._models.values():
                model.reset()
            self._running = True
            self._thread = threading.Thread(
                target=self._control_loop,
                name="admittance-control",
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        """安全停止导纳控制并关闭底层 speedl。"""
        with self._lock:
            self._running = False
            thread = self._thread
            self._thread = None
        self._publish_zero()
        self._enable_speedl(False)
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)

    def is_running(self) -> bool:
        """返回导纳控制是否正在运行。"""
        with self._lock:
            return self._running

    def __enter__(self) -> "AdmittanceController":
        """进入上下文时启动导纳控制。"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """离开上下文时安全停止导纳控制。"""
        self.stop()

    def _required_arms(self) -> List[ArmGroup]:
        """返回当前模式需要的力传感器列表。"""
        if self._mode is AdmittanceMode.SINGLE:
            return [self._active_arm]
        return [ArmGroup.LEFT, ArmGroup.RIGHT]

    @staticmethod
    def _validate_arm(arm: ArmGroup) -> None:
        """校验手臂枚举值。"""
        if arm not in (ArmGroup.LEFT, ArmGroup.RIGHT):
            raise ValueError("only left and right arms are supported")

    def _left_force_callback(self, message: WrenchStamped) -> None:
        """接收左腕力传感器数据。"""
        self._update_force(ArmGroup.LEFT, message)

    def _right_force_callback(self, message: WrenchStamped) -> None:
        """接收右腕力传感器数据。"""
        self._update_force(ArmGroup.RIGHT, message)

    def _update_force(self, arm: ArmGroup, message: WrenchStamped) -> None:
        """保存一帧力传感器数据并清除无效的垂直分量。

        参数:
            arm: 数据所属手臂。
            message: ROS 力传感器消息。
        """
        copied = copy.deepcopy(message)
        copied.wrench.force.z = 0.0
        value = np.array(
            [
                copied.wrench.force.x,
                copied.wrench.force.y,
                copied.wrench.force.z,
                copied.wrench.torque.x,
                copied.wrench.torque.y,
                copied.wrench.torque.z,
            ],
            dtype=np.float64,
        )
        with self._lock:
            self._force[arm] = value
            self._force_update[arm] = True
            self._force_stamp[arm] = time.monotonic()

    def _joint_state_callback(self, message: JointState) -> None:
        """根据腰部关节角更新力坐标变换矩阵。"""
        yaw_index, pitch_index = self.config.waist_joint_indices
        if len(message.position) <= max(yaw_index, pitch_index):
            rospy.logwarn_throttle(
                5.0,
                "joint_states length is too short for waist indices %s",
                self.config.waist_joint_indices,
            )
            return
        waist_z = message.position[yaw_index]
        waist_y = message.position[pitch_index]
        rotation = Rotation.from_euler("zy", [waist_z, waist_y]).as_matrix()
        with self._lock:
            self._waist_rotation = rotation
            self._joint_state_stamp = time.monotonic()

    def _left_tcp_callback(self, message: Pose) -> None:
        """保存左臂 TCP 姿态，用于重力力矩补偿。"""
        self._update_tcp_rotation(ArmGroup.LEFT, message)

    def _right_tcp_callback(self, message: Pose) -> None:
        """保存右臂 TCP 姿态，用于重力力矩补偿。"""
        self._update_tcp_rotation(ArmGroup.RIGHT, message)

    def _update_tcp_rotation(self, arm: ArmGroup, message: Pose) -> None:
        """根据 TCP 四元数更新指定手臂的姿态矩阵。"""
        quaternion = [
            message.quaternion.x,
            message.quaternion.y,
            message.quaternion.z,
            message.quaternion.w,
        ]
        rotation = Rotation.from_quat(quaternion).as_matrix()
        with self._lock:
            self._tcp_rotation[arm] = rotation

    def _control_loop(self) -> None:
        """执行导纳控制主循环，并负责异常后的安全收尾。"""
        rate = rospy.Rate(self.config.frequency)
        try:
            while not rospy.is_shutdown() and self.is_running():
                self._run_once()
                rate.sleep()
        except Exception as error:
            rospy.logerr("admittance control stopped: %s", error)
        finally:
            with self._lock:
                self._running = False
            self._publish_zero()
            self._enable_speedl(False)

    def _run_once(self) -> None:
        """执行一个控制周期并发布双臂速度。"""
        with self._lock:
            required_arms = self._required_arms()
            if any(self._force[arm] is None for arm in required_arms):
                return
            mode = self._mode
            active_arm = self._active_arm
            forces = {arm: self._force[arm].copy() for arm in required_arms}
            desired_forces = {
                arm: self._desired_force[arm].copy() for arm in required_arms
            }
            rotations = {
                arm: self._tcp_rotation[arm].copy() for arm in required_arms
            }
            waist_rotation = self._waist_rotation.copy()
            force_stamps = {
                arm: self._force_stamp[arm] for arm in required_arms
            }
            joint_state_stamp = self._joint_state_stamp

        now = time.monotonic()
        if self.config.state_timeout is not None:
            for arm, stamp in force_stamps.items():
                if stamp is None or now - stamp > self.config.state_timeout:
                    raise RuntimeError(f"{arm.name} force data timed out")
            if (
                joint_state_stamp is None
                or now - joint_state_stamp > self.config.state_timeout
            ):
                raise RuntimeError("joint state data timed out")

        desired_velocity = {
            ArmGroup.LEFT: self._desired_velocity[ArmGroup.LEFT].value(),
            ArmGroup.RIGHT: self._desired_velocity[ArmGroup.RIGHT].value(),
        }
        output_velocity = {
            ArmGroup.LEFT: desired_velocity[ArmGroup.LEFT],
            ArmGroup.RIGHT: desired_velocity[ArmGroup.RIGHT],
        }
        for arm in required_arms:
            adjusted_force = forces[arm].copy()
            adjusted_force[:3] += desired_forces[arm]
            force_length = (
                self.config.left_force_length
                if arm is ArmGroup.LEFT
                else self.config.right_force_length
            )
            weight = (
                self.config.left_weight
                if arm is ArmGroup.LEFT
                else self.config.right_weight
            )
            gravity = np.array([0.0, 0.0, -weight * 9.81])
            adjusted_force[3:] -= np.cross(rotations[arm] @ force_length, gravity)
            self._apply_deadband(adjusted_force[:3], self.config.force_deadband)
            self._apply_deadband(adjusted_force[3:], self.config.torque_deadband)
            if (
                self.config.max_force_norm is not None
                and np.linalg.norm(adjusted_force[:3]) > self.config.max_force_norm
            ):
                raise RuntimeError(f"{arm.name} force limit exceeded")
            adjusted_force[:3] *= self.config.sensor_force_ratio
            adjusted_force[3:] *= self.config.sensor_torque_ratio
            output_velocity[arm] = self._models[arm].step(
                desired_velocity[arm], adjusted_force, waist_rotation
            )

        if mode is AdmittanceMode.SINGLE:
            if active_arm is ArmGroup.LEFT:
                output_velocity[ArmGroup.RIGHT] = desired_velocity[ArmGroup.RIGHT]
            else:
                output_velocity[ArmGroup.LEFT] = desired_velocity[ArmGroup.LEFT]
        elif mode is AdmittanceMode.MASTER_SLAVE:
            master_velocity = output_velocity[active_arm]
            output_velocity[ArmGroup.LEFT] = master_velocity
            output_velocity[ArmGroup.RIGHT] = master_velocity

        self._publish_speed(
            self.config.speed_ratio * output_velocity[ArmGroup.LEFT],
            self.config.speed_ratio * output_velocity[ArmGroup.RIGHT],
        )

    @staticmethod
    def _apply_deadband(values: np.ndarray, threshold: float) -> None:
        """对数组原地应用死区补偿。"""
        mask = np.abs(values) < threshold
        values[mask] = 0.0
        values[~mask] -= np.sign(values[~mask]) * threshold

    def _publish_speed(
        self, left_velocity: Sequence[float], right_velocity: Sequence[float]
    ) -> None:
        """发布双臂六维速度指令。"""
        left = _as_vector(left_velocity, 6, "left_velocity")
        right = _as_vector(right_velocity, 6, "right_velocity")
        left = self._limit_velocity(left, "LEFT")
        right = self._limit_velocity(right, "RIGHT")
        message = SpeedL()
        message.tcp_speed = [
            0.0 if abs(value) < 1e-5 else float(value)
            for value in np.concatenate((left, right))
        ]
        message.acc = self.config.speed_acceleration
        self._speed_publisher.publish(message)

    def _limit_velocity(self, velocity: np.ndarray, arm_name: str) -> np.ndarray:
        """分别按向量模长限制 TCP 线速度和角速度，方向保持不变。"""
        limited = velocity.copy()
        clipped = False
        for part, limit in (
            (limited[:3], self.config.max_linear_speed),
            (limited[3:], self.config.max_angular_speed),
        ):
            norm = float(np.linalg.norm(part))
            if limit is not None and norm > limit:
                part *= limit / norm
                clipped = True
        if clipped:
            rospy.logwarn_throttle(1.0, "%s admittance velocity was limited", arm_name)
        return limited

    def _publish_zero(self) -> None:
        """发布零速度，确保停止时机械臂不继续运动。"""
        self._publish_speed(np.zeros(6), np.zeros(6))

    def _enable_speedl(self, enable: bool) -> bool:
        """调用底层 speedl 使能服务。"""
        try:
            rospy.wait_for_service(self._enable_speedl_service, timeout=5.0)
            client = rospy.ServiceProxy(self._enable_speedl_service, SetBool)
            response = client(SetBoolRequest(data=enable))
            return bool(response.success)
        except (rospy.ROSException, rospy.ServiceException) as error:
            rospy.logerr("speedl enable request failed: %s", error)
            return False


def main() -> None:
    """启动 v2 导纳节点并等待外部调用控制接口。"""
    rospy.init_node("ad_ctrl_rc_v2", anonymous=True)
    controller = AdmittanceController()
    rospy.on_shutdown(controller.stop)
    rospy.loginfo("ad_ctrl_rc_v2 is ready")
    rospy.spin()


if __name__ == "__main__":
    main()
