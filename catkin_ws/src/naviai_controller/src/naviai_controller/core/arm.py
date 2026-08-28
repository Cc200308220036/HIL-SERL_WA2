import time
import rospy
import numpy as np
from typing import List, Optional, Sequence, Union
from std_srvs.srv import SetBool, SetBoolRequest, Trigger, TriggerRequest
from threading import Lock
from sensor_msgs.msg import JointState
import sys
from upperlimb.msg import Pose as UplimbPose
from upperlimb.msg import TcpSpeed, UplimbState, Joints, DualPose
from upperlimb.srv import MoveJ, MoveJRequest, MoveL, MoveLRequest, Servo, ServoRequest, IK, IKRequest, MoveJByPath, MoveJByPathRequest

from upperlimb.msg import SpeedJ, SpeedL

from .enums import (
    ARM_SERVICE_NAMES,
    JOINT_DOF_BY_MODEL,
    JOINT_LAYOUT_BY_MODEL,
    L_GROUP,
    TCP_POSE_DOF,
    TCP_SPEED_DOF,
    ArmGroup,
    as_arm_group,
    as_robot_model,
)
from .tools import list_to_pose, rt_to_matrix, matrix_to_rt, check_length

class ArmController:
    def __init__(self, prefix: str, model="wa1"):
        self.prefix = prefix
        self.model = as_robot_model(model)
        self.joint_dof = JOINT_DOF_BY_MODEL[self.model]
        self.joint_layout = JOINT_LAYOUT_BY_MODEL[self.model]
        # ---------- state cache ----------
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

        self._left_tcp_stamp = None
        self._right_tcp_stamp = None
        self._cmd_state_stamp = None

        self._cmd_state = None

        # ---------- init IO ----------
        self._init_publishers()   
        time.sleep(1)
        self._init_subscribers()
        time.sleep(1)
        rospy.loginfo(f"ArmController初始化完成: {self.model.value}")


    def _init_subscribers(self):
        rospy.Subscriber(
            f"{self.prefix}/joint_states",
            JointState,
            self._joint_states_cb,
        )

        rospy.Subscriber(
            f"{self.prefix}/uplimb_state",
            UplimbState,
            self._cmd_state_cb,
        )

        rospy.Subscriber(
            f"{self.prefix}/tcp_pose/left_arm",
            UplimbPose,
            self._left_tcp_cb,
        )

        rospy.Subscriber(
            f"{self.prefix}/tcp_pose/right_arm",
            UplimbPose,
            self._right_tcp_cb,
        )

        rospy.Subscriber(
            f"{self.prefix}/tcp_speed/dual_arm",
            TcpSpeed,
            self._tcp_speed_cb,
        )

    def _init_publishers(self):
            # ---------- Speed publishers ----------
        self._speedl_pubs = {
            ArmGroup.LEFT:  rospy.Publisher(f"{self.prefix}/speedl/{ARM_SERVICE_NAMES[ArmGroup.LEFT]}",  SpeedL, queue_size=1),
            ArmGroup.RIGHT: rospy.Publisher(f"{self.prefix}/speedl/{ARM_SERVICE_NAMES[ArmGroup.RIGHT]}", SpeedL, queue_size=1),
            ArmGroup.DUAL:  rospy.Publisher(f"{self.prefix}/speedl/{ARM_SERVICE_NAMES[ArmGroup.DUAL]}",  SpeedL, queue_size=1),
        }

        self._speedj_pubs = {
            group: rospy.Publisher(
                f"{self.prefix}/speedj/{ARM_SERVICE_NAMES[group]}",
                SpeedJ,
                queue_size=1,
            )
            for group in self.joint_dof
        }

            # ---------- ServoJ dual-arm publisher ----------
        self._servoj_dual_pub = rospy.Publisher(
            f"{self.prefix}/servoj/dual_arm",
            Joints,
            queue_size=1
        )

        # ---------- ServoL dual-arm publisher ----------
        self._servol_pubs = {
            ArmGroup.LEFT: rospy.Publisher(
                f"{self.prefix}/servol/{ARM_SERVICE_NAMES[ArmGroup.LEFT]}",
                DualPose,
                queue_size=1,
            ),
            ArmGroup.RIGHT: rospy.Publisher(
                f"{self.prefix}/servol/{ARM_SERVICE_NAMES[ArmGroup.RIGHT]}",
                DualPose,
                queue_size=1,
            ),
            ArmGroup.DUAL: rospy.Publisher(
                f"{self.prefix}/servol/{ARM_SERVICE_NAMES[ArmGroup.DUAL]}",
                DualPose,
                queue_size=1,
            ),
        }

    def _joint_states_cb(self, msg: JointState):
        with self._lock:
            self._joint_states = msg
            q = list(msg.position)

            expected_len = sum(length for _, length in self.joint_layout)
            if len(q) < expected_len:
                return

            slices = {}
            start = 0
            for group, length in self.joint_layout:
                slices[group] = q[start:start + length]
                start += length

            self._left_joints = slices.get(ArmGroup.LEFT)
            self._right_joints = slices.get(ArmGroup.RIGHT)
            self._neck_joints = slices.get(ArmGroup.NECK)
            self._waist_joints = slices.get(ArmGroup.WAIST)
            self._lift_joint = slices.get(ArmGroup.LIFT)

    def _left_tcp_cb(self, msg: UplimbPose):
        with self._lock:
            self._left_tcp = [msg.position.x, msg.position.y, msg.position.z, msg.quaternion.x, msg.quaternion.y, msg.quaternion.z, msg.quaternion.w]
            self._left_tcp_stamp = time.monotonic()

    def _right_tcp_cb(self, msg: UplimbPose):
        with self._lock:
            self._right_tcp = [msg.position.x, msg.position.y, msg.position.z, msg.quaternion.x, msg.quaternion.y, msg.quaternion.z, msg.quaternion.w]
            self._right_tcp_stamp = time.monotonic()

    def _tcp_speed_cb(self, msg: TcpSpeed):
        with self._lock:
            self._tcp_speed = msg

    def _cmd_state_cb(self, msg: UplimbState):
        with self._lock:
            self._cmd_state = msg
            self._cmd_state_stamp = time.monotonic()

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

    # 获取关节角度
    def get_joints(self, arm):
        arm = as_arm_group(arm)
        with self._lock:
            if arm == ArmGroup.LEFT:
                return self._left_joints
            elif arm == ArmGroup.RIGHT:
                return self._right_joints
            elif arm == ArmGroup.DUAL:
                if self._left_joints is None or self._right_joints is None:
                    return None
                return self._left_joints + self._right_joints
            elif arm == ArmGroup.NECK:
                return self._neck_joints
            elif arm == ArmGroup.WAIST:
                return self._waist_joints
            elif arm == ArmGroup.LIFT:
                return self._lift_joint
        return None

    # 获取TCP姿态（返回副本，避免调用方原地修改污染订阅缓存）
    def get_tcp_rt(self, arm):
        arm = as_arm_group(arm)
        with self._lock:
            if arm == ArmGroup.LEFT:
                t = self._left_tcp
                return None if t is None else t.copy()
            elif arm == ArmGroup.RIGHT:
                t = self._right_tcp
                return None if t is None else t.copy()
        return None

    def get_tcp_age(self, arm) -> Optional[float]:
        """Return seconds since the latest TCP callback, or ``None``."""
        arm = as_arm_group(arm)
        with self._lock:
            if arm == ArmGroup.LEFT:
                stamp = self._left_tcp_stamp
            elif arm == ArmGroup.RIGHT:
                stamp = self._right_tcp_stamp
            else:
                raise ValueError("get_tcp_age supports LEFT or RIGHT only")
        return None if stamp is None else max(0.0, time.monotonic() - stamp)

    def get_uplimb_state_age(self) -> Optional[float]:
        """Return seconds since the latest UplimbState callback, or ``None``."""
        with self._lock:
            stamp = self._cmd_state_stamp
        return None if stamp is None else max(0.0, time.monotonic() - stamp)

    def get_is_singular(self, arm) -> Optional[bool]:
        """Return the SDK singular flag for one arm, or ``None`` if unavailable."""
        arm = as_arm_group(arm)
        field = {
            ArmGroup.LEFT: "left_arm_is_singular",
            ArmGroup.RIGHT: "right_arm_is_singular",
        }.get(arm)
        if field is None:
            raise ValueError("get_is_singular supports LEFT or RIGHT only")
        with self._lock:
            state = self._cmd_state
            if state is None or not hasattr(state, field):
                return None
            return bool(getattr(state, field))

    def get_cmd_num(self) -> Optional[int]:
        """Return the latest upper-limb command number, or ``None``."""
        with self._lock:
            if self._cmd_state is None or not hasattr(self._cmd_state, "cmd_num"):
                return None
            return int(self._cmd_state.cmd_num)
    
    def get_tcp_matrix(self, arm):
        tcp_rt = self.get_tcp_rt(arm)
        if tcp_rt is None:
            return None
        return rt_to_matrix(tcp_rt)

    def get_tcp_pose(self, arm):
        rt = self.get_tcp_rt(arm)
        if rt is None:
            return None
        return list_to_pose(rt)

    def get_tcp_speed(self, arm):
        arm = as_arm_group(arm)
        with self._lock:
            if self._tcp_speed is None:
                return None
            if arm == ArmGroup.LEFT:
                return list(self._tcp_speed.left_arm)
            elif arm == ArmGroup.RIGHT:
                return list(self._tcp_speed.right_arm)
        return None


    def movej(
        self,
        joints: Sequence[float],
        arm: ArmGroup,
        v: float = 0.3,
        acc: float = 0.5,
        t: Optional[float] = None,
        is_async: bool = False,
    ) -> bool:

        arm = as_arm_group(arm)
        service_suffix = self._joint_service(arm)
        dof = self._joint_dof(arm)
        check_length(joints, dof, "joints")

        service = f"{self.prefix}/movej/{service_suffix}"
        req = MoveJRequest()
        req.joints = list(joints)
        req.v = v
        req.acc = acc
        req.is_async = is_async
        if t is not None:
            req.t = t

        try:    
            rospy.wait_for_service(service)
            client = rospy.ServiceProxy(service, MoveJ)
            resp = client(req)
            # print("resp")
             
        except rospy.ROSException as e:
            rospy.logerr(f"MoveJ service not available: {e}")
            return False
        except rospy.ServiceException as e:
            rospy.logerr(f"MoveJ service call failed: {e}")
            return False

        return resp.success

    def movejh(
        self,
        arm_mask: int,
        joints: Sequence[float],
        v: float = 0.3,
        acc: float = 0.5,
        is_async: bool = False,
    ) -> bool:

        expected_len = self._whole_body_expected_len(arm_mask)

        if len(joints) != expected_len:
            raise ValueError(
                f"joint length mismatch: expect {expected_len}, got {len(joints)}"
            )

        service = f"{self.prefix}/movej/whole_body"
        req = MoveJRequest()
        req.arm_type = arm_mask     # 关键：仍然是 int
        req.joints = list(joints)
        req.v = v
        req.acc = acc
        req.is_async = is_async
        try:
            rospy.wait_for_service(service)
            client = rospy.ServiceProxy(service, MoveJ)
            resp = client(req)
        
        except rospy.ROSException as e:
            rospy.logerr(f"MoveJWholeBody service not available: {e}")
            return False
        except rospy.ServiceException as e:
            rospy.logerr(f"MoveJWholeBody service call failed: {e}")
            return False  

        return resp.success


    def movel(
        self,
        pose: Sequence[float],
        arm: ArmGroup,
        v: float = 0.1,
        acc: float = 0.1,
        is_async: bool = False,
    ) -> bool:

        arm = as_arm_group(arm)
        pose_list = []

        if arm in L_GROUP:
            name = ARM_SERVICE_NAMES[arm]
            check_length(pose, TCP_POSE_DOF[arm], f"{arm} pose")
            if arm == ArmGroup.DUAL:
                pose_list.append(list_to_pose(pose[:7])) # 左臂
                pose_list.append(list_to_pose(pose[7:])) # 右臂
            else:
                pose_list.append(list_to_pose(pose))
            service = f"{self.prefix}/movel/{name}"
        else:
            raise ValueError(f"Unsupported arm group: {arm}")

        req = MoveLRequest()
        for i, pose in enumerate(pose_list):
            req.pose[i] = pose
        req.v = v
        req.acc = acc
        req.is_async = is_async

        try:
            rospy.wait_for_service(service)
            client = rospy.ServiceProxy(service, MoveL)
            resp = client(req)
        except rospy.ROSException as e:
            rospy.logerr(f"MoveL service not available: {e}")
            return False
        except rospy.ServiceException as e:
            rospy.logerr(f"MoveL service call failed: {e}")
            return False

        return resp.success

    def movel_relative_base(
            self, 
            delta_xyz: Sequence[float], 
            arm: ArmGroup, 
            v: float = 0.1, 
            acc: float = 0.1, 
            is_async: bool = False) -> bool:
        """
        相对于基座坐标系移动
        """
        arm = as_arm_group(arm)
        if arm not in L_GROUP:
            raise ValueError(f"Unsupported arm group: {arm}")

        if arm == ArmGroup.DUAL:
            left_pose = self.get_tcp_rt(ArmGroup.LEFT)
            right_pose = self.get_tcp_rt(ArmGroup.RIGHT)
            if left_pose is None or right_pose is None:
                raise ValueError("left_pose or right_pose is None")
            if len(delta_xyz) < 6:
                raise ValueError("delta_xyz must have at least 6 elements [dx_left, dy_left, dz_left, dx_right, dy_right, dz_right]")
            left_pose[0:3] = [left_pose[i] + float(delta_xyz[i]) for i in range(3)]
            right_pose[0:3] = [right_pose[i] + float(delta_xyz[i + 3]) for i in range(3)]
            pose = left_pose + right_pose
        else:
            pose = self.get_tcp_rt(arm)
            if pose is None:
                raise ValueError("pose is None")
            if len(delta_xyz) != 3:
                raise ValueError("delta_xyz must have exactly 3 elements [dx, dy, dz]")
            pose[0:3] = [pose[i] + float(delta_xyz[i]) for i in range(3)]
        return self.movel(pose, arm, v, acc, is_async)

    def movel_relative_eef(
            self,
            transform: Union[np.ndarray, Sequence[Sequence[float]]],
            arm: ArmGroup,
            v: float = 0.1,
            acc: float = 0.1,
            is_async: bool = False,
    ) -> bool:
        """
        相对于末端（工具）坐标系的 MoveL。

        transform: 4x4 齐次变换矩阵 T_delta（在末端系下描述的相对位姿），
        与当前 TCP 在基座系下的位姿右乘：T_new = T_tcp @ T_delta。
        可传入 ``numpy.ndarray`` shape (4, 4)，或 4 行每行 4 个 float 的嵌套序列。
        """
        arm = as_arm_group(arm)
        T = np.asarray(transform, dtype=np.float64)
        if T.shape == (4, 4):
            transform_matrix = T
        elif T.size == 16:
            transform_matrix = T.reshape(4, 4)
        else:
            raise ValueError(
                "transform must be a 4x4 matrix or 16 floats (row-major), "
                f"got shape {tuple(T.shape)}"
            )

        tcp_rt = self.get_tcp_rt(arm)
        if tcp_rt is None:
            raise ValueError("tcp_rt is None")
        tcp_matrix = rt_to_matrix(tcp_rt)

        target_tcp_matrix = tcp_matrix @ transform_matrix
        target_tcp_rt = matrix_to_rt(target_tcp_matrix)
        return self.movel(target_tcp_rt, arm, v, acc, is_async)


    def enable_speedl(self, enable: bool = True) -> bool:
        """
        Enable / disable TCP speed control (SpeedL)
        service: /zj_humanoid/upperlimb/enable_speedl
        """
        service = f"{self.prefix}/enable_speedl"
        return self._call_setbool_service(
            service_name=service,
            enable=enable,
            tag="SpeedL"
        )

    def enable_speedj(self, enable: bool = True) -> bool:
        """
        Enable / disable joint speed control (SpeedJ)
        service: /zj_humanoid/upperlimb/enable_speedj
        """
        service = f"{self.prefix}/enable_speedj"
        return self._call_setbool_service(
            service_name=service,
            enable=enable,
            tag="SpeedJ"
        )


    def speedl(
        self,
        tcp_speed: Sequence[float],
        arm: ArmGroup,
        acc: float = 0.05,
    ) -> None:

        arm = as_arm_group(arm)
        if arm not in L_GROUP:
            raise ValueError(f"Unsupported arm group: {arm}")
        length = TCP_SPEED_DOF[arm]
        check_length(tcp_speed, length, "tcp_speed")

        msg = SpeedL()
        msg.tcp_speed = list(tcp_speed)
        msg.acc = acc

        self._speedl_pubs[arm].publish(msg)


    def speedj(
        self,
        joint_speed: Sequence[float],
        arm: ArmGroup,
        acc: float = 0.5,
    ) -> None:

        arm = as_arm_group(arm)
        check_length(joint_speed, self._joint_dof(arm), "joint_speed")

        msg = SpeedJ()
        msg.joint_speed = list(joint_speed)
        msg.acc = acc

        self._speedj_pubs[arm].publish(msg)


    def stop_speedl(self, arm: ArmGroup):
        arm = as_arm_group(arm)
        if arm not in L_GROUP:
            raise ValueError(f"Unsupported arm group: {arm}")
        length = TCP_SPEED_DOF[arm]
        self.speedl([0.0] * length, arm, acc=0.05)


    def stop_speedj(self, arm: ArmGroup):
        arm = as_arm_group(arm)
        length = self._joint_dof(arm)
        self.speedj([0.0] * length, arm, acc=0.05)


    def _call_setbool_service(
        self,
        service_name: str,
        enable: bool,
        tag: str,
    ) -> bool:
        try:
            rospy.wait_for_service(service_name, timeout=5.0)
            client = rospy.ServiceProxy(service_name, SetBool)
            resp = client(SetBoolRequest(data=enable))

            if resp.success:
                state = "启用" if enable else "禁用"
                rospy.loginfo(f"[{tag}] {state}成功")
            else:
                rospy.logerr(f"[{tag}] 设置失败: {resp.message}")

            return resp.success

        except rospy.ROSException as e:
            rospy.logerr(f"[{tag}] 服务不可用: {e}")
            return False
        except rospy.ServiceException as e:
            rospy.logerr(f"[{tag}] 服务调用失败: {e}")
            return False


    # def set_servo_params(self, time_sec: float, gain: int) -> bool:
    #     """
    #     设置 servoj 参数

    #     参数:
    #     - time_sec: servoj 控制周期，例如 0.02
    #     - gain:     servoj 增益，例如 800
    #     """
    #     service = f"{self.prefix}/set_servo_params"

    #     try:
    #         rospy.wait_for_service(service, timeout=5.0)
    #         client = rospy.ServiceProxy(service, Servo)
    #         resp = client(ServoRequest(time=time_sec, gain=gain, arm_type=3))
    #         return resp.success
    #     except rospy.ROSException as e:
    #         rospy.logerr(f"[ServoJ] set_servo_params 服务不可用: {e}")
    #         return False
    #     except rospy.ServiceException as e:
    #         rospy.logerr(f"[ServoJ] set_servo_params 调用失败: {e}")
    #         return False


    def set_servo_params(
        self,
        time_sec: float,
        gain: int,
        arm: Union[ArmGroup, int] = ArmGroup.LEFT,
    ) -> bool:
        """
        设置 ServoJ / ServoL 共用参数。

        time_sec:
            目标点发布周期。当前仅允许已验证值 0.02 秒（50 Hz）。

        gain:
            位置跟踪增益。当前仅允许已验证值 800。

        arm:
            arm_type。SDK 1.3.2 实测合法值为 LEFT=1 / RIGHT=2 / DUAL=3；
            默认 LEFT，避免默认 0 被拒绝。
        """
        if not np.isfinite(time_sec) or not np.isclose(time_sec, 0.02):
            raise ValueError("only the verified time_sec=0.02 is currently allowed")
        if isinstance(gain, bool) or not isinstance(gain, int) or gain != 800:
            raise ValueError("only the verified gain=800 is currently allowed")

        arm = as_arm_group(arm)
        if arm not in L_GROUP:
            raise ValueError(f"set_servo_params does not support arm group: {arm}")

        service = f"{self.prefix}/set_servo_params"

        req = ServoRequest()
        req.time = time_sec
        req.gain = gain
        req.arm_type = arm.value

        try:
            rospy.wait_for_service(service, timeout=5.0)
            client = rospy.ServiceProxy(service, Servo)
            resp = client(req)

            if not resp.success:
                rospy.logerr(f"[Servo] 设置伺服参数失败: {resp.message}")

            return resp.success

        except rospy.ROSException as exc:
            rospy.logerr(f"[Servo] set_servo_params 服务不可用: {exc}")
            return False

        except rospy.ServiceException as exc:
            rospy.logerr(f"[Servo] set_servo_params 调用失败: {exc}")
            return False

    def clear_servo_params(self) -> bool:
        """
        清除 ServoJ / ServoL 共用参数。
        """
        service = f"{self.prefix}/clear_servo_params"

        try:
            rospy.wait_for_service(service, timeout=5.0)
            client = rospy.ServiceProxy(service, Servo)
            resp = client(ServoRequest())
            if not resp.success:
                rospy.logerr(f"[Servo] 清除伺服参数失败: {resp.message}")
            return resp.success
        except rospy.ROSException as e:
            rospy.logerr(f"[Servo] clear_servo_params 服务不可用: {e}")
            return False
        except rospy.ServiceException as e:
            rospy.logerr(f"[Servo] clear_servo_params 调用失败: {e}")
            return False

    def stop(self) -> bool:
        """立即停止上肢运动。"""
        service = f"{self.prefix}/stop"
        try:
            rospy.wait_for_service(service, timeout=5.0)
            client = rospy.ServiceProxy(service, Trigger)
            resp = client(TriggerRequest())
            if not resp.success:
                rospy.logerr(f"[Stop] 停止上肢失败: {resp.message}")
            return resp.success
        except rospy.ROSException as exc:
            rospy.logerr(f"[Stop] stop 服务不可用: {exc}")
            return False
        except rospy.ServiceException as exc:
            rospy.logerr(f"[Stop] stop 服务调用失败: {exc}")
            return False

    def unlock(self) -> bool:
        """解除上肢 safety lock。已解锁时服务会返回 failed，视为成功。"""
        service = f"{self.prefix}/unlock"
        try:
            rospy.wait_for_service(service, timeout=5.0)
            client = rospy.ServiceProxy(service, Trigger)
            resp = client(TriggerRequest())
            message = str(getattr(resp, "message", "") or "")
            if resp.success:
                rospy.loginfo(f"[Unlock] {message or 'ok'}")
                return True
            # Already unlocked: SDK returns failed instead of a no-op success.
            rospy.logwarn(f"[Unlock] {message or 'already unlocked'}")
            return True
        except rospy.ROSException as exc:
            rospy.logwarn(f"[Unlock] unlock 服务不可用: {exc}")
            return False
        except rospy.ServiceException as exc:
            rospy.logwarn(f"[Unlock] unlock 服务调用失败: {exc}")
            return False


    def servoj_dual_arm(self, joints: Sequence[float]) -> None:
        """
        双臂高频关节位置伺服控制

        输入:
        - joints: 双臂关节数组，顺序为 [left, right]
        """
        check_length(joints, self._joint_dof(ArmGroup.DUAL), "servoj dual_arm joints")

        msg = Joints(joints)
        self._servoj_dual_pub.publish(msg)


    # def IK(pose, q7, arm_type: ArmGroup, q_init=None, q_ref=None, seed_joints=None) -> bool:

    #     service = f"{self.prefix}/IK/{arm_type}"

    def servol(
        self,
        pose: Sequence[float],
        arm: ArmGroup,
    ) -> None:
        """
        笛卡尔空间高频绝对位置控制。

        LEFT / RIGHT:
            pose = [x, y, z, qx, qy, qz, qw]

        DUAL:
            pose = [
                left_x, left_y, left_z,
                left_qx, left_qy, left_qz, left_qw,
                right_x, right_y, right_z,
                right_qx, right_qy, right_qz, right_qw,
            ]
        """
        arm = as_arm_group(arm)

        if arm not in L_GROUP:
            raise ValueError(f"ServoL does not support arm group: {arm}")

        check_length(
            pose,
            TCP_POSE_DOF[arm],
            f"servol {arm.name.lower()} pose",
        )
        pose_array = np.asarray(pose, dtype=float)
        if not np.all(np.isfinite(pose_array)):
            raise ValueError("servol pose must contain only finite values")

        quaternion_offsets = (3,) if arm != ArmGroup.DUAL else (3, 10)
        for offset in quaternion_offsets:
            quaternion_norm = float(np.linalg.norm(pose_array[offset:offset + 4]))
            if quaternion_norm < 1e-8:
                raise ValueError("servol quaternion must not be zero")
            if not np.isclose(quaternion_norm, 1.0, atol=1e-3):
                raise ValueError(
                    f"servol quaternion must be normalized, got norm={quaternion_norm}"
                )

        msg = DualPose()

        if arm == ArmGroup.LEFT:
            msg.left_arm_pose = list_to_pose(pose)

        elif arm == ArmGroup.RIGHT:
            msg.right_arm_pose = list_to_pose(pose)

        elif arm == ArmGroup.DUAL:
            msg.left_arm_pose = list_to_pose(pose[:7])
            msg.right_arm_pose = list_to_pose(pose[7:14])

        self._servol_pubs[arm].publish(msg)


    def movej_by_path(
            self, 
            path: Sequence[Sequence[float]], 
            arm: ArmGroup, 
            total_time: Optional[float] = None, 
            timestamps: Optional[Sequence[float]] = None, 
            is_async: bool = False) -> bool:
        
        arm = as_arm_group(arm)
        if arm == ArmGroup.LIFT:
            raise ValueError("LIFT arm does not support movej_by_path")
        if len(path) < 2:
            raise ValueError("path length must be greater than 2")

        service_suffix = self._joint_service(arm)
        dof = self._joint_dof(arm)

        service = f"{self.prefix}/movej_by_path/{service_suffix}"
        req = MoveJByPathRequest()
        for i, joints in enumerate(path):
            check_length(joints, dof, f"joints{i}")
            req.path.append(Joints(joints))

        if total_time is not None:
            req.time = total_time
        if total_time is None and timestamps is not None:
            if len(timestamps) != len(path):
                raise ValueError("timestamps length must be equal to path length")
            req.timestamp = timestamps
        if total_time is None and timestamps is None:
            raise ValueError("total_time and timestamps cannot be both is none")
        req.is_async = is_async
        try:        
            rospy.wait_for_service(service, timeout=5.0)
            client = rospy.ServiceProxy(service, MoveJByPath)
            resp = client(req)
            return resp.success
        except rospy.ROSException as e:
            rospy.logerr(f"MoveJByPath service not available: {e}")
            return False

    # NOTE 待测试
    def movejh_by_path(
            self, 
            path: Sequence[Sequence[float]], 
            arm_mask: int,
            total_time: Optional[float] = None, 
            timestamps: Optional[Sequence[float]] = None, 
            is_async: bool = False) -> bool:
        
        expected_len = self._whole_body_expected_len(arm_mask)
        if len(path) < 2:
            raise ValueError("path length must be greater than 2")
        
        service = f"{self.prefix}/movej_by_path/whole_body"
        req = MoveJByPathRequest()
        req.arm_type = arm_mask
        for i, joints in enumerate(path):
            check_length(joints, expected_len, f"joints{i}")
            req.path.append(Joints(joints))

        if total_time is not None:  
            req.time = total_time
        if total_time is None and timestamps is not None:
            if len(timestamps) != len(path):
                raise ValueError("timestamps length must be equal to path length")
            req.timestamp = timestamps
        if total_time is None and timestamps is None:
            raise ValueError("total_time and timestamps cannot be both is none")
        req.is_async = is_async
        try:        
            rospy.wait_for_service(service, timeout=5.0)
            client = rospy.ServiceProxy(service, MoveJByPath)
            resp = client(req)
            return resp.success
        except rospy.ROSException as e:
            rospy.logerr(f"MoveJByPath service not available: {e}")
            return False
        except rospy.ServiceException as e:
            rospy.logerr(f"MoveJByPath service call failed: {e}")
            return False
        


PREFIX="/zj_humanoid/upperlimb"
def test_pub_speedl_right_arm(*, name=f"{PREFIX}/speedl/right_arm"):
    """
    右臂笛卡尔空间速度控制
    
    Version
    -------
    - 1.0.0: added
    """
    enable_client = rospy.ServiceProxy(f"{PREFIX}/enable_speedl", SetBool)
    # 开启speedl
    resp = enable_client.call(SetBoolRequest(True))

    pub = rospy.Publisher(name, SpeedL, queue_size=1)
    rate = rospy.Rate(5)

    target_speed = 0.01
    counter = 0
    while 1:
        data = SpeedL()
        data.tcp_speed = [0, 0, target_speed, 0, 0, 0]
        data.acc = 0.5
        target_speed += 0.002
        print(target_speed)

        counter += 1
        if counter > 5:
            break

        pub.publish(data)
        rate.sleep()

    # 关闭speedl
    resp = enable_client.call(SetBoolRequest(False))



