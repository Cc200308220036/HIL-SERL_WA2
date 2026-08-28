# naviai_controller.py
from naviai_controller.core.arm import ArmController
from naviai_controller.core.hand import HandController
from naviai_controller.core.enums import ArmGroup, HandType, RobotModel
# from typing import Optional
from typing import List, Optional, Union, Sequence
import numpy as np
import threading


class NaviController:
    def __init__(self, auto_spin: bool = True, model: Union[RobotModel, str] = RobotModel.WA1):
        self.arm = ArmController("/zj_humanoid/upperlimb", model=model)
        self.hand = HandController("/zj_humanoid/hand")

        self._spin_thread = None
        if auto_spin:
            import rospy
            def _spin():
                rospy.spin()
            self._spin_thread = threading.Thread(target=_spin, daemon=True)
            self._spin_thread.start()

    # [x,y,z,x,y,z,w]
    def get_tcp_rt(self, arm: Union[ArmGroup, int]):
        return self.arm.get_tcp_rt(arm)

    def get_tcp_matrix(self, arm: Union[ArmGroup, int]):
        return self.arm.get_tcp_matrix(arm)

    def get_joints(self, arm: Union[ArmGroup, int]):
        return self.arm.get_joints(arm)

    def get_tcp_speed(self, arm: Union[ArmGroup, int]):
        return self.arm.get_tcp_speed(arm)

    def get_tcp_age(self, arm: Union[ArmGroup, int]):
        return self.arm.get_tcp_age(arm)

    def get_uplimb_state_age(self):
        return self.arm.get_uplimb_state_age()

    def get_is_singular(self, arm: Union[ArmGroup, int]):
        return self.arm.get_is_singular(arm)

    def get_cmd_num(self):
        return self.arm.get_cmd_num()



    def movej(
        self,
        joints: List[float],
        arm: Union[ArmGroup, int],
        v: float = 0.3,
        acc: float = 0.5,
        t: Optional[float] = None,
        is_async: bool = False,
    ) -> bool:
        return self.arm.movej(
            joints=joints,
            arm=arm,
            v=v,
            acc=acc,
            t=t,
            is_async=is_async,)

    def movejh(
        self,
        joints: List[float],
        mask: int,               # 8421
        v: float = 0.3,
        acc: float = 0.5,
        is_async: bool = False,
        ) -> bool:
        return self.arm.movejh(
            arm_mask=mask,
            joints=joints,
            v=v,
            acc=acc,
            is_async=is_async,)

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
        pose,
        arm: Union[ArmGroup, int],
        v: float = 0.1,
        acc: float = 0.1,
        is_async: bool = False,
    ) -> bool:
        """
        pose:
            LEFT / RIGHT: [x, y, z, qx, qy, qz, qw]
            DUAL:        [left_pose, right_pose]
        """
        return self.arm.movel(
            pose=pose,
            arm=arm,
            v=v,
            acc=acc,
            is_async=is_async,)

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


    def set_servo_params(
        self,
        time_sec: float,
        gain: int,
        arm: Union[ArmGroup, int] = ArmGroup.LEFT,
    ) -> bool:
        return self.arm.set_servo_params(time_sec, gain, arm=arm)

    def clear_servo_params(self) -> bool:
        return self.arm.clear_servo_params()

    def stop(self) -> bool:
        return self.arm.stop()

    def unlock(self) -> bool:
        return self.arm.unlock()

    def servoj_dual_arm(self, joints: List[float]) -> None:
        return self.arm.servoj_dual_arm(joints)

    def servol(
        self,
        pose: Sequence[float],
        arm: Union[ArmGroup, int],
    ) -> None:
        """
        笛卡尔空间高频绝对位置控制。

        LEFT / RIGHT:
            [x, y, z, qx, qy, qz, qw]

        DUAL:
            左臂7维位姿 + 右臂7维位姿
        """
        return self.arm.servol(pose=pose, arm=arm)


    # hand control

    def get_hand_joints(self, hand: HandType):
        return self.hand.get_joints(hand)

    def get_hand_pressures(self, hand: HandType):
        return self.hand.get_pressures(hand)

    def grasp_hand(self, hand: HandType, joints):
        return self.hand.grasp(hand, joints)

    def release_hand(self, hand: HandType):
        return self.hand.release(hand)

    def get_hand_joints(self, hand: HandType):
        return self.hand.get_joints(hand)

    def get_hand_force(self, hand: HandType):
        return self.hand.get_force(hand)    
