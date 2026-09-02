#!/usr/bin/env python3
"""WA2 左臂导纳控制的低速手推测试。

默认目标速度和目标力均为零。启动后，左臂会根据手腕外力产生低速运动；
右臂始终发送零速度。请只在机器人周围无障碍、急停可用且有人监护时运行。
"""

import rospy

from naviai_controller import ArmGroup

from ad_ctrl_rc_v2 import (
    AdmittanceConfig,
    AdmittanceController,
    AdmittanceMode,
)


def _read_vector_param(name, default, size):
    """读取并检查 ROS 数组参数，避免错误长度进入控制器。"""
    value = rospy.get_param(name, default)
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f"{name} 必须是长度为 {size} 的数组")
    return [float(item) for item in value]


def main():
    rospy.init_node("wa2_left_admittance_test")

    # 可通过 rosparam 覆盖这些任务参数。例如：
    # _target_force:=[-8,0,0] 表示给 X 方向加入 -8 N 的目标力偏置。
    duration = float(rospy.get_param("~duration", 15.0))
    desired_velocity = _read_vector_param(
        "~desired_velocity", [0, 0, 0, 0, 0, 0], 6
    )
    target_force = _read_vector_param("~target_force", [0, 0, 0], 3)

    config = AdmittanceConfig(
        robot_model="wa2",
        frequency=125.0,

        # 传感器死区：小于阈值的力/力矩不引起导纳运动。
        force_deadband=6.0,          # N
        torque_deadband=0.2,        # N·m

        # 外力进入模型前的缩放。初次实验使用较小值。
        sensor_force_ratio=0.1,
        sensor_torque_ratio=0.1,

        # 虚拟质量越大，响应越慢；阻尼越大，同样外力下速度越小。
        left_mass=6.0,
        left_damping=80.0,

        # 最终输出比例及硬限速。下面是首次手推测试的保守值。
        speed_ratio=0.1,
        max_linear_speed=0.015,     # m/s
        max_angular_speed=0.08,     # rad/s
        speed_acceleration=0.1,

        # 死区处理后的三维力模长超过该值时自动停止。
        max_force_norm=20.0,        # N

        # 力或 joint_states 超过此时间未更新，立即零速并退出控制循环。
        state_timeout=0.2,          # s

        # WA2 默认腰部 yaw/pitch 索引是 (18, 19)。若实际 SDK 排列不同，
        # 应先核对 joint_states，再在这里显式设置 waist_joint_indices。
        waist_joint_indices=(18, 19),
    )

    controller = AdmittanceController(config)
    rospy.on_shutdown(controller.stop)

    # SINGLE + LEFT：左臂经过导纳模型，右臂使用其期望速度（这里固定为零）。
    controller.set_mode(AdmittanceMode.SINGLE, active_arm=ArmGroup.LEFT)
    controller.set_velocity(ArmGroup.LEFT, desired_velocity)
    controller.set_velocity(ArmGroup.RIGHT, [0, 0, 0, 0, 0, 0])
    controller.set_target_force(ArmGroup.LEFT, target_force)

    rospy.loginfo("等待左腕力传感器和 WA2 joint_states 数据……")
    if not controller.wait_until_ready(timeout=5.0):
        raise RuntimeError("左腕力传感器或 joint_states 在 5 秒内未就绪")

    initial_force = controller.get_force(ArmGroup.LEFT)
    rospy.loginfo("左腕当前六维力/力矩: %s", initial_force)
    rospy.logwarn(
        "即将使能 SpeedL：请确认工作空间无障碍、急停可用，并扶稳机器人。"
    )
    input("确认安全后按 Enter 开始；按 Ctrl+C 取消：")

    try:
        controller.start(wait_for_force=True, timeout=2.0)
        rospy.loginfo(
            "左臂导纳已启动，测试 %.1f 秒；可轻推 X/Y 方向，随时按 Ctrl+C 停止。",
            duration,
        )

        # 低频打印力数据；125 Hz 控制循环在控制器内部线程中运行。
        end_time = rospy.Time.now() + rospy.Duration(duration)
        rate = rospy.Rate(2.0)
        while not rospy.is_shutdown() and rospy.Time.now() < end_time:
            if not controller.is_running():
                raise RuntimeError("导纳控制循环已因异常或数据超时停止")
            rospy.loginfo("左腕六维力/力矩: %s", controller.get_force(ArmGroup.LEFT))
            rate.sleep()
    finally:
        # 无论正常结束、Ctrl+C 还是发生异常，都先发零速度，再关闭 SpeedL。
        controller.stop()
        rospy.loginfo("左臂导纳测试已停止，SpeedL 已关闭")


if __name__ == "__main__":
    main()
