#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
naviai_controller 接口简单测试 (WA2)
  - 先测所有只读接口 (不动机器人)
  - 再可选测运动接口 (每步回车确认)
"""
import time
import rospy
import numpy as np
from naviai_controller import NaviController, ArmGroup, HandType, RobotModel


def _fmt(v, n=6):
    if v is None:
        return "None"
    if isinstance(v, (list, tuple, np.ndarray)):
        return ["%.*f" % (n, float(x)) for x in v]
    return v


def test_read(ctrl):
    print("\n========== 只读接口测试 ==========")

    print("\n--- get_joints (关节角, rad) ---")
    for arm in (ArmGroup.LEFT, ArmGroup.RIGHT, ArmGroup.NECK, ArmGroup.WAIST):
        j = ctrl.get_joints(arm)
        print(f"  {arm.name:6}: {_fmt(j)}")

    print("\n--- get_tcp_rt (TCP 位姿 [x,y,z,qx,qy,qz,qw]) ---")
    for arm in (ArmGroup.LEFT, ArmGroup.RIGHT):
        p = ctrl.get_tcp_rt(arm)
        print(f"  {arm.name:6}: {_fmt(p)}")

    print("\n--- get_tcp_matrix (4x4 齐次矩阵) ---")
    for arm in (ArmGroup.LEFT, ArmGroup.RIGHT):
        m = ctrl.get_tcp_matrix(arm)
        if m is None:
            print(f"  {arm.name:6}: None")
        else:
            print(f"  {arm.name:6}:\n{np.array(m)}")

    print("\n--- get_tcp_speed (TCP 速度) ---")
    for arm in (ArmGroup.LEFT, ArmGroup.RIGHT):
        s = ctrl.get_tcp_speed(arm)
        print(f"  {arm.name:6}: {_fmt(s)}")

    print("\n--- get_hand_joints (手指关节) ---")
    for h in (HandType.LEFT, HandType.RIGHT):
        j = ctrl.get_hand_joints(h)
        print(f"  {h.name:6}: {_fmt(j)}")

    print("\n--- get_hand_pressures (手指压力) ---")
    for h in (HandType.LEFT, HandType.RIGHT):
        p = ctrl.get_hand_pressures(h)
        print(f"  {h.name:6}: {_fmt(p)}")

    print("\n--- get_hand_force (手腕力传感器) ---")
    for h in (HandType.LEFT, HandType.RIGHT):
        f = ctrl.get_hand_force(h)
        print(f"  {h.name:6}: {_fmt(f)}")

    print("\n[只读接口测试完成]")


def test_movej_small(ctrl):
    """左臂第2关节 +0.1 rad, 小幅安全运动"""
    print("\n--- movej 测试: 左臂第2关节 +0.1 rad ---")
    cur = ctrl.get_joints(ArmGroup.LEFT)
    if cur is None:
        print("  无法读取左臂关节, 跳过")
        return
    target = list(cur)
    target[1] += 0.1
    print(f"  当前: {_fmt(cur)}")
    print(f"  目标: {_fmt(target)}")
    input("  回车执行 movej (Ctrl+C 取消)...")
    ok = ctrl.movej(target, ArmGroup.LEFT, v=0.2, acc=0.3)
    print(f"  movej 返回: {ok}")


def test_movel_relative(ctrl):
    """右臂沿 BASE Z 上升 2cm"""
    print("\n--- movel_relative_base 测试: 右臂 Z +0.02 ---")
    input("  回车执行 (Ctrl+C 取消)...")
    ok = ctrl.movel_relative_base([0.0, 0.0, 0.02], ArmGroup.RIGHT, v=0.05, acc=0.05)
    print(f"  movel_relative_base 返回: {ok}")


def test_hand(ctrl):
    """右手抓/放"""
    print("\n--- grasp_hand / release_hand 测试: 右手 ---")
    input("  回车抓取 (Ctrl+C 取消)...")
    ctrl.grasp_hand(HandType.RIGHT, [0.1, 1.5, 1.2, 1.2, 1.2, 1.2])
    print("  已抓取, 等 1s ...")
    time.sleep(1.0)
    input("  回车释放 (Ctrl+C 取消)...")
    ctrl.release_hand(HandType.RIGHT)
    print("  已释放")


def test_speedl(ctrl):
    """右臂沿 Z 持续上升 1 秒 (循环发速度)"""
    print("\n--- speedl 测试: 右臂 Z +0.02 m/s 持续 1s ---")
    input("  回车执行 (Ctrl+C 取消)...")
    ctrl.enable_speedl()
    t0 = time.time()
    rate = rospy.Rate(50)
    while time.time() - t0 < 1.0 and not rospy.is_shutdown():
        ctrl.speedl([0.0, 0.0, 0.02, 0.0, 0.0, 0.0], ArmGroup.RIGHT, acc=0.1)
        rate.sleep()
    ctrl.stop_speedl(ArmGroup.RIGHT)
    ctrl.enable_speedl(False)
    print("  speedl 完成")


def main():
    rospy.init_node("test_naviai_interfaces", anonymous=True)
    print("初始化 NaviController (WA2) ...")
    ctrl = NaviController(model=RobotModel.WA2)
    rospy.sleep(1.0)  # 等回调填充状态

    # 1. 只读 (安全)
    test_read(ctrl)

    # 2. 运动类 (每步确认)
    print("\n========== 运动接口测试 (需确认) ==========")
    try:
        test_movej_small(ctrl)
        test_movel_relative(ctrl)
        test_hand(ctrl)
        # test_speedl(ctrl)
    except KeyboardInterrupt:
        print("\n用户取消")

    print("\n全部测试结束")


if __name__ == "__main__":
    main()
