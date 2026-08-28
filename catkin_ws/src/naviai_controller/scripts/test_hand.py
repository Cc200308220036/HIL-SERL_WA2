#!/usr/bin/env python3
"""Explicitly confirmed WA2 dexterous-hand status/grasp/release test."""

import argparse
import sys

import rospy
from naviai_controller import HandType, NaviController


GRASP_TARGET = [0.1, 0.9, 0.7, 0.7, 0.4, 0.4]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand", choices=("left", "right"), default="right")
    parser.add_argument(
        "--action",
        choices=("status", "grasp", "release", "cycle"),
        default="cycle",
    )
    parser.add_argument(
        "--target",
        default=None,
        help="six comma-separated grasp joints: THUMB_MP,THUMB_CMC,INDEX,MIDDLE,RING,LITTLE",
    )
    return parser.parse_args(rospy.myargv(argv=sys.argv)[1:])


def confirmed(token, prompt):
    return input("{}，输入 {}：".format(prompt, token)).strip() == token


def main():
    args = parse_args()
    rospy.init_node("test_hand", anonymous=True)
    ctrl = NaviController(model="wa2")
    rospy.sleep(1.0)
    hand = HandType.LEFT if args.hand == "left" else HandType.RIGHT
    side = args.hand.upper()
    grasp_target = GRASP_TARGET
    if args.target:
        parts = [float(x.strip()) for x in args.target.split(",")]
        if len(parts) != 6:
            raise SystemExit("--target must have 6 comma-separated numbers")
        grasp_target = parts
    print("grasp_target:", grasp_target)

    print("当前{}手:".format(args.hand), ctrl.get_hand_joints(hand))
    if args.action == "status":
        return 0

    if args.action in ("grasp", "cycle"):
        token = "GRASP_{}".format(side)
        if confirmed(token, "确认手指和物体区域安全"):
            ok = ctrl.grasp_hand(hand, grasp_target)
            print("grasp_hand 返回:", ok)
            print("抓握后反馈:", ctrl.get_hand_joints(hand))
            if not ok:
                return 1

    if args.action in ("release", "cycle"):
        token = "RELEASE_{}".format(side)
        if confirmed(token, "确认张开过程不会碰撞"):
            ok = ctrl.release_hand(hand)
            print("release_hand 返回:", ok)
            print("释放后反馈:", ctrl.get_hand_joints(hand))
            if not ok:
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
