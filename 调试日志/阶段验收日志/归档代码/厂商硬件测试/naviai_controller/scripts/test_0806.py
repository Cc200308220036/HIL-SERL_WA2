import rospy
from naviai_controller import NaviController, ArmGroup, HandType

"""
测试代码:movel 移动右臂, 从当前位置移动到右臂当前位置+0.01m
"""

def main():
    rospy.init_node("my_control_node")
    ctrl = NaviController(model="wa2")              # 默认 WA1

    right_tcp_rt = ctrl.get_tcp_rt(ArmGroup.RIGHT)
    print(right_tcp_rt)

    right_tcp_rt[0] += 0.01
    input("Press Enter to continue...")
    ctrl.movel(right_tcp_rt, ArmGroup.RIGHT)


if __name__ == "__main__":
    main()