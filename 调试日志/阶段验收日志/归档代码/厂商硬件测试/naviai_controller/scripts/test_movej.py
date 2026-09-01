import rospy
from naviai_controller import NaviController, ArmGroup


JOINT_INDEX = 7
DELTA_RAD = 0.02

rospy.init_node("test_movej")
ctrl = NaviController(model="wa2")
rospy.sleep(1.0)

current = ctrl.get_joints(ArmGroup.RIGHT)

target = list(current)
target[JOINT_INDEX] += DELTA_RAD

print("当前关节:", current)
print("目标关节:", target)
print(f"joint[{JOINT_INDEX}] += {DELTA_RAD} rad")

if input("确认关节限位和路径安全，输入 MOVE 执行：").strip() == "MOVE":
    ok = ctrl.movej(
        target,
        ArmGroup.RIGHT,
        v=0.1,
        acc=0.2,
        is_async=False,
    )
    print("movej 返回:", ok)
else:
    print("已取消")