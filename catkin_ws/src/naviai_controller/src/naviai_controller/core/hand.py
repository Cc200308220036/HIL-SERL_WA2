import rospy
from typing import Sequence

from zj_humanoid.hand.srv import HandJoint, HandJointRequest
from zj_humanoid.hand.msg import PressureSensor
from geometry_msgs.msg import WrenchStamped
from sensor_msgs.msg import JointState

from .enums import HandType
from .tools import check_length


class HandController:
    """
    Hand (gripper / dexterous hand) control binding
    """

    def __init__(self, prefix: str):
        self.prefix = prefix

        # ---------- State ----------
        self.left_hand_joints = None
        self.right_hand_joints = None
        self.left_hand_pressures = None
        self.right_hand_pressures = None
        self.left_force = None
        self.right_force = None

        self._init_subscribers()


    def _init_subscribers(self):
        rospy.Subscriber(
            f"{self.prefix}/joint_states",
            JointState,
            self._joint_states_cb,
        )

        rospy.Subscriber(
            f"{self.prefix}/finger_pressures/left",
            PressureSensor,
            self._left_pressure_cb,
        )

        rospy.Subscriber(
            f"{self.prefix}/finger_pressures/right",
            PressureSensor,
            self._right_pressure_cb,
        )

        rospy.Subscriber(
            f"/wrist_force_control/left_arm_compensated_force",
            WrenchStamped,
            self._left_force_cb,
        )    

        rospy.Subscriber(
            f"/wrist_force_control/right_arm_compensated_force",
            WrenchStamped,
            self._right_force_cb,
        )    


    def _left_force_cb(self, msg: WrenchStamped):
        self.left_force = msg.wrench.force

    def _right_force_cb(self, msg: WrenchStamped):
        self.right_force = msg.wrench.force


    def _joint_states_cb(self, msg: JointState):
        try:
            if len(msg.position) != 12:
                rospy.logwarn("hand joint_states length != 12")
                return

            self.left_hand_joints = list(msg.position[:6])
            self.right_hand_joints = list(msg.position[6:])

        except Exception as e:
            rospy.logwarn(f"hand joint_states callback error: {e}")


    def _left_pressure_cb(self, msg: PressureSensor):
        self.left_hand_pressures = list(msg.pressure)

    def _right_pressure_cb(self, msg: PressureSensor):
        self.right_hand_pressures = list(msg.pressure)



    def get_joints(self, hand: HandType):
        if hand == HandType.LEFT:
            return self.left_hand_joints
        elif hand == HandType.RIGHT:
            return self.right_hand_joints
        return None

    def get_pressures(self, hand: HandType):
        if hand == HandType.LEFT:
            return self.left_hand_pressures
        elif hand == HandType.RIGHT:
            return self.right_hand_pressures
        return None

    def get_force(self, hand: HandType):
        if self.left_force is None or self.right_force is None:
            return None
        if hand == HandType.LEFT:
            return self.left_force
        elif hand == HandType.RIGHT:
            return self.right_force
        return None

    def grasp(self, hand: HandType, joints: Sequence[float]) -> bool:
        return self.set_joints(hand, joints)

    def release(self, hand: HandType) -> bool:
        return self.set_joints(hand, [0.0] * 6)



    def set_joints(self, hand: HandType, joints: Sequence[float]) -> bool:
        if not isinstance(hand, HandType):
            raise TypeError("hand must be HandType")

        check_length(joints, 6, "hand joints")

        side = "left" if hand == HandType.LEFT else "right"
        service = f"{self.prefix}/joint_switch/{side}"

        try:
            rospy.wait_for_service(service, timeout=5.0)
            client = rospy.ServiceProxy(service, HandJoint)

            req = HandJointRequest(q=list(joints))
            resp = client(req)

        except rospy.ROSException as e:
            rospy.logerr(f"HandJoint service not available: {e}")
            return False
        except rospy.ServiceException as e:
            rospy.logerr(f"HandJoint service call failed: {e}")
            return False
    
        return resp.success