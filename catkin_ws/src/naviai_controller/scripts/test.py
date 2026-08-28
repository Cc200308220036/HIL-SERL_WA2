import time
import cv2
import rospy
from std_srvs.srv import SetBool, SetBoolRequest
from zj_humanoid.upperlimb.msg import SpeedL
# 全局变量：Z轴速度
z_speed = 0.0 


PREFIX = "zj_humanoid/upperlimb"

def main():
    global z_speed
    
    rospy.init_node('simple_arm_remote')

    # 1. 开启 SpeedL 服务
    rospy.wait_for_service(f"{PREFIX}/enable_speedl")
    enable_client = rospy.ServiceProxy(f"{PREFIX}/enable_speedl", SetBool)
    enable_client.call(SetBoolRequest(True))

    # 2. 创建发布器 (需根据实际包名导入 SpeedL)
    pub = rospy.Publisher(f"{PREFIX}/speedl/right_arm/", SpeedL, queue_size=1)

    # 3. 创建一个最小化的窗口 (仅仅是为了让 cv2 能捕获键盘输入，窗口可以是空的)
    cv2.namedWindow("Key Input")

    print("控制启动: W(+) S(-) Space(停止) Q(退出)")

    acc = 5
    while not rospy.is_shutdown():
        # --- A. 捕获按键 ---
        key = cv2.waitKey(4) & 0xFF # 等待50ms，这决定了你的循环响应速度

        if key == ord('w'):      
            z_speed += 0.01
            # z_speed = 0.005
            acc = 0.1
        elif key == ord('s'):    
            z_speed -= 0.01
            # z_speed = -0.005
            acc = 0.1
        elif key == 32:          
            z_speed = 0.0  # 空格键急停
            acc = 0.1
        elif key == ord('q'):    break          # 退出

        # 限幅保护 (防止速度溢出)
        z_speed = max(-0.2, min(0.2, z_speed))

        # --- B. 发布消息 ---
        msg = SpeedL()
        msg.tcp_speed = [0, 0, z_speed, 0, 0, 0] # 只控制Z轴
        msg.acc = acc
        pub.publish(msg)

        # 控制台输出反馈 (替代图形界面)
        print(f"Current Z-Speed: {z_speed:.3f}")

    # 4. 退出清理
    print("正在停止机器人...")
    enable_client.call(SetBoolRequest(False)) # 关闭服务
    cv2.destroyAllWindows()

    # time.sleep(10)





def test():
    
    rospy.init_node('simple_arm_remote')

    # 1. 开启 SpeedL 服务
    rospy.wait_for_service(f"{PREFIX}/enable_speedl")
    enable_client = rospy.ServiceProxy(f"{PREFIX}/enable_speedl", SetBool)
    enable_client.call(SetBoolRequest(True))

    # 2. 创建发布器 (需根据实际包名导入 SpeedL)
    pub = rospy.Publisher(f"{PREFIX}/speedl/right_arm/", SpeedL, queue_size=1)

    # 3. 创建一个最小化的窗口 (仅仅是为了让 cv2 能捕获键盘输入，窗口可以是空的)
    cv2.namedWindow("Key Input")

    print("控制启动: W(+) S(-) Space(停止) Q(退出)")


    for i in range(5):
        msg = SpeedL()
        msg.tcp_speed = [0, 0, 0.005, 0, 0, 0] # 只控制Z轴
        msg.acc = 0.1
        pub.publish(msg)
        time.sleep(0.1)
    for i in range(5):
        msg = SpeedL()
        msg.tcp_speed = [0, 0, -0.005, 0, 0, 0] # 只控制Z轴
        msg.acc = 0.1
        pub.publish(msg)
        time.sleep(0.1)

    msg = SpeedL()
    msg.tcp_speed = [0, 0, 0, 0, 0, 0] # 只控制Z轴
    msg.acc = 0.1
    pub.publish(msg)
    time.sleep(0.1)
    enable_client.call(SetBoolRequest(False))





if __name__ == '__main__':
    main()
    # test()
