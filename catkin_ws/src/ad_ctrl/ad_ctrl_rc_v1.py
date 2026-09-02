#!/usr/bin/env python3
import rospy, time, rospkg, argparse
import numpy as np

from upperlimb.msg import Pose, SpeedL
from sensor_msgs.msg import JointState
from geometry_msgs.msg import WrenchStamped, Vector3
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Float64MultiArray,Float32MultiArray,Bool,Float64
from std_srvs.srv import SetBool, SetBoolRequest
from threading import Lock
from std_msgs.msg import Header

class Admittance_Para():
    def __init__(self):
        self.enable = False
        self.single = True          # True=单臂导纳, False=双臂导纳
        self.single_left = True    # 单臂时: True=左臂导纳, False=右臂导纳; 双臂主从时: True=左主右从, False=右主左从
        self.master_slave = False       # 仅当 single=False 时有效: True=主从模式(一臂导纳一臂跟随), False=双臂模式(双都导纳)
        self.update_left = False
        self.update_right = False
        self.left_weight = 0.0
        self.right_weight = 0.0
        self.left_force_length = np.array([0.0, 0.0, 0.0])
        self.right_force_length = np.array([0.0,  0.0, 0.0])
        self.force_cal_thld = 6                                                    #针对校准后的力外加期望力设定的死区
        self.torque_cal_thld = 0.2                                                 #针对校准后的力矩设定的死区
        self.sensor_force_ratio = 0.1                                              #送入导纳控制器的外力系数
        self.sensor_torque_ratio = 0.1                                             #送入导纳控制器的外力矩系数
        self.speedL_ratio = 1 
        self.left_vel_des = np.array([0.,0.,0.,0.,0.,0.])                          #存储的是调用导纳模块时传入的左手期望速度
        self.right_vel_des = np.array([0.,0.,0.,0.,0.,0.])                         #右手期望速度的存储容器
        self.left_des_force = np.array([0.,0.,0.])                                  #左手z方向的期望外力
        self.right_des_force = np.array([0.,0.,0.])
        self.frequency = 125                                                       #导纳控制器频率，可通过topic修改
        self.left_force = WrenchStamped()
        self.right_force = WrenchStamped()
        self.left_tcp = np.array([0.,0.,0.,0.,0.,0.])
        self.right_tcp = np.array([0.,0.,0.,0.,0.,0.])
        self.waist_rot = np.eye(3)
        self.waist_rot_6x6 = np.eye(6)
        self.waist_rot_6x6_inv = np.eye(6)
        self.left_num = 0
        self.right_num = 0
        self.left_cmd = np.array([0.,0.,0.,0.,0.,0.])
        self.right_cmd = np.array([0.,0.,0.,0.,0.,0.])
        self.left_cmd_old = np.array([0.,0.,0.,0.,0.,0.])
        self.right_cmd_old = np.array([0.,0.,0.,0.,0.,0.])
        self.left_para_K = 50
        self.left_para_D = 40
        self.left_para_M = 3
        self.right_para_K = 50
        self.right_para_D = 40
        self.right_para_M = 3
        self.resample_time = 0.01

class ReSample_6D():
    class ReSample:
        def __init__(self):
            # 实例化时初始化变量
            self.a = self.b = self.c = self.d = self.e = self.f = 0
            self.T = 0
            self.is_constant = False  # 标记是否 start≈end，无需插值
            self.constant_value = 0   # 保存 start≈end 时的值

        def generate_quintic(self, start, end, total_time):
            """
            生成严格满足两端一阶、二阶导数为0的五次多项式
            如果起点和终点几乎相等，则无需插值，直接返回常数
            """ 
            self.T = total_time
            if abs(start - end) < 1e-5:
                self.is_constant = True
                self.constant_value = end
                self.a = self.b = self.c = self.d = self.e = self.f = 0
            else:
                self.is_constant = False
                self.a = 6 * (end - start) / self.T**5
                self.b = -15 * (end - start) / self.T**4
                self.c = 10 * (end - start) / self.T**3
                self.d = 0
                self.e = 0
                self.f = start

        def get_quintic_value(self, t):
            """
            返回给定时刻的插值值
            t > total_time 时直接返回 end
            """
            if self.is_constant:
                return self.constant_value
            if t >= self.T:
                return self.a * self.T**5 + self.b * self.T**4 + self.c * self.T**3 + self.d * self.T**2 + self.e * self.T + self.f
            t = max(t, 0)
            return self.a * t**5 + self.b * t**4 + self.c * t**3 + self.d * t**2 + self.e * t + self.f

    def __init__(self):
        self.lock = Lock()
        self.x_resample = ReSample_6D.ReSample()
        self.y_resample = ReSample_6D.ReSample()
        self.z_resample = ReSample_6D.ReSample()
        self.rx_resample = ReSample_6D.ReSample()
        self.ry_resample = ReSample_6D.ReSample()
        self.rz_resample = ReSample_6D.ReSample()

    def gen(self, cmd_old, cmd_now, time):
        if len(cmd_old)!=6 or len(cmd_now)!=6:
            print('input error')
            return
        with self.lock:
            self.x_resample.generate_quintic(cmd_old[0], cmd_now[0], time)
            self.y_resample.generate_quintic(cmd_old[1], cmd_now[1], time)
            self.z_resample.generate_quintic(cmd_old[2], cmd_now[2], time)
            self.rx_resample.generate_quintic(cmd_old[3], cmd_now[3], time)
            self.ry_resample.generate_quintic(cmd_old[4], cmd_now[4], time)
            self.rz_resample.generate_quintic(cmd_old[5], cmd_now[5], time)

    def get(self, num, freq):
        t = num / freq
        with self.lock: 
            return [
                self.x_resample.get_quintic_value(t),
                self.y_resample.get_quintic_value(t),
                self.z_resample.get_quintic_value(t),
                self.rx_resample.get_quintic_value(t),
                self.ry_resample.get_quintic_value(t),
                self.rz_resample.get_quintic_value(t)
            ]

class Admittance:

    def __init__(self, ad_para, K=0, D=0, M=0, delta_t=0):
        self.ad_para = ad_para
        self.K = K
        self.D = D
        self.M = M
        self.delta_t = delta_t
        self.last_v_tar = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def set_coef(self, K, D, M, delta_t):
        self.K = K
        self.D = D
        self.M = M
        self.delta_t = delta_t

    def ad_ctrl(self, v_des, f_ext):
        a_tar = 1/self.M*(f_ext - self.D*(self.last_v_tar - self.ad_para.waist_rot_6x6 @ v_des))
        v_tar = self.last_v_tar + a_tar*self.delta_t
        self.last_v_tar = v_tar
        return self.ad_para.waist_rot_6x6_inv @ v_tar

class Data_Process():

    def __init__(self, ad_para, left_resample, right_resample):
        self.ad_para = ad_para
        self.left_resample = left_resample
        self.right_resample = right_resample

        # 本体的状态相关
        rospy.Subscriber('/zj_humanoid/upperlimb/joint_states', JointState, self.joint_state_callback)
        self.right_rot = np.eye(3)
        rospy.Subscriber("/zj_humanoid/upperlimb/tcp_pose/left_arm", Pose, self.right_tcp_recived_callback, queue_size=10)
        self.left_rot = np.eye(3)
        rospy.Subscriber("/zj_humanoid/upperlimb/tcp_pose/right_arm", Pose, self.left_tcp_recived_callback, queue_size=10)

        # speedl topic 底层控制实现 直接采用双臂控制实现
        self.dual_v_publisher = rospy.Publisher("/zj_humanoid/upperlimb/speedl/dual_arm", SpeedL, queue_size=10)
        self.enable_speedl_service_name = "/zj_humanoid/upperlimb/enable_speedl"
        
        # 订阅力传感器数据
        rospy.Subscriber("/wrist_force_control/left_arm_compensated_force", WrenchStamped, self.left_force_callback,queue_size=10, tcp_nodelay = True)
        rospy.Subscriber("/wrist_force_control/right_arm_compensated_force", WrenchStamped, self.right_force_callback,queue_size=10, tcp_nodelay = True)
        
        # 使能开关
        rospy.Subscriber('/ad_ctrl_enable', Bool, self.enable_callback)
        rospy.Subscriber('/ad_ctrl_left', Bool, self.single_callback)
        rospy.Subscriber('/ad_ctrl_dual', Bool, self.dual_callback)
        rospy.Subscriber('/ad_ctrl_switch_master_slave', Bool, self.switch_master_slave_callback)
        # 期望力
        rospy.Subscriber("/ad_ctrl_left_des_force", Float64MultiArray, self.des_force_left_callback, queue_size=10)
        rospy.Subscriber("/ad_ctrl_right_des_force", Float64MultiArray, self.des_force_right_callback, queue_size=10)
        # 速度
        rospy.Subscriber('/left_ad_vel', Float64MultiArray, self.left_vel_cmd_callback)
        rospy.Subscriber('/right_ad_vel', Float64MultiArray, self.right_vel_cmd_callback)


        # 订阅主从的相对关系

        self._left_vel_msg = Float32MultiArray()
        self._right_vel_msg = Float32MultiArray()


    def enable_speedl(self, enable: bool = True):
        try:
            rospy.wait_for_service(self.enable_speedl_service_name)
            client = rospy.ServiceProxy(self.enable_speedl_service_name, SetBool)
            
            resp = client(SetBoolRequest(data=enable))
            if resp.success:
                # state = "启用" if enable else "禁用"
                rospy.loginfo(f"speedl success")
            else:
                rospy.logerr(f"speedl failed")
            return resp.success
        except rospy.ServiceException as e:
            rospy.logerr(f"service speedl failed: {e}")
            return False
        except rospy.ROSException as e:
            rospy.logerr(f"ros speedl failed: {e}")
            return False

    def left_vel_cmd_callback(self, data):
        self.ad_para.left_vel_des = data.data
        self.ad_para.left_num = 0
        self.left_resample.gen(self.ad_para.left_cmd_old, self.ad_para.left_vel_des, self.ad_para.resample_time)

    def right_vel_cmd_callback(self, data):
        self.ad_para.right_vel_des = data.data
        self.ad_para.right_num = 0
        self.right_resample.gen(self.ad_para.right_cmd_old, self.ad_para.right_vel_des, self.ad_para.resample_time)

    def des_force_left_callback(self, data):
        self.ad_para.left_des_force = np.array(data.data[0:3])

    def des_force_right_callback(self, data):
        self.ad_para.right_des_force = np.array(data.data[0:3])

    def left_tcp_recived_callback(self, data):
        left_tcp_recived = data
        rot = R.from_quat(np.array([left_tcp_recived.quaternion.x,left_tcp_recived.quaternion.y,left_tcp_recived.quaternion.z,left_tcp_recived.quaternion.w]))
        self.left_rot = rot.as_matrix()
        left_rot_vector = rot.as_rotvec()
        self.ad_para.left_tcp = np.array([
            left_tcp_recived.position.x,
            left_tcp_recived.position.y,
            left_tcp_recived.position.z,
            left_rot_vector[0],
            left_rot_vector[1],
            left_rot_vector[2]
            ])

    def right_tcp_recived_callback(self, data):
        right_tcp_recived = data
        rot = R.from_quat(np.array([right_tcp_recived.quaternion.x,right_tcp_recived.quaternion.y,right_tcp_recived.quaternion.z,right_tcp_recived.quaternion.w]))
        self.right_rot = rot.as_matrix()
        right_rot_vector = rot.as_rotvec()
        self.ad_para.right_tcp = np.array([
            right_tcp_recived.position.x,
            right_tcp_recived.position.y,
            right_tcp_recived.position.z,
            right_rot_vector[0],
            right_rot_vector[1],
            right_rot_vector[2]
            ])

    def enable_callback(self, msg):
        self.ad_para.enable = msg.data 
        if self.ad_para.enable:
            self.enable_speedl(self.ad_para.enable)
            self.run_speedL([0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0.])
        else:
            # 先停止速度控制
            self.run_speedL([0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0.])
            # 再停止导纳控制
            self.enable_speedl(False)

    def single_callback(self, data):
        self.ad_para.single_left = data.data

    def dual_callback(self, data):
        self.ad_para.single = not data.data

    def switch_master_slave_callback(self, data):
        self.ad_para.master_slave = data.data

    def left_force_callback(self, data):
        self.ad_para.update_left = True
        data.wrench.force.z = 0.0
        self.ad_para.left_force = data

    def right_force_callback(self, data):
        self.ad_para.update_right = True
        data.wrench.force.z = 0.0
        self.ad_para.right_force = data

    def run_speedL(self, left_tar_v, right_tar_v):
        msg = SpeedL()

        speed = list(left_tar_v) + list(right_tar_v)
        speed = [0 if abs(v) < 0.00001 else v for v in speed]
        msg.tcp_speed = speed
        msg.acc = 0.3
        self.dual_v_publisher.publish(msg)

    # 左臂导纳 右臂速度跟随（双臂同速，速度来自左臂）
    # TODO： 主从模式 从臂的速度需要计算的得到
    def run_speedl_left(self, left_tar_v):
        msg = SpeedL()
        speed = list(left_tar_v) * 2
        speed = [0 if abs(v) < 0.00001 else v for v in speed]
        msg.tcp_speed = speed
        msg.acc = 0.3
        self.dual_v_publisher.publish(msg)

    # 右臂导纳 左臂速度跟随（双臂同速，速度来自右臂）
    def run_speedl_right(self, right_tar_v):
        msg = SpeedL()
        speed = list(right_tar_v) * 2
        speed = [0 if abs(v) < 0.00001 else v for v in speed]
        msg.tcp_speed = speed
        msg.acc = 0.3
        self.dual_v_publisher.publish(msg)
        

    # 16 - waist-z  17 - waist-y
    def joint_state_callback(self, msg):
        waist_z = msg.position[17]
        waist_y = msg.position[18]
        self.ad_para.waist_rot = np.array([
            [np.cos(waist_z), -np.sin(waist_z), 0],
            [np.sin(waist_z),  np.cos(waist_z), 0],
            [0, 0, 1]
        ])@np.array([
            [np.cos(waist_y), 0, np.sin(waist_y)],
            [0, 1, 0],
            [-np.sin(waist_y), 0, np.cos(waist_y)]
        ])  
        self.ad_para.waist_rot_6x6 = np.block([
                [self.ad_para.waist_rot, np.zeros((3, 3))],
                [np.zeros((3, 3)), self.ad_para.waist_rot]
            ])
        self.ad_para.waist_rot_6x6_inv = np.linalg.inv(self.ad_para.waist_rot_6x6)

    def apply_deadband(self, arr, threshold):
        mask = np.abs(arr) < threshold
        arr[mask] = 0
        arr[~mask] -= np.sign(arr[~mask]) * threshold

if __name__ == "__main__":
    rospy.init_node('ad_ctrl_rc_v1', anonymous=True)
    
    # 在主函数中创建共享的实例
    ad_para = Admittance_Para()
    left_resample = ReSample_6D()
    right_resample = ReSample_6D()
    
    # 传入 Data_Process
    controller = Data_Process(ad_para, left_resample, right_resample)
    
    init = True
    once = True
    rate = rospy.Rate(ad_para.frequency) 
    rospy.sleep(1.0)

    # 传入 Admittance
    ad_left = Admittance(ad_para)
    ad_left.set_coef(ad_para.left_para_K, ad_para.left_para_D, ad_para.left_para_M, 1.0 / ad_para.frequency)
    ad_right = Admittance(ad_para)
    ad_right.set_coef(ad_para.right_para_K, ad_para.right_para_D, ad_para.right_para_M, 1.0 / ad_para.frequency)
    while not rospy.is_shutdown():  
        if not ad_para.enable:
            if once:
                print('waiting enable...')
                if not init:
                    controller.run_speedL([0,0,0,0,0,0],[0,0,0,0,0,0])
                init = False
                once = False
            rospy.sleep(1)
            continue
        else:
            if not once:
                print("ad ctrl running...")
                once = True
        if not (ad_para.update_left and ad_para.update_right):
            rate.sleep()
            continue
        ad_para.update_left = False
        ad_para.update_right = False
        
        ad_para.left_cmd = left_resample.get(ad_para.left_num, ad_para.frequency)
        ad_para.left_num += 1
        ad_para.left_cmd_old = ad_para.left_cmd

        ad_para.right_cmd = right_resample.get(ad_para.right_num, ad_para.frequency)
        ad_para.right_num += 1
        ad_para.right_cmd_old = ad_para.right_cmd
        
        left_wrench = ad_para.left_force.wrench
        right_wrench = ad_para.right_force.wrench

        left_force_adjust = np.array([
            left_wrench.force.x + ad_para.left_des_force[0],
            left_wrench.force.y + ad_para.left_des_force[1],
            left_wrench.force.z + ad_para.left_des_force[2],
            left_wrench.torque.x,
            left_wrench.torque.y,
            left_wrench.torque.z
        ])
        right_force_adjust = np.array([
            right_wrench.force.x + ad_para.right_des_force[0],
            right_wrench.force.y + ad_para.right_des_force[1],
            right_wrench.force.z + ad_para.right_des_force[2],
            right_wrench.torque.x,
            right_wrench.torque.y,
            right_wrench.torque.z
        ])

        G_l = np.array([0,0,-ad_para.left_weight * 9.81])
        G_r = np.array([0,0,-ad_para.right_weight * 9.81])
        left_force_adjust[-3:] = left_force_adjust[-3:]-np.cross((controller.left_rot @ ad_para.left_force_length), G_l)
        right_force_adjust[-3:] = right_force_adjust[-3:]-np.cross((controller.right_rot @ ad_para.right_force_length), G_r)

        controller.apply_deadband(left_force_adjust[:3], ad_para.force_cal_thld)
        controller.apply_deadband(right_force_adjust[:3], ad_para.force_cal_thld)

        controller.apply_deadband(left_force_adjust[3:], ad_para.torque_cal_thld)
        controller.apply_deadband(right_force_adjust[3:], ad_para.torque_cal_thld)

        left_force_adjust[:3] = ad_para.sensor_force_ratio * left_force_adjust[:3]
        left_force_adjust[3:] =  ad_para.sensor_torque_ratio * left_force_adjust[3:]
        right_force_adjust[:3] = ad_para.sensor_force_ratio * right_force_adjust[:3]
        right_force_adjust[3:] =  ad_para.sensor_torque_ratio * right_force_adjust[3:]
        
        left_vel_cmd = ad_left.ad_ctrl(v_des = ad_para.left_cmd, f_ext = left_force_adjust)
        right_vel_cmd = ad_right.ad_ctrl(v_des = ad_para.right_cmd, f_ext = right_force_adjust)


        '''
        单臂导纳 (single=True):
            single_left=True:  左臂导纳
            single_left=False: 右臂导纳
        双臂导纳 (single=False):
            master_slave=False:  双臂模式，双都导纳
            master_slave=True: 主从模式
                single_left=True:  左臂导纳 右臂速度跟随
                single_left=False: 右臂导纳 左臂速度跟随
        '''
        if ad_para.single:
            if ad_para.single_left:
                controller.run_speedL(ad_para.speedL_ratio * left_vel_cmd, ad_para.right_cmd)
                rospy.loginfo("left speedL: %s", np.round(ad_para.speedL_ratio * left_vel_cmd, 5))
            else:
                controller.run_speedL(ad_para.left_cmd, ad_para.speedL_ratio * right_vel_cmd)
                rospy.loginfo("right speedL: %s", np.round(ad_para.speedL_ratio * right_vel_cmd, 3))
        else:
            if not ad_para.master_slave:
                controller.run_speedL(ad_para.speedL_ratio * left_vel_cmd, ad_para.speedL_ratio * right_vel_cmd)
                rospy.loginfo("left speedL: %s", np.round(ad_para.speedL_ratio * left_vel_cmd, 3))
                rospy.loginfo("right speedL: %s", np.round(ad_para.speedL_ratio * right_vel_cmd, 3))
            else:
                if ad_para.single_left:
                    controller.run_speedl_left(ad_para.speedL_ratio * left_vel_cmd)
                    rospy.loginfo("dual master-slave left lead: %s", np.round(ad_para.speedL_ratio * left_vel_cmd, 3))
                else:
                    controller.run_speedl_right(ad_para.speedL_ratio * right_vel_cmd)
                    rospy.loginfo("dual master-slave right lead: %s", np.round(ad_para.speedL_ratio * right_vel_cmd, 3))
        rate.sleep()