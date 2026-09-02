import sys
import os

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

import numpy as np
import rospy
# from utils.get_aruco_pose_v1 import get_aruco_pose, aruco_single_pose_det, get_aruco_pose_v1
from utils.get_aruco_pose_v1 import get_aruco_pose_v1
from pbvs import PBVS
from utils.tools import transform_velocity, transform_velocity_v1, transform_velocity_v2
from utils.vis import CamImg, VisImg
from utils.tf_tranform import TF_transform
import time
import multiprocessing
from naviai_controller import NaviController, ArmGroup, HandType
from std_msgs.msg import Float64MultiArray, Bool
from geometry_msgs.msg import WrenchStamped
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import cv2

'''
    PBVS对齐相关的
     增加了轨迹图像保存功能

    把原有的代码进行了拆分

'''

class Operation:
    def __init__(self, controller, cam_name, vis=False):
        self.controller = controller
        self.cam2chest = np.array([
            [-0.01121016, -0.51776829,  0.85544744,  0.11195075],
            [-0.99949497,  0.03124157,  0.00581146,  0.03255672],
            [-0.02973451, -0.85495027, -0.51785703,  0.41386475],
            [ 0.0,         0.0,         0.0,         1.0        ]]
        )
        self.mtx = np.array([[913.4111328125, 0, 646.2576904296875], 
                        [0, 913.0277099609375,  370.2160339355469], 
                        [0, 0, 1]])
        self.dist = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        s = 0.03
        # 需要调整和修改，作为aruco码的坐标
        self.objPoints = np.array([[-s/2, s/2, 0],
                            [s/2, s/2, 0],
                            [s/2, -s/2, 0],
                            [-s/2, -s/2, 0]], dtype=np.float32).reshape(-1, 3)
        self.pbvs = PBVS(self.objPoints)

        self.aruco2cam = None
        self.current_pose = None
        self.aruco2right_arm = None
        self.action_pbvs_right = None
        self.distance = None
   
        self.cam_name = cam_name
        self.cam_img = CamImg(cam_name)

        self.tf_handler = TF_transform()
        self.cam2base = None
        self.get_cam2base()
        # 创建导纳的话题接口
        self.ad_right_vel_publisher = rospy.Publisher("/right_ad_vel", Float64MultiArray, queue_size=10)
        self.ad_left_vel_publisher = rospy.Publisher("/left_ad_vel", Float64MultiArray, queue_size=10)
        self.ad_dual_enable_pub = rospy.Publisher("/ad_ctrl_dual", Bool, queue_size=1)
        self.ad_ctrl_rc_enable = rospy.Publisher("/ad_ctrl_enable", Bool, queue_size=1)
        self.ad_left_switch = rospy.Publisher("/ad_ctrl_left", Bool, queue_size=1)
        self.ad_right_des_force_pub = rospy.Publisher("/ad_ctrl_right_des_force", Float64MultiArray, queue_size=10)
        self.ad_left_des_force_pub = rospy.Publisher("/ad_ctrl_left_des_force", Float64MultiArray, queue_size=10)
        # 订阅力传感器话题
        self.right_force = WrenchStamped()
        self.left_force = WrenchStamped()
        rospy.Subscriber("/wrist_force_control/right_arm_compensated_force", WrenchStamped, self.right_force_callback, queue_size=10)
        rospy.Subscriber("/wrist_force_control/left_arm_compensated_force", WrenchStamped, self.left_force_callback, queue_size=10)

        self.img_dict = multiprocessing.Manager().dict()
        self.img_dict['img'] = self.cam_img.get_img() # 初始化
        if vis:
            self.vis_img = VisImg(self.img_dict)
            self.vis_img.start()


    def get_cam2base(self):
        while not rospy.is_shutdown():
            try:
                chest2base = self.tf_handler.get_tf("BASE", "WAIST_PITCH")
                if chest2base is None:
                    rospy.logwarn("chest2base is None, retry get_cam2base...")
                    time.sleep(0.1)
                    continue
            
                self.cam2base = chest2base @ self.cam2chest
                return self.cam2base
            except Exception as e:
                rospy.logerr(f"get_cam2base error: {e}")
                time.sleep(0.1)
        return None
   

    def right_force_callback(self, data):
        self.right_force = data

    def left_force_callback(self, data):
        self.left_force = data

    def single_arm_operation(self, target_pose = None, dis_threshold = 0.01, angle_threshold = 1.0):
        '''
            pbvs 控制单臂 采用speedl控制
        '''
        if target_pose is None:
            raise ValueError("target_pose is None")
        self.get_cam2base()
        self.controller.enable_speedl()
        while not rospy.is_shutdown():
            t0 = time.time()
            image = self.cam_img.get_img()
            # 如果pose为None，则停止运动
            self.current_pose, image= get_aruco_pose_v1(image, self.mtx, self.dist, self.objPoints)
            if self.current_pose is None:
                time.sleep(0.1)
                print("no aruco pose")
                self.controller.speedl([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], ArmGroup.RIGHT, 0.05)
                continue

            self.aruco2cam = self.current_pose
            self.img_dict['img'] = image  # 可视化传图

            # 计算pbvs速度
            action_cam, _ = self.pbvs.cal_action_curve(self.current_pose, target_pose)
            rot = - action_cam[3:]
            trans = - action_cam[:3] + np.cross(rot, self.aruco2cam[:3, 3])
            action_aruco2cam = np.concatenate([trans, rot])

            # 计算两者的差值，用于判断是否到达目标位姿
            cur2tar = np.linalg.inv(target_pose) @ self.current_pose
            self.distance = np.linalg.norm(cur2tar[:3, 3])
            rotation_matrix = cur2tar[:3, :3]
            rotation_vector = R.from_matrix(rotation_matrix).as_rotvec()
            rotation_error_deg = np.degrees(np.linalg.norm(rotation_vector))

            self.update_transform()

            action_pbvs_right = transform_velocity(action_aruco2cam, self.aruco2cam, self.aruco2right_arm, self.cam2base)
            # 如果速度大于0.3，则进行按比例缩放，防止速度过大
            if max(abs(action_pbvs_right)) > 0.3:
                action_pbvs_right = action_pbvs_right / (max(abs(action_pbvs_right)) / 0.3)
            
            action_pbvs_right = np.array([0.0 if abs(v) < 0.0001 else v for v in action_pbvs_right])
            print(f"action_pbvs_right: {action_pbvs_right}")
            self.controller.speedl((action_pbvs_right*0.1).tolist(), ArmGroup.RIGHT, 0.05)
            
            print(f"distance:{self.distance}，rotation_error_deg:{rotation_error_deg}")
            if self.distance <= dis_threshold and rotation_error_deg <= angle_threshold:
                print("end")
                self.controller.speedl([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], ArmGroup.RIGHT, 0.05)
                time.sleep(0.5)
                print("enable_speedl False")
                self.controller.enable_speedl(False)
                print("enable_speedl False end")
                break
            
            print("----------------------------------")
            self.current_pose = None
            t1 = time.time()
            print(f"time: {t1 - t0}")
            time.sleep(max(0.3 - (t1 - t0), 0))

        rospy.loginfo("pbvs operation end")

    def single_arm_pbvs(self, arm = ArmGroup.RIGHT, target_pose = None, dis_threshold = 0.001, angle_threshold = 0.5):
        '''
            pbvs 控制单臂 采用speedl控制
            相比于single_arm_operation，增加了target_pose参数，用于指定目标位姿
        '''
        if target_pose is None:
            raise ValueError("target_pose is None")
        self.get_cam2base()
        rospy.loginfo(f"arm {arm} begin pbvs operation")
        trajectory_points = []
        _last_image = None
        self.controller.enable_speedl()
        while not rospy.is_shutdown():
            t0 = time.time()
            image = self.cam_img.get_img()
            # 如果pose为None，则停止运动
            self.current_pose, image = get_aruco_pose_v1(image, self.mtx, self.dist, self.objPoints)
            if self.current_pose is None:
                time.sleep(0.1)
                rospy.logwarn("no aruco pose")
                self.controller.speedl([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], arm, 0.05)
                continue

            trajectory_points.append(self.current_pose[:3, 3].tolist())
            aruco2cam = self.current_pose
            self.img_dict['img'] = image  # 可视化传图

            # 计算pbvs速度
            action_cam, _ = self.pbvs.cal_action_curve(self.current_pose, target_pose)
            rot = - action_cam[3:]
            trans = - action_cam[:3] + np.cross(rot, aruco2cam[:3, 3])
            action_aruco2cam = np.concatenate([trans, rot])

            # 计算两者的差值，用于判断是否到达目标位姿
            cur2tar = np.linalg.inv(target_pose) @ self.current_pose
            self.distance = np.linalg.norm(cur2tar[:3, 3])
            rotation_matrix = cur2tar[:3, :3]
            rotation_vector = R.from_matrix(rotation_matrix).as_rotvec()
            rotation_error_deg = np.degrees(np.linalg.norm(rotation_vector))

            arm_tcp_matrix = self.controller.get_tcp_matrix(arm)
            arm2cam = np.linalg.inv(self.cam2base) @ arm_tcp_matrix
            cam2arm = np.linalg.inv(arm_tcp_matrix) @ self.cam2base
            aruco2arm = cam2arm @ aruco2cam

            action_pbvs_arm = transform_velocity(action_aruco2cam, aruco2cam, aruco2arm, self.cam2base)
            # 如果速度大于0.3，则进行按比例缩放，防止速度过大
            if max(abs(action_pbvs_arm)) > 0.3:
                action_pbvs_arm = action_pbvs_arm / (max(abs(action_pbvs_arm)) / 0.3)
            action_pbvs_arm = np.array([0.0 if abs(v) < 0.0001 else v for v in action_pbvs_arm])
            rospy.loginfo(f"action_pbvs_arm: {action_pbvs_arm}")
            self.controller.speedl((action_pbvs_arm*0.2).tolist(), arm, 0.1)
            
            rospy.loginfo(f"distance:{self.distance}，rotation_error_deg:{rotation_error_deg}")
            if self.distance <= dis_threshold and rotation_error_deg <= angle_threshold:
                _last_image = image.copy()
                self.controller.speedl([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], arm, 0.05)
                time.sleep(0.5)
                self.controller.enable_speedl(False)
                rospy.loginfo("pbvs operation end and enable_speedl False")
                break
        
            rospy.loginfo("----------------------------------")
            self.current_pose = None
            t1 = time.time()
            rospy.loginfo(f"time: {t1 - t0}")
            time.sleep(max(0.3 - (t1 - t0), 0))

        # 在最后一帧图像上绘制轨迹并更新可视化
        if _last_image is not None and len(trajectory_points) >= 2:
            img_with_traj = self.draw_trajectory_on_image(_last_image, trajectory_points)
            if img_with_traj is not None:
                save_path = f"trajectory_points_{arm}.png"
                cv2.imwrite(save_path, img_with_traj)
                rospy.loginfo(f"trajectory points saved to {save_path}")
        rospy.loginfo("pbvs operation end")

    def dual_arm_operation(self, target_pose = None, dis_threshold = 0.01, angle_threshold = 1.0):
        '''
            pbvs 控制主臂，双臂控制从臂 采用speedl控制
        '''
        if target_pose is None:
            raise ValueError("target_pose is None")
        # speedl 使能
        self.right_tcp_matrix = self.controller.get_tcp_matrix(ArmGroup.RIGHT)
        self.left_tcp_matrix = self.controller.get_tcp_matrix(ArmGroup.LEFT)
        left2right_arm = np.linalg.inv(self.right_tcp_matrix) @ self.left_tcp_matrix
        self.controller.enable_speedl(True)
        while not rospy.is_shutdown():
            t0 = time.time()
            image = self.cam_img.get_img()
            # 如果pose为None，则停止运动
            self.current_pose, image = get_aruco_pose_v1(image, self.mtx, self.dist, self.objPoints)
            if self.current_pose is None:
                time.sleep(0.1)
                print("no aruco pose")
                self.controller.speedl([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], ArmGroup.DUAL, 0.05)
                continue
            self.aruco2cam = self.current_pose
            self.img_dict['img'] = image  # 可视化传图
            self.update_transform()  # 更新变换矩阵  ----->  这个可能不需要每次都更新 后续进行优化测试


            # 计算两者的差值，用于判断是否到达目标位姿
            cur2tar = np.linalg.inv(target_pose) @ self.current_pose
            self.distance = np.linalg.norm(cur2tar[:3, 3])

            # 计算cam 速度
            action_cam, _ = self.pbvs.cal_action_curve(self.current_pose, target_pose)
            rot = - action_cam[3:]
            trans = - action_cam[:3] + np.cross(rot, self.aruco2cam[:3, 3])
            # aruco2cam 速度
            action_aruco2cam = np.concatenate([trans, rot])

            #  主臂的速度
            # 计算right_arm 速度
            action_pbvs_right = transform_velocity_v1(action_aruco2cam, self.aruco2cam, self.right2cam, self.cam2base)
            # 如果速度大于0.3，则进行按比例缩放，防止速度过大
            # if max(abs(action_pbvs_right)) > 0.3:
            #     action_pbvs_right = action_pbvs_right / (max(abs(action_pbvs_right)) / 0.3)
            # 需要接入导纳控制器
            print(f"action_pbvs_right: {action_pbvs_right}")

            # 计算从臂速度
            action_pbvs_left = transform_velocity_v2(action_pbvs_right, self.right_tcp_matrix, left2right_arm)
            speedl_dual = np.concatenate([action_pbvs_left, action_pbvs_right])
            # 限制速度
            if max(abs(speedl_dual)) > 0.3:
                speedl_dual = speedl_dual / (max(abs(speedl_dual)) / 0.3)
            # 限制速度
            speedl_dual = np.array([ 0.0 if abs(v) < 0.0001 else v for v in speedl_dual])
            
            self.controller.speedl((speedl_dual*0.1).tolist(), ArmGroup.DUAL, 0.05)
            print(f"current_pose: {self.current_pose} \n target_pose: {target_pose} \n distance: {self.distance}")
            print(f"dual_speedl: {speedl_dual}")
            # 如果距离小于阈值，则停止运动
            if self.distance <= dis_threshold:
                print("end")
                self.controller.stop_speedl(ArmGroup.DUAL)
                self.controller.enable_speedl(False)
                break
            self.current_pose = None
            t1 = time.time()
            print(f"time: {t1 - t0}")
            time.sleep(max(0.1 - (t1 - t0), 0))

    def draw_trajectory_on_image(self, image=None, points=None, color=(255, 255, 0), thickness=2, radius=4, save_path=None):
        """
        将整个 PBVS 过程中的运动点（二维码在 base 下的轨迹）投影到图像上并绘制。
        :param image: 要绘制的图像，None 则使用最后一次保存的 _last_image
        :param color: 轨迹颜色 BGR，默认绿色
        :param thickness: 连线粗细
        :param radius: 轨迹点圆半径
        :param save_path: 若给出路径则保存绘制后的图像
        :return: 绘制了轨迹的图像，若无有效轨迹或图像则返回原图或 None
        """
        if image is None:
            image = self.img_dict['img']
        if image is None or len(points) < 2:
            return image
        img = image.copy()
        # 将 base 下 3D 点投影到当前相机图像（cam2base 对应此图时的相机位姿）
        pts_cam = np.array(points, dtype=np.float32)
        pts_2d, _ = cv2.projectPoints(pts_cam, np.zeros((3, 1)), np.zeros((3, 1)), self.mtx, self.dist)
        pts_2d = pts_2d.reshape(-1, 2).astype(np.int32)
        # 裁剪到图像内
        h, w = img.shape[:2]
        for i in range(len(pts_2d)):
            x, y = pts_2d[i, 0], pts_2d[i, 1]
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(img, (x, y), radius, color, -1)
        for i in range(len(pts_2d) - 1):
            pt1 = tuple(pts_2d[i])
            pt2 = tuple(pts_2d[i + 1])
            if 0 <= pt1[0] < w and 0 <= pt1[1] < h and 0 <= pt2[0] < w and 0 <= pt2[1] < h:
                cv2.line(img, pt1, pt2, color, thickness)
        if save_path:
            cv2.imwrite(save_path, img)
        return img


    def update_transform(self):

        #  如果后续腰部不一移动的话 可以只计算一次
        # self.chest2base = self.tf_handler.get_tf("BASE", "CHEST")
        # self.cam2base = self.chest2base @self.cam2chest 
        # 计算aruco2right_arm
        self.right_tcp_matrix = self.controller.get_tcp_matrix(ArmGroup.RIGHT)
        self.right2cam = np.linalg.inv(self.cam2base) @ self.right_tcp_matrix
        self.cam2right_arm = np.linalg.inv(self.right_tcp_matrix) @ self.cam2base
        self.aruco2right_arm = self.cam2right_arm @ self.current_pose

        # 计算 left2right
        self.left_tcp_matrix = self.controller.get_tcp_matrix(ArmGroup.LEFT)
        # self.left2right_arm = np.linalg.inv(self.right_tcp_matrix) @ self.left_tcp_matrix
        
    
    def pub_ad_ctrl_dual(self, left_vel, right_vel):
        left_vel = [0.0 if abs(v) < 0.0001 else v for v in left_vel]
        right_vel = [0.0 if abs(v) < 0.0001 else v for v in right_vel]
        self.ad_ctrl_rc_left.publish(Float64MultiArray(data=left_vel))
        self.ad_ctrl_rc_right.publish(Float64MultiArray(data=right_vel))

    # 测试二维码姿态
    def ts_aruco_pose(self):
        
        while not rospy.is_shutdown():
            image = self.cam_img.get_img()
            # print(image)
            # aruco2cam, image = aruco_single_pose_det(image, self.mtx, self.dist)
            aruco2cam, image = get_aruco_pose_v1(image, self.mtx, self.dist, self.objPoints)
            print(f"aruco2cam:{aruco2cam}")

            self.img_dict['img'] = image
            time.sleep(0.1)

    def get_target_pose(self, skip_frames=5):
        '''
            获取目标位姿，跳过前几帧稳定检测结果
        '''
        target_pose = None
        valid_count = 0
        while not rospy.is_shutdown():
            image = self.cam_img.get_img()
            aruco2cam, image = get_aruco_pose_v1(image, self.mtx, self.dist, self.objPoints)
            if aruco2cam is not None:
                valid_count += 1
                if valid_count <= skip_frames:
                    time.sleep(0.1)
                    continue
                target_pose = aruco2cam
                break
            time.sleep(0.1)

        return target_pose

    def ts_dual_arm_control(self):

        # 给主臂一个速度
        count =  15
        # 右臂
        right_tcp_matrix = self.controller.get_tcp_matrix(ArmGroup.RIGHT)
        # 应该采用相对关系
        left2right_arm = np.linalg.inv(right_tcp_matrix) @ self.controller.get_tcp_matrix(ArmGroup.LEFT)
        self.controller.enable_speedl(True)
        screw_direction = np.array([-0.9970157, 0.01299241, 0.07609786, 0.0, 0.0, 0.0])  # 螺丝方向向量
        for i in range(count):
            t0 = time.time()
            # 给定固定速度
            # action_pbvs_right = [0.00, 0.001, 0.0, 0.0, 0.0, 0.0]
            action_pbvs_right = screw_direction * (0.01)
            # 计算双臂基于base的变换矩阵
            right_tcp_matrix = self.controller.get_tcp_matrix(ArmGroup.RIGHT)
            action_pbvs_left = transform_velocity_v2(action_pbvs_right, right_tcp_matrix, left2right_arm)
            # 计算双臂速度
            speed_dual = np.concatenate([action_pbvs_left, action_pbvs_right])
            # 限制速度，防止速度过大
            if max(abs(speed_dual)) > 0.3:
                speed_dual = speed_dual / (max(abs(speed_dual)) / 0.3)
            # 执行速度
            speed_dual = [ 0.0 if abs(v) < 0.0001 else v for v in speed_dual]
            print(f"speed_dual: {speed_dual}")
            speed_dual = np.array(speed_dual)
            self.controller.speedl((speed_dual).tolist(), ArmGroup.DUAL, 0.05)
            
            # 等待0.1秒
            print(f"time: {time.time() - t0}")
            time.sleep(max(0.1 - (time.time() - t0), 0))

        # 关闭速度控制
        self.controller.stop_speedl(ArmGroup.DUAL)
        self.controller.enable_speedl(False)
        



    def ts_ad_ctrl(self):
        '''
            单手导纳螺丝测试
        '''
        # 1. 开启右手导纳
        self.ad_ctrl_rc_enable.publish(Bool(data=True))
        self.ad_left_switch.publish(Bool(data=False))
        self.force_threshold = 50
        init_force = self.right_force.wrench.force
        self.ad_right_des_force_pub.publish(Float64MultiArray(data=[-15.0, 0.0, 0.0]))
        speed_max = 0.05
        speed_min = 0.01
        while not rospy.is_shutdown():
            # 2. 发布右手期望速度 当前的期望力是-2
            t0 = time.time()
            screw_vel = np.array([-0.9970157, 0.01299241, 0.07609786, 0.0, 0.0, 0.0]) * 0.08
            self.ad_right_vel_publisher.publish(Float64MultiArray(data=screw_vel.tolist()))

            # 3. 获取右手力传感器数据 
            # z 或者 其他的方向
            force = self.right_force.wrench.force
            rospy.loginfo(f"right_force: x={force.x:.4g}, y={force.y:.4g}, z={force.z:.4g}")
            if self.right_force  is not None and abs(self.right_force.wrench.force.x - init_force.x) >  self.force_threshold:
                # 4. 超出阈值则 停止导纳 并停止速度控制
                # self.controller.speedl([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], ArmGroup.RIGHT, 0.05)
                self.ad_right_des_force_pub.publish(Float64MultiArray(data=[0.0, 0.0, 0.0]))
                self.ad_ctrl_rc_enable.publish(Bool(data=False))
                break
            print(f"time: {time.time() - t0}")
            time.sleep(max(0.1 - (time.time() - t0), 0))

        print("end")

    def ts_ad_ctrl_with_state_machine(self):
        '''
            单手导纳螺丝测试 - 带状态机的动态速度和力控制
            
            功能说明：
            - 接近阶段：正转等待接触螺丝
            - 恒速阶段：保持速度完成卡入/拧紧
            - 完成：达到目标力矩后停止
        '''
        STATE_APPROACHING = "approaching"      # 接近阶段：正转等待接触
        STATE_DRIVING = "driving"              # 恒速工作阶段：保持速度完成卡入/拧紧
        STATE_COMPLETED = "completed"           # 完成
        
        # 速度参数
        speed_max = 0.03                       # 最大速度
        speed_min = 0.01                       # 最小速度
        speed_approach = speed_min             # 接近阶段速度系数
            
        des_force_init = -3.0                 # 初始期望力
        contact_force_threshold = 5.0         # 接触检测阈值（力突然增大的阈值）
        contact_check_window = 3              # 接触检测窗口（连续N次检测）
        
        # 恒速阶段参数
        speed_drive = 0.03
        drive_duration = 3.0
        drive_start_time = None
        drive_complete_y_threshold = 6.0
        contact_count = 0 

        force_data = []
        speed_data = []
        
        # ========== 初始化 ==========
        self.ad_ctrl_rc_enable.publish(Bool(data=True))
        self.ad_left_switch.publish(Bool(data=False))   # 右臂导纳
        
        init_force = self.right_force.wrench.force # 初始力传感器数据
        force_data.append([init_force.x, init_force.y, init_force.z])
        # NOTE 螺丝的方向向量需要修改
        # [-0.99994391 -0.00776298  0.00720499] 2026.5.7
        # screw_direction = np.array([-0.99092, -0.01698769, 0.13337532, 0.0, 0.0, 0.0])  # 螺丝方向向量
        screw_direction = np.array([-0.99994391, -0.00776298,  0.00720499, 0.0, 0.0, 0.0])  # 螺丝方向向量
        self.ad_right_des_force_pub.publish(Float64MultiArray(data=[des_force_init, 0.0, 0.0])) # 设置初始期望力
        
        state = STATE_APPROACHING
        rospy.loginfo(f"[{state}] 开始拧螺丝流程，初始力: x={init_force.x:.4g}")
        while not rospy.is_shutdown():
            t0 = time.time()
            # 记录速度
            speed_now = self.controller.get_tcp_speed(ArmGroup.RIGHT)
            speed_data.append(speed_now)
            force = self.right_force.wrench.force
            force_data.append([force.x, force.y, force.z])
            force_x_diff = abs(force.x - init_force.x)
            current_time = time.time()
            rospy.loginfo(f"[{state}] force: x={force.x:.4g}, diff_x={force_x_diff:.4g}, y={force.y:.4g}, z={force.z:.4g}")
            # ========== 渐进接近螺丝阶段 ==========
            if state == STATE_APPROACHING:
                screw_vel = screw_direction * speed_approach  # 渐进接近螺丝
                speed_approach = min(speed_approach + 0.005, speed_max)
                rospy.loginfo(f"[{state}] speed_approach: {speed_approach:.4g}")
                self.ad_right_vel_publisher.publish(Float64MultiArray(data=screw_vel.tolist())) # 发布速度
                
                if force_x_diff > contact_force_threshold:
                    contact_count += 1
                    # 进入反转的条件  反力大于阈值 或者 产生旋转而产生y轴方向的力
                    if contact_count >= contact_check_window or abs(force.y - init_force.y) > 2.0:
                        state = STATE_DRIVING
                        rospy.loginfo(f"[{state}] 检测到接触，进入恒速工作阶段")
                        drive_start_time = time.time()
                        contact_count = 0
                else:
                    contact_count = 0
            
            # ========== 恒速工作阶段，螺丝枪反转后正转拧紧 ==========
            elif state == STATE_DRIVING:
                screw_vel = screw_direction * speed_drive  # 速度保持匀速
                rospy.loginfo(f"[{state}] speed_drive: {speed_drive:.4g}")
                self.ad_right_vel_publisher.publish(Float64MultiArray(data=screw_vel.tolist()))
                
                if drive_start_time is None:
                    drive_start_time = current_time
                drive_elapsed = current_time - drive_start_time
                force_y_diff = abs(force.y - init_force.y)
                # 检查恒速工作时间 或者 y轴方向的力大于6N
                if drive_elapsed >= drive_duration or force_y_diff > drive_complete_y_threshold:
                    state = STATE_COMPLETED
                    rospy.loginfo(
                        f"[{state}] 恒速工作完成，elapsed={drive_elapsed:.4g}s, "
                        f"force_y_diff={force_y_diff:.4g}"
                    )            
        
            elif state == STATE_COMPLETED:
                # 完成状态：停止速度控制
                self.ad_right_vel_publisher.publish(Float64MultiArray(data=[0.0]*6))
                self.ad_right_des_force_pub.publish(Float64MultiArray(data=[0.0]*3))
                time.sleep(2.0)
                # self.ad_right_vel_publisher.publish(Float64MultiArray(data=[0.01, 0.0, 0.0, 0.0, 0.0, 0.0]))
                # time.sleep(4.0) 
                # self.ad_right_vel_publisher.publish(Float64MultiArray(data=[0.0]*6))
                self.ad_ctrl_rc_enable.publish(Bool(data=False))
                rospy.loginfo(f"[{state}] 拧螺丝流程完成")
                break
            
            if force_x_diff > 80.0:  
                rospy.logwarn(f"force过大！紧急停止。当前力: {force_x_diff:.4g}")
                self.ad_right_vel_publisher.publish(Float64MultiArray(data=[0.0]*6))
                self.ad_right_des_force_pub.publish(Float64MultiArray(data=[0.0]*3))
                self.ad_ctrl_rc_enable.publish(Bool(data=False))
                break
            
            elapsed = time.time() - t0
            time.sleep(max(0.1 - elapsed, 0))

        print("拧螺丝流程结束")
        # ========== 数据保存与可视化 ==========
        # 保存力数据到CSV文件
        force_data = np.array(force_data)
        np.savetxt(f"force_data_{time.strftime('%Y%m%d_%H%M%S')}.csv", force_data, delimiter=",", header="x,y,z")
        # 保存速度数据到CSV文件（过滤掉未取到速度的帧）
        valid_speed_data = [s for s in speed_data if s is not None]
        speed_array = np.array(valid_speed_data) if len(valid_speed_data) > 0 else np.array([])
        if speed_array.size > 0:
            if speed_array.ndim == 1:
                speed_array = speed_array.reshape(-1, 1)
            speed_header = ",".join([f"v{i}" for i in range(speed_array.shape[1])])
            np.savetxt(
                f"speed_data_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                speed_array,
                delimiter=",",
                header=speed_header,
            )

        # 可视化力数据和速度数据
        plt.figure(figsize=(10, 9))
        plt.subplot(2, 1, 1)
        plt.title(f"Force / Speed Data {time.strftime('%Y%m%d_%H%M%S')}")
        plt.xlabel("Frame")
        plt.ylabel("Force (N)")
        plt.plot(force_data[:, 0], label="force_x")
        plt.plot(force_data[:, 1], label="force_y")
        plt.plot(force_data[:, 2], label="force_z")
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.xlabel("Frame")
        plt.ylabel("TCP Speed")
        if speed_array.size > 0:
            for i in range(speed_array.shape[1]):
                plt.plot(speed_array[:, i], label=f"speed_{i}")
            plt.legend()
        else:
            plt.text(0.5, 0.5, "No valid speed data", ha="center", va="center", transform=plt.gca().transAxes)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def set_ad_force(self, arm = ArmGroup.RIGHT, force = np.array([0.0, 0.0, 0.0])):
        if arm == ArmGroup.RIGHT:
            self.ad_right_des_force_pub.publish(Float64MultiArray(data=list(force)))
        else:
            self.ad_left_des_force_pub.publish(Float64MultiArray(data=list(force)))


    def set_ad_speed(self, arm = ArmGroup.RIGHT, speed = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])):
        if arm == ArmGroup.RIGHT:
            self.ad_right_vel_publisher.publish(Float64MultiArray(data=list(speed)))
        else:
            self.ad_left_vel_publisher.publish(Float64MultiArray(data=list(speed)))

    def ad_ctrl_with_arm(
                self, 
                arm = ArmGroup.RIGHT, 
                direction = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                des_force_init = 3.0,
                drive_duration = 3.0,
                speed_threshold = 0.001,
                ):
        '''
            单手导纳螺丝测试 - 带状态机的动态速度和力控制
            
            功能说明：
            - 接近阶段：正转等待接触螺丝
            - 恒速阶段：保持速度完成卡入/拧紧
            - 完成：达到目标力矩后停止
        '''
        STATE_APPROACHING = "approaching"      # 接近阶段：正转等待接触
        STATE_DRIVING = "driving"              # 恒速工作阶段：保持速度完成卡入/拧紧
        STATE_COMPLETED = "completed"           # 完成
        
        # 速度参数
        speed_max = 0.03                       # 最大速度
        speed_min = 0.01                       # 最小速度
        speed_approach = speed_min             # 接近阶段速度系数
            
        contact_force_threshold = 5.0         # 接触检测阈值（力突然增大的阈值）
        contact_check_window = 3              # 接触检测窗口（连续N次检测）
        
        # 恒速阶段参数
        speed_drive = 0.03
        # drive_duration = 3.0
        drive_start_time = None
        drive_complete_y_threshold = 20
        contact_count = 0 

        force_data = []
        speed_data = []
        # NOTE 螺丝的方向向量需要修改
        screw_direction = np.array(direction)  # 螺丝方向向量
        # ========== 初始化 ==========
        self.ad_ctrl_rc_enable.publish(Bool(data=True))
        if arm == ArmGroup.RIGHT:
            self.ad_left_switch.publish(Bool(data=False))   # 右臂导纳
            init_force = self.right_force.wrench.force
        else:
            self.ad_left_switch.publish(Bool(data=True))   # 左臂导纳
            init_force = self.left_force.wrench.force
           
        self.set_ad_force(arm, np.array([des_force_init, 0.0, 0.0]))
        force_data.append([init_force.x, init_force.y, init_force.z])
        
        state = STATE_APPROACHING
        rospy.loginfo(f"[{state}] 开始拧螺丝流程，初始力: x={init_force.x:.4g}")
        while not rospy.is_shutdown():
            t0 = time.time()
            # 记录速度
            speed_now = self.controller.get_tcp_speed(arm)
            speed_data.append(speed_now[:3])
            if arm == ArmGroup.RIGHT:
                force = self.right_force.wrench.force
            else:
                force = self.left_force.wrench.force
            force_data.append([force.x, force.y, force.z])
            force_x_diff = abs(force.x - init_force.x)
            current_time = time.time()
            rospy.loginfo(f"[{state}] force: x={force.x:.4g}, diff_x={force_x_diff:.4g}, y={force.y:.4g}, z={force.z:.4g}")
            # ========== 渐进接近螺丝阶段 ==========
            if state == STATE_APPROACHING:
                screw_vel = screw_direction * speed_approach  # 渐进接近螺丝
                speed_approach = min(speed_approach + 0.005, speed_max)
                rospy.loginfo(f"[{state}] speed_approach: {speed_approach:.4g}")
                # self.ad_right_vel_publisher.publish(Float64MultiArray(data=screw_vel.tolist())) # 发布速度
                self.set_ad_speed(arm, screw_vel)
                
                if force_x_diff > contact_force_threshold:
                    contact_count += 1
                    # 进入反转的条件  反力大于阈值 或者 产生旋转而产生y轴方向的力
                    if contact_count >= contact_check_window or abs(force.y - init_force.y) > 2.0:
                        state = STATE_DRIVING
                        rospy.loginfo(f"[{state}] 检测到接触，进入恒速工作阶段")
                        drive_start_time = time.time()
                        contact_count = 0
                else:
                    contact_count = 0
            
            # ========== 恒速工作阶段，螺丝枪反转后正转拧紧 ==========
            elif state == STATE_DRIVING:
                screw_vel = screw_direction * speed_drive  # 速度保持匀速
                rospy.loginfo(f"[{state}] speed_drive: {speed_drive:.4g}")
                # self.ad_right_vel_publisher.publish(Float64MultiArray(data=screw_vel.tolist()))
                self.set_ad_speed(arm, screw_vel)
                
                if drive_start_time is None:
                    drive_start_time = current_time
                drive_elapsed = current_time - drive_start_time
                force_y_diff = abs(force.y - init_force.y)
                speed_tag = False
                if arm == ArmGroup.RIGHT:
                    speed_tag = (speed_now[0] < 0 and drive_elapsed > 2.0)
                else:
                    speed_tag = (speed_now[0] > 0 and drive_elapsed > 2.0)
                # 检查恒速工作时间 或者 y轴方向的力大于6N
                # if drive_elapsed >= drive_duration or force_y_diff > drive_complete_y_threshold or speed_tag:
                if drive_elapsed >= drive_duration or force_y_diff > drive_complete_y_threshold or speed_tag:
                    state = STATE_COMPLETED
                    rospy.loginfo(
                        f"[{state}] 恒速工作完成，elapsed={drive_elapsed:.4g} / {drive_duration:.4g}s, "
                        f"force_y_diff={force_y_diff:.4g} / {drive_complete_y_threshold:.4g}, "
                        f"speed_now={speed_now[0]:.4g} / {speed_threshold:.4g}"
                    )            
        
            elif state == STATE_COMPLETED:
                # 完成状态：停止速度控制
                # self.ad_right_vel_publisher.publish(Float64MultiArray(data=[0.0]*6))
                self.set_ad_speed(arm, np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
                self.set_ad_force(arm, np.array([0.0, 0.0, 0.0]))
                time.sleep(1.0)
                # NOTE 需要反方向移出来
                # self.set_ad_speed(arm, np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0]))
                self.set_ad_speed(arm, -1*screw_direction*0.01)
                time.sleep(4.0)
                self.set_ad_speed(arm, np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
                self.ad_ctrl_rc_enable.publish(Bool(data=False))
                rospy.loginfo(f"[{state}] 拧螺丝流程完成")
                break
            
            if force_x_diff > 40.0:  
                rospy.logwarn(f"force过大！紧急停止。当前力: {force_x_diff:.4g}")
                self.set_ad_speed(arm, np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
                self.set_ad_force(arm, np.array([0.0, 0.0, 0.0]))
                self.ad_ctrl_rc_enable.publish(Bool(data=False))
                break
            
            elapsed = time.time() - t0
            time.sleep(max(0.1 - elapsed, 0))

        print("拧螺丝流程结束")
        # ========== 数据保存与可视化 ==========
        ts = time.strftime('%Y%m%d_%H%M%S')
        path = os.path.join(_PKG_DIR, "data", ts)
        os.makedirs(path, exist_ok=True)
        arm_label = "right_arm" if arm == ArmGroup.RIGHT else "left_arm"
        # 保存力数据到CSV文件
        force_data = np.array(force_data)
        np.savetxt(f"{path}/force_data_{ts}_{arm_label}.csv", force_data, delimiter=",", header="x,y,z")
        # 保存速度数据到CSV文件（过滤掉未取到速度的帧）
        valid_speed_data = [s for s in speed_data if s is not None]
        speed_array = np.array(valid_speed_data) if len(valid_speed_data) > 0 else np.array([])
        if speed_array.size > 0:
            if speed_array.ndim == 1:
                speed_array = speed_array.reshape(-1, 1)
            speed_header = ",".join([f"v{i}" for i in range(speed_array.shape[1])])
            np.savetxt(
                f"{path}/speed_data_{ts}_{arm_label}.csv",
                speed_array,
                delimiter=",",
                header=speed_header,
            )

        # 保存力/速度曲线图
        plt.figure(figsize=(10, 9))
        plt.subplot(2, 1, 1)
        plt.title(f"Force / Speed Data {ts}")
        plt.xlabel("Frame")
        plt.ylabel("Force (N)")
        plt.plot(force_data[:, 0], label="force_x")
        plt.plot(force_data[:, 1], label="force_y")
        plt.plot(force_data[:, 2], label="force_z")
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.xlabel("Frame")
        plt.ylabel("TCP Speed")
        if speed_array.size > 0:
            for i in range(speed_array.shape[1]):
                plt.plot(speed_array[:, i], label=f"speed_{i}")
            plt.legend()
        else:
            plt.text(0.5, 0.5, "No valid speed data", ha="center", va="center", transform=plt.gca().transAxes)
        plt.grid(True)
        plt.tight_layout()
        plot_path = f"{path}/force_speed_plot_{ts}_{arm_label}.png"
        plt.savefig(plot_path)
        plt.close()
        rospy.loginfo(f"force/speed plot saved to {plot_path}")


    def ts_ad_ctrl_with_dual_arm_control(self):
        '''
            双臂导纳螺丝测试 有点问题

            现改左臂导纳 右臂速度跟随 跟随实现在ad_ctrl_rc_v1.py中
        '''
        # ========== 状态定义 ==========
        STATE_APPROACHING = "approaching"      # 接近阶段：正转等待接触
        STATE_TIGHTENING = "tightening"        # 拧紧阶段：正转拧紧
        STATE_COMPLETED = "completed"           # 完成
        
        # ========== 参数配置 ==========
        # 速度参数
        speed_max = 0.03                       # 最大速度
        speed_min = 0.01                       # 最小速度
        speed_approach = 0.005                 # 接近阶段速度系数
        speed_tighten = 0.03                  # 拧紧阶段速度系数
        # 接触检测参数
        contact_force_threshold = 5.0         # 接触检测阈值（力突然增大的阈值）
        contact_check_window = 3              # 接触检测窗口（连续N次检测）
        # 拧紧阶段参数
        tighten_duration = 4                # 拧紧持续时间（秒）
        
        # ========== 初始化 ==========
        self.ad_ctrl_rc_enable.publish(Bool(data=True))
        self.ad_left_switch.publish(Bool(data=True)) # 左臂导纳

        # ========== 数据保存与可视化 ==========
        # 实时记录力数据 速度信息是否需要记录？
        force_data = []
        speed_data = []

        state = STATE_APPROACHING
        init_force = self.left_force.wrench.force
        force_data.append([init_force.x, init_force.y, init_force.z])
        contact_detected_count = 0
        tighten_start_time = None
        screw_direction = np.array([-0.9970157, 0.01299241, 0.07609786, 0.0, 0.0, 0.0])  # 螺丝方向向量
        
        rospy.loginfo(f"[{state}] 开始拧螺丝流程，初始力: x={init_force.x:.4g}")
        
        # ========== 主循环 ==========
        while not rospy.is_shutdown():
            t0 = time.time()
            force = self.left_force.wrench.force
            # 记录力数据
            force_data.append([force.x, force.y, force.z])
            force_x_diff = abs(force.x - init_force.x)
            current_time = time.time()
            rospy.loginfo(f"[{state}] force: x={force.x:.4g}, diff_x={force_x_diff:.4g}, y={force.y:.4g}, z={force.z:.4g}")
            # ========== 状态机逻辑 ==========
            if state == STATE_APPROACHING:
                # 接近阶段的速度应该慢慢提速
                screw_vel = screw_direction * speed_approach  # 缓慢接近螺丝
                speed_approach += 0.005
                if speed_approach > speed_max:
                    speed_approach = speed_max
                rospy.loginfo(f"[{state}] speed_approach: {speed_approach:.4g}")
                self.ad_left_vel_publisher.publish(Float64MultiArray(data=screw_vel.tolist()))
                
                # 检测接触：力突然增大
                if force_x_diff > contact_force_threshold:
                    contact_detected_count += 1
                    # 进入反转的条件  反力大于阈值 或者 产生旋转而产生y轴方向的力
                    if contact_detected_count >= contact_check_window:
                        state = STATE_TIGHTENING
                        rospy.loginfo(f"[{state}] 检测到接触，进入拧紧阶段")
                        # self.ad_right_des_force_pub.publish(Float64MultiArray(data=[des_force_reverse, 0.0, 0.0]))
                        tighten_start_time = time.time()
                        contact_detected_count = 0
                else:
                    contact_detected_count = 0
            
            elif state == STATE_TIGHTENING:
                screw_vel = screw_direction * speed_tighten  # 速度保持匀速
                self.ad_left_vel_publisher.publish(Float64MultiArray(data=screw_vel.tolist()))
                # 检查反转时间
                if current_time - tighten_start_time >= tighten_duration:
                    state = STATE_COMPLETED
                    
            elif state == STATE_COMPLETED:
                # 完成状态：停止速度控制
                self.ad_left_vel_publisher.publish(Float64MultiArray(data=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
                time.sleep(2.0)
                self.ad_left_vel_publisher.publish(Float64MultiArray(data=(-1*screw_direction*0.01).tolist()))
                time.sleep(3.0)
                self.ad_left_vel_publisher.publish(Float64MultiArray(data=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
                self.ad_ctrl_rc_enable.publish(Bool(data=False))
                rospy.loginfo(f"[{state}] 拧螺丝流程完成")
                break
            
            # ========== 安全保护 ==========
            if force_x_diff > 80.0:  # 安全阈值
                rospy.logwarn(f"力过大！紧急停止。当前力: {force_x_diff:.4g}")
                self.ad_left_vel_publisher.publish(Float64MultiArray(data=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
                self.ad_ctrl_rc_enable.publish(Bool(data=False))
                break
        
            # ========== 循环控制 ==========
            elapsed = time.time() - t0
            time.sleep(max(0.1 - elapsed, 0))

        print("拧螺丝流程结束")

        # ========== 数据保存与可视化 ==========
        # 保存力数据到CSV文件
        force_data = np.array(force_data)
        np.savetxt(f"force_data_{time.strftime('%Y%m%d_%H%M%S')}.csv", force_data, delimiter=",", header="x,y,z")
        # 可视化力数据
        plt.figure(figsize=(10, 6))
        plt.title(f"Force Data {time.strftime('%Y%m%d_%H%M%S')}")
        plt.xlabel("Frame")
        plt.ylabel("Force (N)")
        plt.plot(force_data[:, 0], label="x")
        plt.plot(force_data[:, 1], label="y")
        plt.plot(force_data[:, 2], label="z")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


    def R_catch_gun(self):
        '''
        这个函数现在是完整的，从初始位置进行抓取，最终手抓着枪在螺丝口
        '''

        # 中间值
        joint = [0.2654625732657223, -0.33848245764602325, -0.2145416002258571, -1.0331102633852198, -0.10339989536078065, -0.8037776558366528, -0.1962924135959343]

        self.controller.movejh(joint+[-0.028893967050862557, 0.0], 10, 0.8, 1)

        # 抓取位置上方
        joint = [-0.3571418963429096, -0.6845629140934761, 0.2332849284994154, -1.5602577424368749, 1.4072955855726832, 0.6730993500461931, -0.07276673245409221]
        self.controller.movej(joint, ArmGroup.RIGHT, 1, 1)

        # 张开手
        self.controller.grasp_hand(HandType.RIGHT, [-0.8, 1.5, 0.5, 0.5, 0.5, 0.5])
        # time.sleep(0.25)

        # 弯腰
        # self.controller.movej([-0.028893967050862557, 0.5216074114082403], ArmGroup.WAIST, 0.3, 0.5)
        # NOTE 现在桌子高度提升1cm
        self.controller.movej([0.013451432604567071], ArmGroup.LIFT, 0.05, 0.1)
        # input("弯腰")
        self.controller.movej([-0.028893967050862557, 0.5216074114082403], ArmGroup.WAIST, 0.5, 0.8)

        # 下降 
        # input("右手下降")
        self.controller.movel_relative_base([0.0, 0.0, -0.06], ArmGroup.RIGHT)

        # 抓取
        # input("grasp screwdriver enter")
        self.controller.grasp_hand(HandType.RIGHT, [0.3, 1.5, 1.15, 1.2, 1.1, 1.1])
        time.sleep(0.25)

        # 抬起
        # input("右手抬起")

        self.controller.movel_relative_base([0.0, 0.0, 0.06], ArmGroup.RIGHT)

        # 抓完枪从这开始
        # input("ts")
        # self.controller.movej([0.0, 0.0] , ArmGroup.WAIST, 0.6, 0.8)
        self.controller.movej([0.0, 0.0], ArmGroup.WAIST, 0.6, 1)
        self.controller.movej([0.013451432604567071], ArmGroup.LIFT, 0.05, 0.1)
        # self.controller.movejh([0.0, 0.0] + [0.003451432604567071] , 24, 0.05, 0.1)

        # 枪竖在身前
        joint = [0.9192619811710756, -0.22200777254965942, -0.1458959580577357, -1.731342085180495, 0.7199283627733166, -0.6243449706771076, -0.3579211082948256]
        self.controller.movej(joint, ArmGroup.RIGHT, 1, 1)

        self.controller.movej([-1.9, 0.0], ArmGroup.WAIST, 0.8, 1)

        # # 伸手

        # # 最终位置

        joints = [
            [0.9189384070896267, -0.6862526898521537, -0.6706612128164124, -1.7767871061819736, 0.7372335840182132, -0.2822306995052362, -0.3647859928820154],
            [0.3516531211835172, -1.5647563209859072, -0.16944496065207204, -1.3789126258860052, 0.4072838947422497, -0.09888925842669959, -0.5523090386511943],
        ]
        # self.controller.movej_by_path(joints, ArmGroup.RIGHT, total_time=3)
        self.controller.movej_by_path(joints, ArmGroup.RIGHT, total_time=5)

        self.controller.movej([-1.9, 0.3], ArmGroup.WAIST, 0.8, 1)


    def R_place_gun(self):
        # 最终位置
        joint = [0.3516531211835172, -1.5647563209859072, -0.16944496065207204, -1.3789126258860052, 0.4072838947422497, -0.09888925842669959, -0.5523090386511943]

        self.controller.movejh(joint+[-1.9, 0.0], 10, 0.8, 1)

        # # 伸手

        # # 枪竖在身前
        joints = [
            [0.9189384070896267, -0.6862526898521537, -0.6706612128164124, -1.7767871061819736, 0.7372335840182132, -0.2822306995052362, -0.3647859928820154],
            [0.9192619811710756, -0.22200777254965942, -0.1458959580577357, -1.731342085180495, 0.7199283627733166, -0.6243449706771076, -0.3579211082948256],
        ]
        self.controller.movej_by_path(joints, ArmGroup.RIGHT, total_time=3)
        self.controller.movej([0.0, 0.0], ArmGroup.WAIST, 0.8, 1)
        self.controller.movej([0.013451432604567071], ArmGroup.LIFT, 0.05, 0.1)

        # 这是右手抓枪往上后的关节值--只在place有用
        joint = [-0.3580886501367786, -0.6850782357787466, 0.2330092913189219, -1.559735829427723, 1.4071757433202947, 0.6728553939363806, -0.07294341078459364]
        self.controller.movej(joint, ArmGroup.RIGHT, 1, 1)

        # 弯腰
        self.controller.movej([-0.028893967050862557, 0.5216074114082403], ArmGroup.WAIST, 0.5, 0.1)

        # 下降 
        # input("右手下降")
        self.controller.movel_relative_base([0.0, 0.0, -0.06], ArmGroup.RIGHT)

        # 张开手
        self.controller.grasp_hand(HandType.RIGHT, [-0.8, 1.5, 0.5, 0.5, 0.5, 0.5])
        time.sleep(0.5)

        # 抬起
        self.controller.movel_relative_base([0.0, 0.0, 0.06], ArmGroup.RIGHT)

        # 腰部回正
        self.controller.movej([0.0, 0.0], ArmGroup.WAIST, 0.6, 0.8)
        self.controller.movej([0.003451432604567071], ArmGroup.LIFT, 0.05, 0.1)
        
        # 手指恢复初始状态
        self.controller.grasp_hand(HandType.RIGHT, [-0.1, 0.0, 0.35, 0.35, 0.35, 0.35])

        # 中间值

        # 右手初始位置

        t0 = time.time()
        joints = [
            [0.2654625732657223, -0.33848245764602325, -0.2145416002258571, -1.0331102633852198, -0.10339989536078065, -0.8037776558366528, -0.1962924135959343],
            [0.6977575460814478, -0.05221526936566079, -0.4212095644697911, -1.193883318812209, 0.24680313456883596, -0.7542039588905946, 0.25471592693025236],
        ]
        self.controller.movej_by_path(joints, ArmGroup.RIGHT, total_time=4)
        rospy.loginfo(f"movej333 time: {time.time() - t0}")

        pass


    def L_catch_gun(self):
        # 先弯腰过去

        joint = [0.3643803683871738, 0.17916416732077778, 0.9963804705830626, -0.7262908478737473, 1.2330089979241166, 0.2349223177685588, -1.121363243664836]

        self.controller.movejh(joint+[-1.4327980168809518, 0.3364750999185162] , 9, 0.8, 1)
        self.controller.movej([0.013451432604567071], ArmGroup.LIFT, 0.05, 0.1)
        
        # 张开手
        self.controller.grasp_hand(HandType.LEFT, [-0.8, 1.5, 0, 0, 0, 0])
        time.sleep(0.5)

        t0 = time.time()
        # 移动到抓取位置
        joints = [
            [0.36469195824338385, 0.17736653353495058, 0.9140728116426544, -0.7237132207882496, 2.1556145779368308, 0.23643921217898883, -1.0931236627742926],
            [-0.05700895946119999, 0.32594695804618823, 1.1494190268831517, -0.7298526195358591, 2.3682866390254276, -0.28972285742633863, -1.026257084775663],
            # [0.02864229832084675, 0.5252685922187084, 1.1614272205724774, -0.7120263840987037, 2.3309318089559383, -0.28978485502720297, -0.7745550718372854],
            [0.028474519167502876, 0.5077836075952291, 1.1619185738072701, -0.7306148162610498, 2.331339272614059, -0.2897029825973534, -0.8458237761365216]
        ]
        self.controller.movej_by_path(joints, ArmGroup.LEFT, total_time=5)
        # for joint in joints:
        #     self.controller.movej(joint, ArmGroup.LEFT, 0.3, 1)
        
        rospy.loginfo(f"movej444 time: {time.time() - t0}")

        # 抓住枪
        # input("抓住枪")
        self.controller.grasp_hand(HandType.LEFT, [-0.8, 1.5, 1.1, 1.1, 1.1, 1.1])
        time.sleep(0.25)
        self.controller.grasp_hand(HandType.LEFT, [0.3, 1.5, 1.1, 1.1, 1.1, 1.1])
        time.sleep(0.25)

        # 抬起
        self.controller.movel_relative_base([0.0, 0.0, 0.06], ArmGroup.LEFT)

        # 抬升lift
        self.controller.movej([0.11], ArmGroup.LIFT, 0.05, 0.1)

        t0 = time.time()
        joints = [
            [0.5842070119433629, 0.16976853473352094, 1.1748136001642706, -1.5298677246283205, 2.327576225889061, 0.018513254880111497, -0.5441019801651011],
            [0.8703064210703815, 0.4294067745331631, 0.8306506197550334, -1.9370844477881066, 0.4536388779661138, -0.4723540848398542, -0.3691179936666204],
            [1.0633363369925064, 0.7809640219147695, 0.964802037078698, -1.9756187053844771, -0.6819863056671238, -0.2310017556078717, 0.3964766746742606],

            [0.2553718556146123, 1.0361561141507991, 0.33965691171943035, -1.1273390506185934, -0.9458909296517959, -0.13123497898558895, 0.8488184085778885],
        ]
        for joint in joints:
            self.controller.movej(joint, ArmGroup.LEFT, 0.8, 1)
            # input("go to xxx")
            # self.controller.movej(joint, ArmGroup.LEFT, 0.2, 1)
        rospy.loginfo(f"movej555 time: {time.time() - t0}")
        # input("movej")
        # self.controller.movej([-1.2285328897974068, 0.40510875786139877], ArmGroup.WAIST, 0.5, 0.8)
        self.controller.movej([0.1], ArmGroup.LIFT, 0.05, 0.1)
        # joint =  [-0.1446256301824178, 1.093956032477763, 0.02972087859234307, -1.0758651254262077, -1.0749969881499055, -0.13974330428238932, 0.8421885809233537]
        joint = [-0.10571285083187831, 1.1871933048360006, 0.048080711658258224, -1.097294478102549, -0.9612307379575213, -0.15584108517054804, 0.6840629583207645,]
        # self.controller.movej(joint, ArmGroup.LEFT, 0.5, 0.8)
        self.controller.movejh(joint +[-1.2285328897974068, 0.40510875786139877], 9, 0.2, 0.5)



    def L_place_gun(self):
        # 线回到腰的位置
        joint = [0.2553718556146123, 1.0361561141507991, 0.33965691171943035, -1.1273390506185934, -0.9458909296517959, -0.13123497898558895, 0.8488184085778885]
        # self.controller.movej(joint,  ArmGroup.LEFT, 0.8, 1)
        waist = [-1.4327980168809518, 0.3364750999185162]
        # self.controller.movej(waist, ArmGroup.WAIST, 0.1, 0.2)
        self.controller.movejh(joint + waist, 9, 0.2, 0.5)

        # # 抬升lift
        self.controller.movej([0.11], ArmGroup.LIFT, 0.05, 0.1)
        t0 = time.time()
        joints = [
            [0.14295982287421793, 0.33633728132826946, 1.1592820442547236, -1.2087375862323826, 2.276954858480167, -0.4709965450283922, -0.5475825076792707],

            [0.5869154468473425, 0.17498167271241982, 1.2574328489608888, -1.5299857692469232, 2.482963690335964, -0.26189928340398744, -0.49263495361484805],
            
            [0.5842070119433629, 0.16976853473352094, 1.1748136001642706, -1.5298677246283205, 2.327576225889061, 0.018513254880111497, -0.5441019801651011],
            [0.8703064210703815, 0.4294067745331631, 0.8306506197550334, -1.9370844477881066, 0.4536388779661138, -0.4723540848398542, -0.3691179936666204],
            [1.0633363369925064, 0.7809640219147695, 0.964802037078698, -1.9756187053844771, -0.6819863056671238, -0.2310017556078717, 0.3964766746742606],

            [0.2553718556146123, 1.0361561141507991, 0.33965691171943035, -1.1273390506185934, -0.9458909296517959, -0.13123497898558895, 0.8488184085778885],
        ]
        for joint in joints[::-1]:
            self.controller.movej(joint, ArmGroup.LEFT, 0.5, 1)
            # self.controller.movej(joint, ArmGroup.LEFT, 0.2, 1)
        rospy.loginfo(f"movej666 time: {time.time() - t0}")
        # 下降lift
        self.controller.movej([0.0134514478633620137], ArmGroup.LIFT, 0.05, 0.1)
        # time.sleep(1)
        # input("下降")
        self.controller.movel_relative_base([0.0, 0.0, -0.06], ArmGroup.LEFT)

        self.controller.grasp_hand(HandType.LEFT, [-0.8, 1.5, 1.1, 1.1, 1.1, 1.1])
        time.sleep(0.25)
        self.controller.grasp_hand(HandType.LEFT, [-0.8, 1.5, 0, 0, 0, 0])
        time.sleep(0.25)
        
        t0 = time.time()
        # input("goback")
        joints = [
            [0.3643803683871738, 0.17916416732077778, 0.9963804705830626, -0.7262908478737473, 1.2330089979241166, 0.2349223177685588, -1.121363243664836],
            [0.36469195824338385, 0.17736653353495058, 0.9140728116426544, -0.7237132207882496, 2.1556145779368308, 0.23643921217898883, -1.0931236627742926],
            [-0.05700895946119999, 0.32594695804618823, 1.1494190268831517, -0.7298526195358591, 2.3682866390254276, -0.28972285742633863, -1.026257084775663],
            # [0.028474519167502876, 0.5077836075952291, 1.1619185738072701, -0.7306148162610498, 2.331339272614059, -0.2897029825973534, -0.8458237761365216],
        ]
        # self.controller.movej_by_path(joints[::-1], ArmGroup.LEFT, total_time=4)
        self.controller.movej_by_path(joints[::-1], ArmGroup.LEFT, total_time=5)
        rospy.loginfo(f"movej777 time: {time.time() - t0}")
        
        self.controller.grasp_hand(HandType.LEFT, [-0.1, 0.0, 0.35, 0.35, 0.35, 0.35])
        self.controller.movej([0.0034514478633620137], ArmGroup.LIFT, 0.05, 0.1)

        joint = [0.6671378505961911, 0.13622468828998535, 0.4040960908287161, -1.0310438833484115, -0.2242008857683686, -0.9697996963146885, -0.41568548924100707]

        self.controller.movejh(joint+[0.0, 0.0], 9, 0.8, 1)


if __name__ == "__main__":

    rospy.init_node("aaaaa")
    controller = NaviController()

    # 当前是胸部相机
    cam2chest = np.array([
            [-0.01121016, -0.51776829,  0.85544744,  0.11195075],
            [-0.99949497,  0.03124157,  0.00581146,  0.03255672],
            [-0.02973451, -0.85495027, -0.51785703,  0.41386475],
            [ 0.0,         0.0,         0.0,         1.0        ]]
        )

    mtx = np.array([[913.4111328125, 0, 646.2576904296875], 
                    [0, 913.0277099609375,  370.2160339355469], 
                    [0, 0, 1]])

    dist = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    s = 0.03
    # 需要调整和修改，作为aruco码的坐标
    objPoints = np.array([[-s/2, s/2, 0],
                        [s/2, s/2, 0],
                        [s/2, -s/2, 0],
                        [-s/2, -s/2, 0]], dtype=np.float32).reshape(-1, 3)
    cam_name = "zj_humanoid/sensor/realsense_up"
    # 需要知道目标的target_pose
    # target_pose = np.array(
    #     [[ 0.92740577, -0.21211491,  0.30810033,  0.16351271],
    #     [-0.33333523, -0.84237959,  0.42341971,  0.00182009],
    #     [ 0.16972379, -0.49538257, -0.85193306,  0.37715086],
    #     [ 0.        ,  0.        ,  0.        ,  1.        ]]
    #     )

    left_target_pose = np.array(
            [[ 0.99107865,  0.13246734,  0.01468012, -0.24424216],
            [ 0.09513131, -0.62596401, -0.77402784,  0.02079183],
            [-0.09334419,  0.76851901, -0.63298136,  0.4442741 ],
            [ 0.        ,  0.        ,  0.        ,  1.        ]]
        )

# [[ 0.94613099  0.32375922 -0.00401462 -0.13101455]
#  [ 0.23884495 -0.70624766 -0.6664588  -0.02993899]
#  [-0.2186075   0.62959845 -0.74553105  0.43872787]
#  [ 0.          0.          0.          1.        ]]
# [-0.1446256301824178, 1.093956032477763, 0.02972087859234307, -1.0758651254262077, -1.0749969881499055, -0.13974330428238932, 0.8421885809233537, 
# 0.4919284776042332, -0.18829614695277996, -0.27675171344071714, -0.7681162134052102, 0.464820160113959, -0.9576615746388625, 0.04354610627369946, 
# 1.1865569543413874e-07, 1.1865569543413874e-07, -1.2285328897974068, 0.40510875786139877, 0.09999998513338891]
    # right_target_pose = np.array(
    #         [[ 0.94833925, -0.30360308,  0.09207514,  0.23030111],
    #         [-0.17704922, -0.74727836, -0.64049092, -0.06076331],
    #         [ 0.26326077,  0.59110085, -0.76242609,  0.42420521],
    #         [ 0.        ,  0.        ,  0.        ,  1.        ]]
    #     )
    right_target_pose = np.array(
        [[ 0.9468031, -0.29839036,  0.12052836,  0.2308402 ],
        [-0.19582872, -0.83141457, -0.52000089, -0.05899469],
        [ 0.25537229,  0.46873553, -0.8456193,   0.4156174 ],
        [ 0.0,          0.0,          0.0,          1.0        ]]
    )

    right_target_pose = np.array(
        [[ 0.94602072, -0.30055838,  0.12128255,  0.19058691],
        [-0.19679447, -0.83000945, -0.52187763, -0.05288666],
        [ 0.25752035,  0.46983931, -0.84435436,  0.39901528],
        [ 0.0,          0.0,          0.0,          1.0        ]]
    )
#     [[ 0.94207133 -0.33540088  0.00280188  0.18807681]
#  [-0.23091905 -0.65461692 -0.71982851 -0.06564701]
#  [ 0.24326528  0.67748279 -0.69414629  0.40180083]
#  [ 0.          0.          0.          1.        ]]
# [[ 0.94155219 -0.33678528  0.00742612  0.18756406]
#  [-0.23028698 -0.65959068 -0.71547749 -0.06379803]
#  [ 0.24586049  0.67194926 -0.69859632  0.40367402]
#  [ 0.          0.          0.          1.        ]]
# [[ 0.95592246 -0.2753473   0.10196132  0.19399915]
#  [-0.18248088 -0.82917331 -0.52836764 -0.05262699]
#  [ 0.23002821  0.4864725  -0.842871    0.39533087]
#  [ 0.          0.          0.          1.        ]]


    dis_threshold = 0.001
    # operation = PBVSController(controller, cam2chest, mtx, dist, cam_name, target_pose, dis_threshold, objPoints, True)
    # while True:
    # operation.ts_aruco_pose()
    # input("start")

    operation = Operation(controller, cam_name, True)
    operation.ts_aruco_pose()
    # operation.single_arm_pbvs(ArmGroup.LEFT, left_target_pose)
    # operation.single_arm_pbvs(ArmGroup.RIGHT, right_target_pose)

    # input("go")
    # operation.ts_ad_ctrl_with_state_machine()

 

    # operation.ts_ad_ctrl()
    # input("start, dual_arm_control")
    # operation.ts_dual_arm_control()
    # input("start, dual_arm_control_with_ad_ctrl")
    # operation.ts_ad_ctrl_with_dual_arm_control()

    

