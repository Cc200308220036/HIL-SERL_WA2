# naviai_controller

基于 ROS Noetic 的 ZJ Humanoid 双臂机器人高层 Python 控制接口封装。

## 概述

`naviai_controller` 将 [`zj_humanoid`](https://zj-humanoid.github.io/zj_humanoid_sdk_ros/) ROS SDK 封装为统一的 Python 接口。本平台中所有与物理机器人的交互均通过此接口完成。

当前提供两种使用形式：

- 包式引用：通过 `src/naviai_controller/` 安装为 ROS/Python 包后使用。
- 单文件引用：直接使用 `naviai_controller_r3.py`，适合拷贝到脚本目录或在没有安装 Python 包时直接引用。

### 硬件组件

| 组件 | WA1 自由度 | WA2 自由度 | 控制方式 |
|------|-----------|-----------|----------|
| 左臂 | 7 | 8 | MoveJ / MoveL / SpeedJ / SpeedL / ServoJ |
| 右臂 | 7 | 8 | MoveJ / MoveL / SpeedJ / SpeedL / ServoJ |
| 左手 | 6 | 6 | 手部控制器（抓取/释放） |
| 右手 | 6 | 6 | 手部控制器（抓取/释放） |
| 颈部 | 2 | 2 | MoveJ / SpeedJ |
| 腰部 | 2 | 4 | MoveJ / SpeedJ |
| 升降柱 | 1 | 不支持 | MoveJ / SpeedJ |

所有通信均使用 `/zj_humanoid/upperlimb` 和 `/zj_humanoid/hand` 命名空间。

## 架构

```
NaviController（统一高层 API）
├── ArmController  （手臂运动 + 状态反馈）
│   ├── MoveJ / MoveL / MoveJByPath / MoveLByPath
│   ├── SpeedJ / SpeedL / ServoJ
│   └── TCP 位姿、关节状态、速度订阅
└── HandController （手部抓取 + 感知）
    ├── 抓取 / 释放
    └── 关节状态、压力传感器订阅
```

`naviai_controller_r3.py` 是当前包式实现的单文件版本，已经内联了 `NaviController`、`ArmController`、`HandController`、枚举和常用工具函数。它与 `src/naviai_controller/naviai_controller.py` 暴露的高层 API 保持一致，只是使用方式从“安装包导入”变为“文件导入”。

## 安装

```bash
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

### 依赖

- ROS Noetic（`roscpp`, `rospy`, `std_msgs`, `std_srvs`, `sensor_msgs`, `geometry_msgs`）
- `zj_humanoid` ROS 包和对应版本的消息/服务类型
- `numpy`, `scipy`

### zj_humanoid 消息类型

仓库中的 `third_party/zj_humanoid/zj_humanoid_types_25_R3.run` 是当前 R3 SDK 使用的消息/服务类型安装包，用于提供代码中依赖的 ROS 类型，例如 `MoveJ`、`MoveL`、`SpeedJ`、`SpeedL`、`TcpSpeed`、`UplimbState`、`HandJoint`、`PressureSensor` 等。

该文件和 SDK 版本强相关。更换机器人 SDK 或接入不同版本控制器时，需要安装与目标 SDK 匹配的消息类型包；可以使用仓库内已有的 `.run` 文件，也可以从 ZJ Humanoid 官方 SDK 文档/网站下载对应版本后安装。

建议把这类安装包视为外部依赖文件管理，而不是业务代码。如果需要保留多个 SDK 版本，可以统一放在 `third_party/zj_humanoid/` 下，例如：

```text
third_party/
└── zj_humanoid/
    ├── zj_humanoid_types_25_R3.run
    ├── zj_humanoid_types_dev-v1.2.0+6330ca4+62.run
    └── zj_humanoid_types_dev-v1.3.0+78a9d8e+71.run
```

## 快速开始

### 包式引用

```python
import rospy
from naviai_controller import NaviController, ArmGroup, HandType

rospy.init_node("my_control_node")
ctrl = NaviController()              # 默认 WA1
# ctrl = NaviController(model="wa2")  # WA2 机型

# === 关节空间运动 ===
ctrl.movej([0.0, 0.4, 0.5, 0.0, 0.0, 0.0, 0.0], ArmGroup.LEFT, v=0.3, acc=0.5)

# === 笛卡尔空间运动（TCP 位姿格式: [x, y, z, qx, qy, qz, qw]）===
ctrl.movel([0.4, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0], ArmGroup.RIGHT, v=0.1, acc=0.1)

# === 双臂同步运动 ===
left_joints = [...]   # 7 个关节角
right_joints = [...]  # 7 个关节角
ctrl.movejh(left_joints + right_joints, mask=3)  # 3 = LEFT | RIGHT (8421码)

# === 速度控制（非阻塞）===
ctrl.enable_speedl()
ctrl.speedl([0.0, 0.0, 0.02, 0.0, 0.0, 0.0], ArmGroup.RIGHT, acc=0.1)
ctrl.stop_speedl(ArmGroup.RIGHT)

# === 手部控制 ===
ctrl.grasp_hand(HandType.RIGHT, [0.1, 1.5, 1.2, 1.2, 1.2, 1.2])
ctrl.release_hand(HandType.RIGHT)

# === 状态读取 ===
joints = ctrl.get_joints(ArmGroup.LEFT)          # 7 个关节角度 (rad)
tcp    = ctrl.get_tcp_rt(ArmGroup.LEFT)           # [x,y,z,qx,qy,qz,qw]
force  = ctrl.get_hand_force(HandType.LEFT)       # 手指压力传感器读数
```

### 单文件引用

如果不想安装为 Python 包，或者希望像普通脚本一样引用，可以直接把 `naviai_controller_r3.py` 放在调用脚本同级目录，或把仓库根目录加入 `PYTHONPATH`：

```python
import rospy
from naviai_controller_r3 import NaviController, ArmGroup, HandType

rospy.init_node("my_control_node")
ctrl = NaviController(model="wa2")

ctrl.movej([0.0, 0.4, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0], ArmGroup.LEFT)
print(ctrl.get_tcp_rt(ArmGroup.LEFT))
```

两种引用方式的控制接口一致，后续示例中的 `NaviController`、`ArmGroup`、`HandType` 均可按实际使用方式从对应位置导入。

### 机型选择

`NaviController` 默认按 WA1 机型工作。WA2 机型需要在构造时显式指定：

```python
ctrl_wa1 = NaviController()
ctrl_wa2 = NaviController(model="wa2")
```

WA1 的 `upperlimb/joint_states.position` 顺序为 `[left7, right7, neck2, waist2, lift1]`，WA2 的顺序为 `[left8, right8, neck2, waist4]`。WA2 没有升降柱，调用 `ArmGroup.LIFT` 的运动接口会报不支持，`get_joints(ArmGroup.LIFT)` 返回 `None`。

笛卡尔末端控制接口不随单臂关节数变化：单臂 TCP 位姿仍为 `[x,y,z,qx,qy,qz,qw]`，单臂 TCP 速度仍为 `[vx,vy,vz,wx,wy,wz]`。

## API 参考

### NaviController

#### 手臂控制

| 方法 | 说明 |
|------|------|
| `get_joints(arm)` | 获取指定手臂组的关节角度 |
| `get_tcp_rt(arm)` | 获取 TCP 位姿，格式 `[x,y,z,qx,qy,qz,qw]` |
| `get_tcp_matrix(arm)` | 获取 TCP 位姿 4×4 齐次变换矩阵 |
| `get_tcp_speed(arm)` | 获取当前 TCP 线速度/角速度 |
| `movej(joints, arm, v, acc, t, is_async)` | 关节空间点到点运动 |
| `movejh(joints, mask, v, acc, is_async)` | 多组关节同步运动（8421 掩码） |
| `movej_by_path(path, arm, total_time, timestamps, is_async)` | 关节空间轨迹跟随 |
| `movejh_by_path(path, arm_mask, total_time, timestamps, is_async)` | 多组关节轨迹跟随 |
| `movel(pose, arm, v, acc, is_async)` | 笛卡尔空间直线运动 |
| `movel_relative_base(delta_xyz, arm, v, acc, is_async)` | 基坐标系下相对偏移 |
| `movel_relative_eef(transform, arm, v, acc, is_async)` | 末端坐标系下相对偏移 |
| `speedj(joint_speed, arm, acc)` | 关节速度控制 |
| `speedl(tcp_speed, arm, acc)` | 笛卡尔速度控制，格式 `[vx,vy,vz,wx,wy,wz]` |
| `enable_speedj(enable)` / `stop_speedj(arm)` | 开启/停止关节速度模式 |
| `enable_speedl(enable)` / `stop_speedl(arm)` | 开启/停止笛卡尔速度模式 |
| `servoj_dual_arm(joints)` | 高频关节流控（伺服模式） |
| `set_servo_params(time_sec, gain)` / `clear_servo_params()` | 配置/清除伺服参数 |

#### 手部控制

| 方法 | 说明 |
|------|------|
| `get_hand_joints(hand)` | 获取 6 个手指关节位置 |
| `get_hand_pressures(hand)` | 获取手指压力传感器读数 |
| `get_hand_force(hand)` | 获取各手指力数组 |
| `grasp_hand(hand, joints)` | 手指运动至目标关节位置 |
| `release_hand(hand)` | 释放（张开）手爪 |

### 枚举类型

| 枚举 | 取值 | 用途 |
|------|------|------|
| `ArmGroup` | `LEFT=1`, `RIGHT=2`, `DUAL=3`, `NECK=4`, `WAIST=8`, `LIFT=16` | 指定运动目标的身体部件；WA2 不支持 `LIFT` |
| `HandType` | `LEFT=1`, `RIGHT=2` | 指定左手/右手 |
| `RobotModel` | `WA1`, `WA2` | 可选机型枚举，也可直接传入字符串 `"wa1"` / `"wa2"` |
| `CmdState` | `STOPPED`, `MOVEJ`, `MOVEL`, `SPEEDJ`, `SPEEDL`, `SERVOJ`, ... | 当前运动状态（只读） |

## ROS API（zj_humanoid SDK）

本包通过以下 SDK 接口与机器人控制器通信：

### 订阅话题
- `/zj_humanoid/upperlimb/joint_states` — 全部关节位置
- `/zj_humanoid/upperlimb/tcp_pose/{left_arm,right_arm}` — 末端位姿
- `/zj_humanoid/upperlimb/tcp_speed` — TCP 速度
- `/zj_humanoid/upperlimb/uplimb_state` — 运动状态
- `/zj_humanoid/hand/joint_states` — 手指关节位置
- `/zj_humanoid/hand/finger_pressures/{left,right}` — 压力传感器

### 服务客户端
- `MoveJ`, `MoveL`, `MoveJByPath`, `MoveLByPath`, `MoveJByPose`
- `IK`, `FK`, `ArmType`, `Servo`
- `HandJoint`（手部抓取）

### 发布话题
- `speedl/{left_arm,right_arm,dual_arm}` — 笛卡尔速度指令
- `speedj/{left_arm,right_arm,dual_arm,neck,waist,lift}` — 关节速度指令

## 安全注意事项

- 发送运动指令前务必确认当前手臂状态，双臂工作空间内存在碰撞风险。
- 力传感器读数（`get_hand_force`）在插入任务中用作安全检查，未经物理测试不要降低力阈值。
- 速度控制指令（`speedj`/`speedl`）为非阻塞调用，务必调用对应的 `stop` 或 `disable` 方法停止。
