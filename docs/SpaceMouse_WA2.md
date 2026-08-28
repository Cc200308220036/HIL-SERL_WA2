# Orin 22 + ROS1 容器内使用 SpaceMouse 遥控 WA2

## 1. 目标和最简方案

本文只实现一个最小闭环：使用一只 SpaceMouse，在 Orin 的 `hilserl` 容器中低速遥控 WA2 左臂 TCP 平移/旋转；暂不接入 HIL-SERL Actor、Learner、Replay Buffer、示范保存或人工干预 Wrapper。

推荐链路：

```text
SpaceMouse USB
    ↓ /dev/input/event*（容器已映射 /dev）
spacenavd（容器内 ARM64 Linux 开源驱动）
    ↓ 本地 spnav socket
ROS1 spacenav_node
    ↓ sensor_msgs/Joy: /spacenav/joy
WA2 专用 teleop 节点
    ↓ 连续笛卡尔控制适配（尚待当前 SDK 确认）
    ↓
WA2 左臂
```

这条路线适合当前阶段，原因是：

- `spacenavd` 和 `spacenav_node` 都能在 ARM64/ROS1 下工作。
- `/spacenav/joy` 可以先用 `rostopic echo` 独立检查，不必一开始就连接机器人动作。
- 连续遥控应使用厂家确认支持的连续笛卡尔接口；不应以 10～50 Hz 重复调用点到点 `movej`/`movel` Service。
- teleop 节点可以实现按键使能、死区、速度缩放、数据超时和退出归零。
- 以后仍可以把相同 6D SpaceMouse 动作映射迁移到 HIL-SERL intervention Wrapper。

HIL-SERL 自带的 `pyspacemouse.py` 使用 `easyhid` 直接读 HID。本阶段不选它作为主路线，因为现有开发记录已经验证了 `spacenavd + ROS1` 思路，而且 ROS topic 更容易分层调试；后续做 HIL-SERL 干预时再决定复用直接 HID 代码还是订阅 `/spacenav/joy`。

当前 `hilserl` 容器已经核对的环境事实：

```text
容器系统：Ubuntu 22.04（arm64）
ROS：Noetic（源码工作空间 + /opt/ros/noetic 叠加）
SpaceMouse USB ID：256f:c63a
设备名称：3Dconnexion SpaceMouse Wireless BT
容器权限：privileged=true，已挂载 /dev 和 /dev/bus/usb
spacenavd 候选版本：0.7.1-1（Jammy arm64）
libspnav-dev 候选版本：0.2.3-1（Jammy arm64）
```

`libspnav-dev` 不是 ROS2 专用依赖。ROS1 分支的 `spacenav_node/package.xml` 明确声明 `libspnav-dev`，其 `CMakeLists.txt` 直接链接 `spnav` 和 `X11`；从源码编译 ROS1 `spacenav_node` 时必须安装该开发包。旧开发记录在 ROS1 部分只写 `spacenavd`，遗漏了源码编译依赖。

2026-08-07 对运行中 WA2 SDK 的只读检查发现：

```text
运行中 upperlimb SDK 版本：1.3.2
/zj_humanoid/upperlimb/enable_speedl：不存在
/zj_humanoid/upperlimb/speedl/left_arm：机器人节点没有订阅
/zj_humanoid/upperlimb/servol/left_arm：存在，类型 upperlimb/DualPose
/zj_humanoid/upperlimb/set_servo_params：存在，类型 upperlimb/Servo
```

因此本文现有 teleop 脚本当前只允许 dry-run。`naviai_controller` 中的 `speedl()` 来自另一接口版本，与当前运行中的 SDK 1.3.2 不匹配；在确认 `servol` 的控制周期、目标语义、参数、停止和 watchdog 之前，禁止传入 `_execute:=true`。

---

## 2. 安全边界

SpaceMouse 是输入设备，不是安全设备。正式发送速度前必须满足：

- 物理急停已测试且有人能够立即触发。
- `hilserl` 是唯一发送机器人动作的容器/程序。
- `assembly`、其他遥操脚本、HIL-SERL Actor 和测试节点没有同时控制 WA2。
- 已确认所选连续笛卡尔接口的消息语义、单位、参考坐标系和允许控制频率。
- 已确认左臂当前姿态远离关节限位、奇异位形、自碰撞和外部障碍物。
- 第一阶段只开放一个平移轴，旋转速度保持为零。
- 必须按住指定 SpaceMouse 按钮才允许运动；松手立即发零速度。
- SpaceMouse 数据超过 watchdog 时间未更新时立即发零速度。
- Ctrl+C、节点异常或正常退出时都按厂家定义发送安全停止并退出连续控制模式。

文中的速度数值只能作为格式示例，不能替代 WA2 厂家限值和现场风险评估。

---

## 3. 路径和运行位置

当前工程的实际映射是：

```text
Orin 宿主机：/home/naviai/hilserl_orin/catkin_ws
                    ↕ bind mount
hilserl 容器：/root/catkin_ws
```

当前开发电脑上的对应副本是：

```text
/media/cyw/XIAKE/ZJhum/hilserl_orin
```

本文明确标记“Orin 宿主机”和“容器终端”。不要把开发电脑的 `/media/cyw/...` 路径直接复制到 Orin 命令中。

---

## 4. Gate A：Orin 宿主机识别 SpaceMouse

### 4.1 插入设备并读取真实 USB ID

在 **Orin 宿主机**执行：

```bash
lsusb
```

再筛选常见厂商：

```bash
lsusb | grep -Ei '3Dconnexion|SpaceMouse|046d|256f'
```

常见旧设备可能使用 Logitech `046d`，较新的 3Dconnexion 设备常见 `256f`。原开发记录中的 `25af` 不应直接复制，必须以这台 Orin 的 `lsusb` 输出为准，同时记录实际 `idVendor:idProduct`。

检查输入设备：

```bash
ls -l /dev/input/by-id 2>/dev/null
ls -l /dev/input/event* 2>/dev/null
```

本机当前识别结果为：

```text
/dev/input/by-id/usb-3Dconnexion_SpaceMouse_Wireless_BT-event-joystick
  -> ../event14
```

`event14` 可能在重插或重启后改变，脚本和配置中应优先使用 `/dev/input/by-id` 稳定名称。设备拔出后对应符号链接和 event 节点消失属于正常现象。

拔出和插入 SpaceMouse 前后比较设备列表，确定对应的 event 设备。也可查看内核日志：

```bash
dmesg --ctime | tail -n 40
```

### 4.2 选择驱动运行位置

最简方案让 `spacenavd` 和 `spacenav_node` 都运行在 `hilserl` 容器内。这样它们天然共享 spnav Unix socket，不必把宿主机 `/run` socket 额外挂载进容器。

先检查宿主机是否已经运行一个 `spacenavd`：

```bash
pgrep -a spacenavd || true
```

如果宿主机服务正在运行，先停止它，避免宿主机与容器两个 daemon 同时读取一个设备：

```bash
sudo systemctl stop spacenavd
```

本阶段不要设置宿主机 `spacenavd` 开机自启动；选择一种架构即可。若以后改为宿主机运行 daemon，需要在 Compose 中显式共享它的 socket，不能宿主机和容器各运行一份。

### 4.3 确认容器设备映射

现有 Compose 已设置：

```yaml
privileged: true
volumes:
  - /dev:/dev
  - /dev/bus/usb:/dev/bus/usb
```

检查运行中容器：

```bash
docker inspect hilserl --format 'privileged={{.HostConfig.Privileged}}'
docker exec hilserl bash -lc 'lsusb; ls -l /dev/input/event* 2>/dev/null'
```

容器必须能看到与宿主机相同的 SpaceMouse USB 和 event 设备。看不到时不要继续机器人控制，应先修复 Compose 设备映射并重建容器。

当前已确认宿主机和 `hilserl` 容器都能看到 `256f:c63a` 以及同一个 `/dev/input/by-id` 设备，且宿主机没有运行 `spacenavd`，因此 Gate A 已通过。内核可能限制普通用户读取 `dmesg`；只要 USB、event、容器映射三项一致，`dmesg: Operation not permitted` 不构成本 Gate 的失败。内核识别成功仍不等于 `spacenavd 0.7.1` 一定支持该具体型号，必须继续通过 Gate C 的 Joy 数据验证。

---

## 5. Gate B：容器内安装 `spacenavd` 和 ROS1 节点

进入容器：

```bash
docker exec -it hilserl bash
```

以下命令均在 **容器终端**执行。

### 5.1 安装系统依赖

```bash
apt-get update
apt-get install -y spacenavd libspnav-dev libx11-dev git
```

依赖用途：

- `spacenavd`：读取 SpaceMouse event 设备并提供 spnav socket。
- `libspnav-dev`：提供 ROS1 `spacenav_node` 编译所需的 `spnav.h` 和链接库。
- `libx11-dev`：ROS1 `spacenav_node` 的 CMake 直接链接 `X11`。
- `git`：获取 `joystick_drivers` ROS1 源码。

当前容器已经安装 `libx11-dev` 和 `ros-noetic-roslint`，命令重复指定 `libx11-dev` 不会造成问题；`spacenavd` 与 `libspnav-dev` 当前尚未安装。

验证架构和程序：

```bash
uname -m
which spacenavd
spacenavd -h || true
dpkg-query -W spacenavd libspnav-dev libx11-dev
source /ros_noetic/catkin_ws/devel/setup.bash
source /opt/ros/noetic/setup.bash --extend
rospack find roslint
```

Orin 预期架构为：

```text
aarch64
```

这些 apt 安装目前只保存在 `hilserl` 容器可写层；删除并重建容器后会丢失，功能验证通过后再加入 `Dockerfile.hilserl`。

### 5.2 下载 ROS1 `spacenav_node`

加载 ROS 环境：

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate hil-actor

if test -f /ros_noetic/catkin_ws/devel/setup.bash; then
  source /ros_noetic/catkin_ws/devel/setup.bash
fi
if test -f /opt/ros/noetic/setup.bash; then
  source /opt/ros/noetic/setup.bash --extend
fi
```

此顺序不能颠倒。当前容器的 `/ros_noetic/catkin_ws/devel/setup.bash` 会重建 ROS 搜索路径；先加载它，再用 `/opt/ros/noetic/setup.bash --extend` 合并 `upperlimb`、`roslint` 等安装包。

检查源码是否已存在：

```bash
test -d /root/catkin_ws/src/joystick_drivers && echo present || echo absent
```

不存在时下载 ROS1 分支：

```bash
cd /root/catkin_ws/src
if test -d joystick_drivers/.git; then
  echo "joystick_drivers already present"
else
  git clone --branch ros1 --single-branch \
    https://github.com/ros-drivers/joystick_drivers.git
fi
```

如果 Orin 不能访问 GitHub，应在联网电脑获取同一 ROS1 源码快照并传到：

```text
/home/naviai/hilserl_orin/catkin_ws/src/joystick_drivers
```

传输后容器会在 `/root/catkin_ws/src/joystick_drivers` 看到它。为了后续复现，应记录 commit：

```bash
git -C /root/catkin_ws/src/joystick_drivers rev-parse HEAD
```

### 5.3 使用 catkin 白名单编译

当前工作空间还包含 HIL-SERL 的 Franka `egg_flip_controller`，不能全量编译。将已有白名单更新为 WA2 控制包和 SpaceMouse 节点：

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate hil-actor
source /ros_noetic/catkin_ws/devel/setup.bash
source /opt/ros/noetic/setup.bash --extend

cd /root/catkin_ws

if test ! -e src/CMakeLists.txt; then
  catkin_init_workspace src
fi

catkin_make \
  -DCATKIN_WHITELIST_PACKAGES='naviai_controller;spacenav_node' \
  -DPYTHON_EXECUTABLE=/usr/bin/python3
```

加载编译结果：

```bash
source /root/catkin_ws/devel/setup.bash
rospack find naviai_controller
rospack find spacenav_node
```

如果报缺少 `Franka`/`franka_hw`，说明没有使用白名单或旧 CMake 缓存没有更新；不要为 WA2 安装 Franka 控制栈。

---

## 6. Gate C：只验证 SpaceMouse 数据，不连接机器人动作

需要三个容器终端。每个新终端都先加载：

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate hil-actor
if test -f /ros_noetic/catkin_ws/devel/setup.bash; then source /ros_noetic/catkin_ws/devel/setup.bash; fi
if test -f /opt/ros/noetic/setup.bash; then source /opt/ros/noetic/setup.bash --extend; fi
source /root/catkin_ws/devel/setup.bash
```

### 6.1 终端 1：启动驱动 daemon

先确认容器里没有旧 daemon：

```bash
pgrep -a spacenavd || true
```

然后以前台调试模式启动：

```bash
cd /
spacenavd -d -v
```

当前 Jammy `spacenavd 0.7.1` 中，`-d` 表示 **do not daemonize**，即留在前台；`-v` 表示详细输出。该 Ubuntu 包中的 daemon 使用相对 socket 路径 `run/spnav.sock`，而 `libspnav` 客户端连接 `/var/run/spnav.sock`。后台 daemon 会自动 `chdir("/")`，但前台 `-d` 模式不会，因此必须先 `cd /`，使 `run/spnav.sock` 正确落到 `/run/spnav.sock`（`/var/run` 指向同一运行目录）。

如果从 `/root/catkin_ws` 直接启动，daemon 虽然可能已经打开 event 设备，却不会在客户端期望的位置创建 socket，随后 `spacenav_node` 会报：

```text
connect failed: No such file or directory
```

保持 daemon 终端运行，并在另一个容器终端检查：

```bash
ls -l /run/spnav.sock /var/run/spnav.sock
```

至少应看到客户端可访问的 socket。继续观察 daemon 是否出现设备名称、实际 event 路径或权限错误。若 apt 安装过程已经自动启动了一个 daemon，先在容器中执行 `pgrep -a spacenavd`，并用 `pkill -TERM spacenavd` 停止旧实例后再从 `/` 启动前台调试实例，保证容器内只有一个 daemon。

当前 Orin 的 Xorg 使用：

```text
DISPLAY=:0
XAUTHORITY=/run/user/1000/gdm/Xauthority
```

如果启动时只出现：

```text
Authorization required, but no authorization protocol specified
```

这不是 SpaceMouse event 设备权限问题，而是容器内 `root` 被宿主机 X Server 拒绝。当前 Ubuntu 包的 `spacenavd` 编译时启用了 X11，并可能在创建 spnav Unix socket 前连接 X Server。先按 `Ctrl+C` 停止失败实例，然后在 **Orin 宿主机终端**执行最小范围授权：

```bash
DISPLAY=:0 \
XAUTHORITY=/run/user/1000/gdm/Xauthority \
xhost +SI:localuser:root
```

再回到容器执行：

```bash
cd /
spacenavd -d -v
```

不要使用范围更大的 `xhost +`。完成全部调试并停止容器内 daemon 后，可以在宿主机撤销授权：

```bash
DISPLAY=:0 \
XAUTHORITY=/run/user/1000/gdm/Xauthority \
xhost -SI:localuser:root
```

若宿主机图形会话发生变化，应通过 `ps -ef | grep Xorg` 核对新的 `-auth` 路径，不能假定该路径永久不变。

### 6.2 终端 2：启动 ROS1 节点

```bash
rosrun spacenav_node spacenav_node
```

检查节点和话题：

```bash
rosnode info /spacenav
rostopic list | grep spacenav
```

节点名可能随所用源码版本略有差异；`rosnode list | grep spacenav` 可以得到实际名称。

### 6.3 终端 3：观察原始数据

```bash
rostopic echo /spacenav/joy
```

缓慢推动和旋转 SpaceMouse，并分别按两个按钮。预期为 `sensor_msgs/Joy`：

```text
axes:    6 个连续值
buttons: 通常为 2 个 0/1 值
```

检查频率：

```bash
rostopic hz /spacenav/joy
```

检查单条消息：

```bash
rostopic echo -n 1 /spacenav/joy
```

这一阶段不启动任何 WA2 teleop 节点，移动 SpaceMouse 不应导致机器人运动。

---

## 7. Gate D：标定轴和按钮

不能假设不同 SpaceMouse 型号、`spacenav_node` 版本和安装方向具有相同轴序与符号。根据 `/spacenav/joy` 实测填写记录：

| 物理操作 | 变化最大的 `axes` 下标 | 正向/负向 | 计划映射 |
|---|---:|---|---|
| 手柄向前/后 | 待测 | 待测 | BASE X |
| 手柄向左/右 | 待测 | 待测 | BASE Y |
| 手柄上提/下压 | 待测 | 待测 | BASE Z |
| 绕 X 旋转 | 待测 | 待测 | TCP `wx` |
| 绕 Y 旋转 | 待测 | 待测 | TCP `wy` |
| 绕 Z 旋转 | 待测 | 待测 | TCP `wz` |
| 左按钮 | `buttons` 待测 | 0/1 | deadman 使能 |
| 右按钮 | `buttons` 待测 | 0/1 | 暂不使用 |

最终 teleop 参数：

```text
axis_map  = [BASE_X来源, BASE_Y来源, BASE_Z来源, WX来源, WY来源, WZ来源]
axis_sign = [六个方向符号，每项为 +1 或 -1]
axis_enable = [六个输出轴开关，每项为 0 或 1]
deadman_button = 实测按钮下标
```

`axis_enable` 用于真正关闭尚未验收的输出轴。例如第一次只开放映射后的 BASE X：

```text
axis_enable = [1, 0, 0, 0, 0, 0]
```

例如，只有在实测轴序恰好为 0～5 且方向全部符合期望时，才可以使用：

```text
axis_map  = [0, 1, 2, 3, 4, 5]
axis_sign = [1, 1, 1, 1, 1, 1]
axis_enable = [1, 1, 1, 1, 1, 1]  # 仅 dry-run 全轴标定使用
```

不要未经观察直接采用这个示例。

---

## 8. Gate E：WA2 最小遥控节点

### 8.1 为什么使用 `speedl`

SpaceMouse 连续产生 6D 输入：

```text
[vx, vy, vz, wx, wy, wz]
```

它与 `naviai_controller.speedl()` 的 6D TCP 速度接口自然对应。遥控循环以固定频率重复发布速度，松手发布全零。

不选择以下方式：

- `movej`：输入是 WA2 左臂 8 个绝对关节目标，不适合直接映射 SpaceMouse 6D 输入。
- 高频 `movel`：它是点到点 Service，不应被当作实时摇杆速度通道反复调用。
- HIL-SERL `SpacemouseIntervention`：它还要求完整 Gym Env 和动作语义，超出当前“只遥控”的范围。

### 8.2 创建自研目录

在容器内执行；目录位于 bind mount 中，会同步到 Orin 宿主机：

```bash
mkdir -p /root/catkin_ws/src/hilserl_wa2/interventions
```

创建文件：

```text
/root/catkin_ws/src/hilserl_wa2/interventions/spacemouse_wa2_teleop.py
```

文件内容如下：

```python
#!/usr/bin/env python3
import math
import threading
import time

import rospy
from sensor_msgs.msg import Joy
from naviai_controller import NaviController, ArmGroup, RobotModel


class SpaceMouseWA2Teleop:
    def __init__(self):
        self.joy_topic = rospy.get_param("~joy_topic", "/spacenav/joy")
        self.execute = bool(rospy.get_param("~execute", False))
        self.deadzone = float(rospy.get_param("~deadzone", 0.15))
        self.watchdog = float(rospy.get_param("~watchdog", 0.25))
        self.publish_rate = float(rospy.get_param("~publish_rate", 50.0))

        # 以下映射必须由 /spacenav/joy 实测后显式传入。
        for name in ("~axis_map", "~axis_sign", "~axis_enable", "~deadman_button"):
            if not rospy.has_param(name):
                raise ValueError("缺少必填参数 {}".format(name))

        self.deadman_button = int(rospy.get_param("~deadman_button"))
        self.axis_map = list(rospy.get_param("~axis_map"))
        self.axis_sign = [
            float(x) for x in rospy.get_param("~axis_sign")
        ]
        self.axis_enable = [
            int(x) for x in rospy.get_param("~axis_enable")
        ]

        # 必须显式传入，避免脚本自带一个被误认为通用安全的速度。
        if not rospy.has_param("~linear_scale"):
            raise ValueError("缺少 ~linear_scale，单位 m/s")
        if not rospy.has_param("~angular_scale"):
            raise ValueError("缺少 ~angular_scale，单位 rad/s")
        if not rospy.has_param("~acc"):
            raise ValueError("缺少 ~acc")

        self.linear_scale = float(rospy.get_param("~linear_scale"))
        self.angular_scale = float(rospy.get_param("~angular_scale"))
        self.acc = float(rospy.get_param("~acc"))

        self._validate_config()

        self._lock = threading.Lock()
        self._axes = None
        self._buttons = None
        self._last_message = None
        self._ctrl = None
        self._speed_mode_enabled = False

        self._subscriber = rospy.Subscriber(
            self.joy_topic,
            Joy,
            self._joy_callback,
            queue_size=1,
            tcp_nodelay=True,
        )
        rospy.on_shutdown(self.stop_robot)

    def _validate_config(self):
        if not (
            len(self.axis_map) == len(self.axis_sign) == len(self.axis_enable) == 6
        ):
            raise ValueError("axis_map、axis_sign 和 axis_enable 必须各有 6 项")
        if any((not isinstance(i, int)) or i < 0 for i in self.axis_map):
            raise ValueError("axis_map 必须是 6 个非负整数")
        if any(sign not in (-1.0, 1.0) for sign in self.axis_sign):
            raise ValueError("axis_sign 每项只能是 +1 或 -1")
        if any(enabled not in (0, 1) for enabled in self.axis_enable):
            raise ValueError("axis_enable 每项只能是 0 或 1")

        values = [
            self.deadzone,
            self.watchdog,
            self.publish_rate,
            self.linear_scale,
            self.angular_scale,
            self.acc,
        ]
        if not all(math.isfinite(x) for x in values):
            raise ValueError("所有数值参数必须是有限数")
        if not 0.0 <= self.deadzone < 1.0:
            raise ValueError("deadzone 必须在 [0, 1) 内")
        if self.watchdog <= 0.0 or self.publish_rate <= 0.0:
            raise ValueError("watchdog 和 publish_rate 必须大于 0")
        if self.linear_scale < 0.0 or self.angular_scale < 0.0:
            raise ValueError("速度缩放不能小于 0")
        if self.linear_scale == 0.0 and self.angular_scale == 0.0:
            raise ValueError("linear_scale 和 angular_scale 不能同时为 0")
        if self.acc <= 0.0:
            raise ValueError("acc 必须大于 0")

    def _joy_callback(self, msg):
        with self._lock:
            self._axes = list(msg.axes)
            self._buttons = list(msg.buttons)
            self._last_message = time.monotonic()

    def _snapshot(self):
        with self._lock:
            axes = None if self._axes is None else self._axes.copy()
            buttons = None if self._buttons is None else self._buttons.copy()
            stamp = self._last_message
        return axes, buttons, stamp

    def _deadband(self, value):
        value = max(-1.0, min(1.0, float(value)))
        if abs(value) < self.deadzone:
            return 0.0
        # 去掉死区后的输出重新线性映射到 [0, 1]。
        return math.copysign(
            (abs(value) - self.deadzone) / (1.0 - self.deadzone), value
        )

    def _map_axes(self, axes):
        if axes is None or max(self.axis_map) >= len(axes):
            return None
        normalized = [
            self._deadband(axes[src]) * sign * enabled
            for src, sign, enabled in zip(
                self.axis_map, self.axis_sign, self.axis_enable
            )
        ]
        return [
            normalized[0] * self.linear_scale,
            normalized[1] * self.linear_scale,
            normalized[2] * self.linear_scale,
            normalized[3] * self.angular_scale,
            normalized[4] * self.angular_scale,
            normalized[5] * self.angular_scale,
        ]

    def _deadman_pressed(self, buttons):
        return (
            buttons is not None
            and 0 <= self.deadman_button < len(buttons)
            and buttons[self.deadman_button] == 1
        )

    def stop_robot(self):
        if self._ctrl is None or not self._speed_mode_enabled:
            return
        try:
            # 连发数帧零速度，降低单帧丢失的风险。
            for _ in range(5):
                self._ctrl.stop_speedl(ArmGroup.LEFT)
                rospy.sleep(0.02)
            self._ctrl.enable_speedl(False)
        except Exception as exc:
            rospy.logerr("停止 speedl 时发生异常：%s", exc)
        finally:
            self._speed_mode_enabled = False

    def run(self):
        rospy.loginfo("等待 SpaceMouse 话题：%s", self.joy_topic)
        try:
            rospy.wait_for_message(self.joy_topic, Joy, timeout=5.0)
        except rospy.ROSException as exc:
            raise RuntimeError("5 秒内未收到 SpaceMouse Joy：{}".format(exc))

        if self.execute:
            self._ctrl = NaviController(model=RobotModel.WA2)
            if not self._ctrl.enable_speedl(True):
                raise RuntimeError("WA2 enable_speedl 失败")
            self._speed_mode_enabled = True
            rospy.logwarn("实机模式：只有按住 deadman 按钮才发送非零速度")
        else:
            rospy.logwarn("DRY-RUN：只打印映射速度，不连接机器人速度控制")

        rate = rospy.Rate(self.publish_rate)
        while not rospy.is_shutdown():
            axes, buttons, stamp = self._snapshot()
            fresh = stamp is not None and time.monotonic() - stamp <= self.watchdog
            pressed = self._deadman_pressed(buttons)
            command = self._map_axes(axes)

            active = fresh and pressed and command is not None
            if not active:
                command = [0.0] * 6

            if self.execute:
                self._ctrl.speedl(command, ArmGroup.LEFT, acc=self.acc)
            else:
                rospy.loginfo_throttle(
                    0.2,
                    "fresh=%s deadman=%s command=%s",
                    fresh,
                    pressed,
                    [round(x, 4) for x in command],
                )
            rate.sleep()


def main():
    rospy.init_node("spacemouse_wa2_teleop", anonymous=False)
    teleop = SpaceMouseWA2Teleop()
    teleop.run()


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, TypeError, ValueError) as exc:
        rospy.logfatal("SpaceMouse teleop 启动失败：%s", exc)
        raise SystemExit(1)
```

赋予权限并做语法检查：

```bash
chmod +x /root/catkin_ws/src/hilserl_wa2/interventions/spacemouse_wa2_teleop.py
python -m py_compile \
  /root/catkin_ws/src/hilserl_wa2/interventions/spacemouse_wa2_teleop.py
```

该脚本不需要重新 `catkin_make`，因为它以普通 Python 文件运行；但必须已经 source `naviai_controller` 所在的 `/root/catkin_ws/devel/setup.bash`。

---

## 9. Gate F：先运行 teleop dry-run

保持 `spacenavd` 和 `spacenav_node` 两个终端运行，在第三个容器终端执行。

以下仅为命令格式；先把 `AXIS_*` 和 `DEADMAN_INDEX` 替换为 Gate D 实测结果：

```bash
python /root/catkin_ws/src/hilserl_wa2/interventions/spacemouse_wa2_teleop.py \
  _axis_map:="[AXIS_X,AXIS_Y,AXIS_Z,AXIS_RX,AXIS_RY,AXIS_RZ]" \
  _axis_sign:="[SIGN_X,SIGN_Y,SIGN_Z,SIGN_RX,SIGN_RY,SIGN_RZ]" \
  _axis_enable:="[1,1,1,1,1,1]" \
  _deadman_button:=DEADMAN_INDEX \
  _linear_scale:=LINEAR_MAX_M_S \
  _angular_scale:=0.0 \
  _acc:=ACC_LIMIT
```

注意没有 `_execute:=true`，所以是 dry-run：

- 不创建 `NaviController`。
- 不调用 `enable_speedl`。
- 不发布任何机器人速度。
- 只打印映射后的 6D command。

检查：

1. 不按按钮时 command 始终全零。
2. 按住 deadman 且只推动一个轴时，只出现一个预期方向的非零平移分量。
3. 松开按钮后下一帧变为全零。
4. 停止 `spacenav_node` 后，超过 watchdog 时间 command 变为全零。
5. SpaceMouse 静止时死区能消除漂移。
6. `_angular_scale:=0.0` 时旋转手柄不会产生机器人角速度。

只有 dry-run 六项全部通过后才允许连接 WA2。

---

## 10. Gate G：检查 WA2 `speedl` 接口

在实机动作前只读检查：

```bash
rosservice list | grep '/zj_humanoid/upperlimb/enable_speedl'
rostopic list | grep '/zj_humanoid/upperlimb/speedl/left_arm'
rostopic info /zj_humanoid/upperlimb/speedl/left_arm
```

检查 Python 导入：

```bash
python -c "
import rospy
from upperlimb.msg import SpeedL
from naviai_controller import NaviController, ArmGroup, RobotModel
print('WA2 speedl imports: PASS')
"
```

确认当前左臂状态：

```bash
rostopic echo -n 1 /zj_humanoid/upperlimb/joint_states
rostopic echo -n 1 /zj_humanoid/upperlimb/tcp_pose/left_arm
```

还必须查阅当前真机 SDK 文档，确认：

- `tcp_speed` 六维顺序。
- 线速度和角速度单位。
- 速度参考坐标系。
- 推荐发布频率。
- 指令 watchdog/超时行为。
- `enable_speedl(False)` 的停止语义。

若这些信息没有确认，不要进入实机 Gate。

### 10.1 当前实测结论（2026-08-07）

```text
upperlimb SDK version: 1.3.2
Unknown service: /zj_humanoid/upperlimb/enable_speedl
Unknown topic: /zj_humanoid/upperlimb/speedl/left_arm
```

机器人节点实际公开的连续控制候选为：

```text
/zj_humanoid/upperlimb/servol/left_arm  upperlimb/DualPose
/zj_humanoid/upperlimb/servoj/left_arm  upperlimb/Joints
/zj_humanoid/upperlimb/speedj/left_arm  upperlimb/SpeedJ
/zj_humanoid/upperlimb/set_servo_params upperlimb/Servo
```

这说明 Gate G 当前未通过。Python 能导入 `upperlimb/SpeedL` 只证明类型包中存在消息定义，不证明机器人节点支持该控制入口。临时创建一个 `/speedl/left_arm` Publisher 也只会让 Topic 暂时出现在 ROS Master 中；机器人节点没有对应 Subscriber 时不会执行动作。

下一步必须从当前 SDK 1.3.2 的厂家文档或已验收示例确认：

1. `servol/left_arm` 是否为当前版本推荐的笛卡尔连续控制入口。
2. `DualPose` 在单左臂 Topic 中未使用的右臂字段应如何填写。
3. `set_servo_params` 的 `v`、`acc`、`time`、`lookahead_time`、`gain` 和 `arm_type` 合法范围。
4. ServoL 接收的是绝对目标位姿、单周期增量还是其他语义。
5. 推荐发布频率、状态到位判断和控制器端命令超时。
6. 松开 deadman、客户端崩溃和网络中断时的厂家安全停止方法。

在这些问题得到确认前：

- 继续保留 SpaceMouse Joy 和映射层 dry-run。
- 不绕过缺失的 `enable_speedl` 直接向 `/speedl/left_arm` 发布。
- 不自行猜测 `servol` 语义后试动。
- 不使用高频 `movel` 代替实时遥控。

---

## 11. Gate H：第一次实机遥控

> **当前阻塞：禁止执行本节命令。** Gate G 已确认当前 SDK 1.3.2 不提供文中假设的 `enable_speedl`/`speedl`运行接口。只有完成 ServoL 或其他厂家支持接口的独立低速验收，并相应重写 teleop 执行层后，本节才能恢复。

### 11.1 只开放单个平移方向

第一次实机测试仍传入完整 `axis_map`，但必须通过 `axis_enable` 在代码中强制关闭其他5个输出轴。不能只依赖操作者“尽量不碰其他方向”。例如只开放映射后的 BASE X：

```text
axis_enable = [1, 0, 0, 0, 0, 0]
angular_scale = 0.0
linear_scale  = 厂家允许范围内的低速值
```

一个用于说明参数格式的保守起点可以是：

```text
linear_scale = 0.01 m/s
angular_scale = 0.0 rad/s
acc = 0.05
```

这些数值不构成 WA2 安全保证；若厂家给出的最小调试速度或加速度更低，应使用厂家值。

### 11.2 启动实机模式

将下列轴和符号替换为实测值：

```bash
python /root/catkin_ws/src/hilserl_wa2/interventions/spacemouse_wa2_teleop.py \
  _axis_map:="[AXIS_X,AXIS_Y,AXIS_Z,AXIS_RX,AXIS_RY,AXIS_RZ]" \
  _axis_sign:="[SIGN_X,SIGN_Y,SIGN_Z,SIGN_RX,SIGN_RY,SIGN_RZ]" \
  _axis_enable:="[1,0,0,0,0,0]" \
  _deadman_button:=DEADMAN_INDEX \
  _deadzone:=0.15 \
  _watchdog:=0.25 \
  _publish_rate:=50.0 \
  _linear_scale:=0.01 \
  _angular_scale:=0.0 \
  _acc:=0.05 \
  _execute:=true
```

运行过程：

1. 不按 SpaceMouse 按钮，确认机器人完全静止。
2. 按住 deadman，但不推动手柄，确认机器人仍静止。
3. 轻推已选定的单个平移方向，观察左臂是否按预期低速移动。
4. 立即松开 deadman，确认运动停止。
5. Ctrl+C 退出，确认脚本连续发送零速度并关闭 speed 模式。
6. 用关节状态和 TCP 状态验证实际运动方向与幅度。

任何方向不一致、松手不停、明显漂移、振动、服务失败或状态反馈中断，都应立即急停并返回 dry-run/接口检查，不要靠继续试动猜测。

### 11.3 逐步开放其他轴

建议顺序：

```text
一个平移轴
  → 通过 axis_enable 逐个增加到三个平移轴
  → 单个旋转轴（极低 angular_scale）
  → 三个旋转轴
  → 六维组合遥控
```

每增加一个轴都重新验证方向、松手归零、工作空间和奇异位形风险。

---

## 12. 停止和故障恢复

### 正常停止顺序

1. 松开 deadman 按钮。
2. Ctrl+C 结束 teleop 节点。
3. 确认左臂停止且 `enable_speedl(False)` 成功。
4. Ctrl+C 结束 `spacenav_node`。
5. Ctrl+C 结束容器内 `spacenavd`。

### teleop 进程被强制终止

`kill -9` 无法执行 Python shutdown 回调。若进程异常消失且机器人没有立即停止：

1. 立即使用物理急停。
2. 不要先尝试重启 Python 脚本抢救。
3. 按机器人厂商流程清除速度模式和错误状态。
4. 查清 SDK 自身为何没有在命令超时后停止，再恢复测试。

可靠遥控不能只依赖客户端退出回调，最终必须确认控制器端也有速度指令 watchdog。

---

## 13. 常见问题

### 宿主机 `lsusb` 看得到，容器看不到

检查运行中的容器是否确实由当前 Compose 创建，并含 `/dev`、`/dev/bus/usb` 映射和 `privileged: true`。修改 Compose 后需要重建容器；注意当前 `hil-actor` 位于容器可写层，删除重建前要先固化环境。

### `spacenavd` 报 permission denied

初始 privileged/root 方案通常不需要额外 udev 放权。仍失败时，在宿主机用 `udevadm info` 确认真实 event 节点和 USB ID，并检查是否有宿主机 daemon 占用。不要直接复制错误 vendor ID。

后续收缩容器权限时，再为实测 ID 配置最小 udev rule 和 `input` 组权限。

### `Authorization required, but no authorization protocol specified`

这是 X11 授权错误，不是 `/dev/input/event*` 的 udev 权限错误。当前容器已挂载 `/tmp/.X11-unix` 并设置 `DISPLAY=:0`，但宿主机 Xorg 仍会校验访问者。按 Gate C 中的命令，在宿主机使用：

```bash
DISPLAY=:0 \
XAUTHORITY=/run/user/1000/gdm/Xauthority \
xhost +SI:localuser:root
```

只授权本机 `root`，不要使用 `xhost +`。授权后重新启动 `spacenavd -d -v`，并确认它成功识别设备且 ROS 节点能够连接 spnav socket。

### `spacenav_node` 连接不到 daemon

确认 daemon 和 ROS 节点运行在同一个容器：

```bash
pgrep -a spacenavd
ls -l /run/spnav.sock /var/run/spnav.sock
```

如果 daemon 在宿主机、节点在容器，它们默认不共享 Unix socket；应统一运行位置或显式挂载 socket。

如果错误为：

```text
connect failed: No such file or directory
```

且 daemon 已打开 SpaceMouse event 设备但上述 socket 不存在，应停止旧实例，并从容器根目录重新以前台模式启动：

```bash
pkill -TERM spacenavd 2>/dev/null || true
cd /
spacenavd -d -v
```

当前 Ubuntu 包的前台模式不会自动切换工作目录；从 `/root/catkin_ws` 启动会使相对 socket 路径无法落到客户端预期的 `/run` 目录。

### `/spacenav/joy` 不出现

依次检查：

```bash
rosnode list | grep spacenav
rosnode info /实际节点名
rostopic list | grep spacenav
```

同时查看 `spacenavd` 和 `spacenav_node` 两个终端日志。

### 有话题但 axes 始终为零

检查 daemon 是否打开了正确 event 设备、是否有另一个 daemon 抢占，以及该 SpaceMouse 型号是否被当前 `spacenavd` 支持。先解决输入层，不要启动机器人控制。

### 静止时有小幅非零值

提高 `_deadzone`，同时检查设备是否机械回中。不要用非常大的 deadzone 掩盖持续漂移或硬件故障。

### 按钮下标不对

用 `rostopic echo /spacenav/joy` 实测。修改 `_deadman_button`，不要假设按钮 0 一定是左键。

### `enable_speedl` 失败

先用 `rosservice list` 判断服务是“存在但调用失败”还是“当前 SDK 根本未提供”。当前实测 SDK 1.3.2 属于后者，应停止实机 Gate 并确认厂家支持的 ServoL 接口；不要绕过 enable 接口直接发布非零速度。

### 松手后仍运动

立即急停。检查 Joy 消息是否正确报告按钮松开、watchdog 是否生效、发布频率是否正常，以及机器人 SDK 是否具备速度命令超时。该问题未解决前禁止继续遥控。

### `No module named rospy/rospkg/upperlimb/naviai_controller`

按照 `docs/hil-serl-controller.md` 完成 ROS、`hil-actor`、NaviAI 类型和 catkin 工作空间加载。`rospkg`/`catkin-pkg` 应安装到 `hil-actor`，不要从 PyPI 安装一个替代版 `rospy`。

---

## 14. 最小验收清单

只有全部通过才算“SpaceMouse 可以遥控 WA2”：

- [ ] Orin 和容器都能识别同一 SpaceMouse。
- [ ] 只有容器内一份 `spacenavd` 在读设备。
- [ ] `/spacenav/joy` 有 6 轴和按钮数据。
- [ ] 六轴顺序、符号、按钮下标已经逐项记录。
- [ ] teleop dry-run 中不按 deadman 时 command 恒为零。
- [ ] 松开 deadman 或 Joy 超时后 command 在 0.25 秒内归零。
- [x] 已核对当前 SDK 1.3.2 不提供可用的 `enable_speedl`/`speedl`运行接口。
- [ ] 厂家已确认 ServoL 或其他连续笛卡尔控制接口及其安全停止语义。
- [ ] teleop 执行层已针对厂家确认的接口完成独立验收。
- [ ] 实机首次通过 `axis_enable` 强制只开放一个低速平移轴。
- [ ] Ctrl+C 能发零速度并退出 speed 模式。
- [ ] 物理急停和控制器端 watchdog 已独立验证。
- [ ] `assembly`、Actor 和其他控制脚本没有同时发送动作。

---

## 15. 后续接入 HIL-SERL 时保留什么

当前最简实现完成后，可复用：

- SpaceMouse 的真实轴序和方向标定。
- deadzone、最大线速度和角速度。
- deadman 与按钮映射。
- Joy 数据新鲜度 watchdog。
- WA2 坐标系和 `speedl` 安全边界。
- 停止和控制权仲裁逻辑。

后续需要改变的是上层组织：把 `/spacenav/joy` 映射成 WA2 Env action，在人工输入有效时写入 `info["intervene_action"]`，再交给 HIL-SERL Actor 收集干预 transition。不要直接复用 Franka Wrapper 中的方向、动作尺度和安全 box。
