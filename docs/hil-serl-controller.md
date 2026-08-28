# 在 `hilserl` 容器的 `hil-actor` 环境中使用 WA2 控制器

## 1. 目标、当前状态与结论

本文说明如何从当前“源码已挂载，但独立 catkin 工作空间尚未编译”的状态开始，在 Orin 的 `hilserl` 容器和 `hil-actor` Conda 环境中：

1. 编译并加载 `/root/catkin_ws/src/naviai_controller`。
2. 验证 NaviAI ROS SDK、WA2 状态反馈和 Service。
3. 先只读熟悉 `naviai_controller`。
4. 分别完成一次受控的左臂 `movej` 和 `movel` 测试。
5. 为后续 `hilserl_wa2` Gymnasium Env 接入建立边界。

根据 `hil-serl部署.md` 和 `调试日志/0805调试日志.md`，当前 Orin 侧已经完成：

- `hilserl` 容器正在运行，基础镜像为 `ros1_docker:latest`。
- 容器使用 host 网络，并具有 GPU、USB 和设备访问能力。
- 容器内已有 Conda 环境 `/opt/conda/envs/hil-actor`，Python 为 3.10.20。
- JAX 0.4.35、Agentlace 0.1.3 和 HIL-SERL Actor 核心导入已经通过。
- 宿主机独立工作空间映射到容器 `/root/catkin_ws`。
- `naviai_controller` 已复制到独立工作空间，但尚未执行 `catkin_make`。

当前还不能直接启动 HIL-SERL Actor 控制机器人，因为 WA2 Gymnasium Env、动作限幅、控制权仲裁和异常停止尚未实现。本阶段只完成独立 ROS 控制器的分层验收。

---

## 2. 三层路径不要混淆

开发过程中同时存在三种路径：

| 所在位置 | 路径 | 用途 |
|---|---|---|
| 当前开发电脑 | `/media/cyw/XIAKE/ZJhum/hilserl_orin` | 当前看到和编辑的工程副本 |
| Orin 宿主机 | `/home/naviai/hilserl_orin` | 现有 Compose 配置实际使用的工程根目录 |
| `hilserl` 容器 | `/root/catkin_ws` | 容器内编译和运行 ROS 代码的位置 |

现有 `docker/docker-compose.hilserl.yml` 声明的是：

```text
Orin /home/naviai/hilserl_orin/catkin_ws
  ↕ bind mount（双向）
容器 /root/catkin_ws
```

如果 Orin 上实际采用了不同路径，不要凭印象修改命令，先在 **Orin 宿主机**检查真实挂载：

```bash
docker inspect hilserl \
  --format '{{range .Mounts}}{{println .Source "->" .Destination}}{{end}}'
```

必须看到某个实际宿主机目录映射到 `/root/catkin_ws`。容器中产生的 `build/`、`devel/` 和源码修改都会通过该挂载写回 Orin 宿主机；而 `hilserl_orin/docs` 没有挂载进容器。

---

## 3. 为什么不能直接全量 `catkin_make`

当前独立工作空间具有以下状态：

```text
catkin_ws/
└── src/
    ├── hil-serl-main/
    ├── hilserl_wa2/
    └── naviai_controller/
```

目前没有：

```text
catkin_ws/src/CMakeLists.txt
catkin_ws/build/
catkin_ws/devel/
```

此外，`hil-serl-main/serl_robot_infra/egg_flip_controller` 自身也是一个 catkin package，并且要求 `Franka`、`franka_hw` 和 `franka_gripper`。它属于上游 Franka 代码，不是 WA2 控制所需依赖。

因此首次构建采用以下策略：

```text
初始化 catkin 工作空间
  → 使用 CATKIN_WHITELIST_PACKAGES
  → 只配置和编译 naviai_controller
  → 不编译 Franka egg_flip_controller
```

不要为了让全量构建通过而安装 Franka 控制栈，也不要删除 HIL-SERL 上游源码。

---

## 4. Gate 0：确认只有 `hilserl` 拥有机器人控制权

所有以下命令先在 **Orin 宿主机**执行：

```bash
docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'
docker inspect hilserl --format '{{.HostConfig.NetworkMode}}'
```

预期：

- `hilserl` 为 `Up`。
- 网络模式为 `host`。
- `assembly` 或其他容器没有同时运行机器人动作脚本。

两个容器可以同时存在，但同一时间只能有一个动作命令来源。首次测试前应停止 `assembly` 内的机器人应用、HIL-SERL Actor、遥操节点和其他可能发送动作的节点。不要随意停止机器人厂商 SDK/驱动本身；它负责提供 `/zj_humanoid/...` 接口。

进入容器：

```bash
docker exec -it hilserl bash
```

从本节之后，除非标记为“Orin 宿主机”，命令均在 **`hilserl` 容器内**执行。

---

## 5. Gate 1：加载 `hil-actor` 与 ROS 基础环境

进入容器后执行：

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate hil-actor

if test -f /opt/ros/noetic/setup.bash; then
  source /opt/ros/noetic/setup.bash
fi
if test -f /ros_noetic/catkin_ws/devel/setup.bash; then
  source /ros_noetic/catkin_ws/devel/setup.bash
fi
```

检查当前解释器和 ROS 环境：

```bash
which python
python --version
which catkin_make
echo "$ROS_MASTER_URI"
echo "$ROS_IP"
echo "$ROS_HOSTNAME"
```

预期 Python：

```text
/opt/conda/envs/hil-actor/bin/python
Python 3.10.20
```

现有 Compose 默认值为：

```text
ROS_MASTER_URI=http://192.168.217.1:11311
ROS_IP=192.168.217.100
ROS_HOSTNAME=192.168.217.100
```

这些只是 Compose 默认值，必须与 Orin 和机器人控制网络的真实地址一致。若 `ROS_IP`/`ROS_HOSTNAME` 不是 Orin 当前可达地址，应在 Compose 环境变量中修正后重建容器，不能在每个终端临时使用不同值。

先验证 ROS Python 模块，不发送任何机器人命令：

```bash
python -c "import rospy; print(rospy.__file__)"
python -c "import numpy, scipy; print(numpy.__version__, scipy.__version__)"
```

如果 Conda Python 无法导入 `rospy`，先检查是否正确 source ROS：

```bash
echo "$PYTHONPATH" | tr ':' '\n'
python -c "import sys; print('\n'.join(sys.path))"
```

不要用 `pip install rospy` 代替 ROS 环境；应加载基础镜像中已经构建好的 ROS Noetic Python 路径。

---

## 6. Gate 2：确认 NaviAI SDK 类型已经存在

`naviai_controller` 的 Python 源码会直接导入：

```python
from upperlimb.msg import Pose, TcpSpeed, UplimbState, Joints
from upperlimb.srv import MoveJ, MoveL
```

当前 `naviai_controller/package.xml` 中的 `zj_humanoid` 依赖仍被注释，因此 catkin 构建成功也不代表运行时一定能导入 `upperlimb`；这一 Gate 必须单独执行。

因此编译控制包之前先检查容器是否已有与真机 SDK 匹配的类型：

```bash
rospack find upperlimb
rossrv show upperlimb/MoveJ
rossrv show upperlimb/MoveL
python -c "from upperlimb.srv import MoveJ, MoveL; print('upperlimb types: ok')"
```

如果全部成功，不要重复安装 `third_party/zj_humanoid` 中的 `.run` 文件。

如果失败，先在当前正常控制 WA2 的 `assembly` 容器中只读核对所用类型包版本和路径，再为 `hilserl` 安装 **同一版本**。`naviai_controller/third_party/zj_humanoid/` 中同时存在 R3、1.2.x、1.3.x 和 1.4.0 多个安装包，不能任选一个安装。错误版本即使 import 成功，也可能因 Service MD5/字段不一致而无法与真机通信。

可在 Orin 宿主机比较两个容器：

```bash
docker exec assembly bash -lc \
  'source /opt/ros/noetic/setup.bash; rospack find upperlimb; rossrv md5 upperlimb/MoveJ; rossrv md5 upperlimb/MoveL'

docker exec hilserl bash -lc \
  'source /opt/ros/noetic/setup.bash; rospack find upperlimb; rossrv md5 upperlimb/MoveJ; rossrv md5 upperlimb/MoveL'
```

只有类型、版本和 MD5 一致后才能继续。安装 `.run` 文件可能修改容器系统目录，应先查看对应安装包说明并记录版本；本文不指定一个未经确认的版本。

---

## 7. Gate 3：初始化并只编译 `naviai_controller`

### 7.1 初始化工作空间

在容器内执行：

```bash
cd /root/catkin_ws
catkin_init_workspace src
```

检查：

```bash
ls -l /root/catkin_ws/src/CMakeLists.txt
```

该文件通常是指向 catkin 顶层 CMake 文件的符号链接，并会通过 bind mount 同步到 Orin 宿主机。

### 7.2 使用白名单构建

为了避开上游 Franka package，只构建 `naviai_controller`：

```bash
cd /root/catkin_ws
catkin_make \
  -DCATKIN_WHITELIST_PACKAGES=naviai_controller \
  -DPYTHON_EXECUTABLE=/usr/bin/python3
```

这里显式使用系统 `/usr/bin/python3` 完成 ROS/catkin 构建，避免 CMake 因当前激活的 Conda 环境选择错误的 Python 库；运行脚本时仍使用 `hil-actor` 的 Python 3.10。

构建成功后加载叠加工作空间：

```bash
source /root/catkin_ws/devel/setup.bash
```

检查包和 Python 导入来源：

```bash
rospack find naviai_controller
python -c "import naviai_controller; print(naviai_controller.__file__)"
python -c "from naviai_controller import NaviController, ArmGroup, RobotModel; print('controller import: ok')"
```

`naviai_controller.__file__` 应指向 `/root/catkin_ws/devel/...` 或当前挂载源码对应路径，不能意外指向旧的 `/root/ros_docker_test`。

白名单会保存在 catkin CMake 缓存中。以后需要同时构建自研 ROS package 时，应显式更新：

```bash
catkin_make \
  -DCATKIN_WHITELIST_PACKAGES='naviai_controller;hilserl_wa2' \
  -DPYTHON_EXECUTABLE=/usr/bin/python3
```

但当前 `hilserl_wa2` 只有 README，还不是完整 catkin package，现阶段不要把它加入白名单。若未来确实要恢复全量构建，可传空白名单；在 Franka package 仍位于工作空间时不推荐这样做：

```bash
catkin_make -DCATKIN_WHITELIST_PACKAGES=''
```

### 7.3 新终端的固定加载顺序

每次重新 `docker exec -it hilserl bash` 后执行：

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate hil-actor
source /opt/ros/noetic/setup.bash
if test -f /ros_noetic/catkin_ws/devel/setup.bash; then
  source /ros_noetic/catkin_ws/devel/setup.bash
fi
source /root/catkin_ws/devel/setup.bash
```

现有 `docker/entrypoint.hilserl.sh` 已包含这套顺序，但当前 Compose 没有配置 `entrypoint:`，因此通过普通 `docker exec ... bash` 进入时不要假定它已经自动执行。

---

## 8. Gate 4：只读熟悉 `naviai_controller`

### 8.1 先阅读这四个文件

```text
/root/catkin_ws/src/naviai_controller/src/naviai_controller/core/enums.py
/root/catkin_ws/src/naviai_controller/src/naviai_controller/naviai_controller.py
/root/catkin_ws/src/naviai_controller/src/naviai_controller/core/arm.py
/root/catkin_ws/src/naviai_controller/scripts/ts_wa2.py
```

阅读重点：

- `RobotModel.WA2`：左臂 8 轴、右臂 8 轴、颈部 2 轴、腰部 4 轴。
- `ArmGroup.LEFT`：映射到 `left_arm`。
- `get_joints(LEFT)`：从 `/zj_humanoid/upperlimb/joint_states` 的缓存读取左臂 8 轴。
- `get_tcp_rt(LEFT)`：读取 `[x,y,z,qx,qy,qz,qw]`。
- `movej()`：调用 `/zj_humanoid/upperlimb/movej/left_arm`。
- `movel()`：调用 `/zj_humanoid/upperlimb/movel/left_arm`。
- `movel_relative_base()`：读取当前 TCP，只修改基坐标系下 XYZ，最后仍调用 `movel()`。

不要直接运行完整 `scripts/ts_wa2.py`：它在只读输出后会依次进入左臂 `movej`、右臂 `movel` 和手爪测试，其中包含固定的 `+0.1 rad` 与 `+0.02 m` 示例。这些值不能被视为当前姿态下的通用安全值。

### 8.2 ROS 接口只读检查

先确认 ROS Master 可达：

```bash
rosnode list
rostopic list | grep '^/zj_humanoid/upperlimb'
rosservice list | grep '^/zj_humanoid/upperlimb'
```

重点检查：

```bash
rosservice type /zj_humanoid/upperlimb/movej/left_arm
rosservice type /zj_humanoid/upperlimb/movel/left_arm
rostopic type /zj_humanoid/upperlimb/joint_states
rostopic type /zj_humanoid/upperlimb/tcp_pose/left_arm
```

预期两个 Service 分别为：

```text
upperlimb/MoveJ
upperlimb/MoveL
```

`ArmController.movej()` 和 `movel()` 当前使用没有超时的 `rospy.wait_for_service()`。所以必须先执行上述 Service 检查，否则服务不存在时 Python 测试可能一直等待。

### 8.3 创建只读测试

建议把自研测试放到独立目录，不修改上游 HIL-SERL：

```text
/root/catkin_ws/src/hilserl_wa2/scripts/00_read_wa2_state.py
```

代码：

```python
#!/usr/bin/env python3
import time
import rospy
from naviai_controller import NaviController, ArmGroup, RobotModel


def wait_value(getter, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not rospy.is_shutdown() and time.monotonic() < deadline:
        value = getter()
        if value is not None:
            return list(value)
        rospy.sleep(0.05)
    return None


def main():
    rospy.init_node("hilserl_read_wa2_state", anonymous=False)
    ctrl = NaviController(model=RobotModel.WA2)

    left_q = wait_value(lambda: ctrl.get_joints(ArmGroup.LEFT))
    left_tcp = wait_value(lambda: ctrl.get_tcp_rt(ArmGroup.LEFT))

    if left_q is None or len(left_q) != 8:
        raise RuntimeError("未取得 WA2 左臂 8 轴状态")
    if left_tcp is None or len(left_tcp) != 7:
        raise RuntimeError("未取得左臂 TCP [x,y,z,qx,qy,qz,qw]")

    print("left joints:", left_q)
    print("left tcp   :", left_tcp)
    print("READ-ONLY WA2 CONTROLLER: PASS")


if __name__ == "__main__":
    main()
```

运行：

```bash
python /root/catkin_ws/src/hilserl_wa2/scripts/00_read_wa2_state.py
```

通过条件：

- 5 秒内取得 8 个有限的左臂关节值。
- 取得 7 个 TCP 值，四元数顺序为 `qx,qy,qz,qw`。
- 机器人没有发生运动。

只有只读 Gate 通过后才能进入动作测试。

---

## 9. Gate 5：第一次 `movej` 调用

### 9.1 调用语义

```python
ctrl.movej(
    joints=target_q,          # WA2 左臂必须为 8 个绝对关节目标
    arm=ArmGroup.LEFT,
    v=v,
    acc=acc,
    is_async=False,
)
```

内部调用：

```text
/zj_humanoid/upperlimb/movej/left_arm
```

首次测试不要手写一组绝对关节值。应读取当前 8 轴，复制后只对一个经确认的关节加入小增量，再检查所得绝对目标是否处于厂家软限位和无碰撞范围。

### 9.2 `movej` 实验脚本

建议保存为：

```text
/root/catkin_ws/src/hilserl_wa2/scripts/01_movej_left.py
```

代码：

```python
#!/usr/bin/env python3
import math
import sys
import time
import rospy
from naviai_controller import NaviController, ArmGroup, RobotModel


def required(name):
    if not rospy.has_param(name):
        raise ValueError("缺少必填参数 {}".format(name))
    return rospy.get_param(name)


def wait_left_joints(ctrl, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not rospy.is_shutdown() and time.monotonic() < deadline:
        value = ctrl.get_joints(ArmGroup.LEFT)
        if value is not None:
            return list(value)
        rospy.sleep(0.05)
    return None


def main():
    rospy.init_node("hilserl_movej_left", anonymous=False)

    index = int(required("~joint_index"))       # Python 下标 0...7
    delta = float(required("~delta"))           # rad
    v = float(required("~v"))                   # rad/s
    acc = float(required("~acc"))               # rad/s^2
    execute = bool(rospy.get_param("~execute", False))

    if not 0 <= index < 8:
        raise ValueError("joint_index 必须在 0...7")
    if not all(math.isfinite(x) for x in (delta, v, acc)):
        raise ValueError("delta/v/acc 必须是有限数值")
    if v <= 0.0 or acc <= 0.0:
        raise ValueError("v 和 acc 必须大于 0")

    ctrl = NaviController(model=RobotModel.WA2)
    current = wait_left_joints(ctrl)
    if current is None or len(current) != 8:
        raise RuntimeError("未取得 WA2 左臂 8 轴状态")

    target = current.copy()
    target[index] += delta

    print("current:", [round(x, 6) for x in current])
    print("target :", [round(x, 6) for x in target])
    print("changed: joint[{}] += {} rad".format(index, delta))

    if not execute:
        print("DRY-RUN：未发送命令；检查通过后增加 _execute:=true")
        return 0

    if input("完成限位/碰撞检查并确认急停可用；输入 MOVE：").strip() != "MOVE":
        print("取消")
        return 0

    ok = ctrl.movej(
        target,
        ArmGroup.LEFT,
        v=v,
        acc=acc,
        is_async=False,
    )
    print("movej returned:", ok)
    return 0 if ok else 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, TypeError, ValueError) as exc:
        print("ERROR:", exc)
        sys.exit(1)
```

先 dry-run，所有占位符都替换成厂家限值和当前实验计划批准的值：

```bash
python /root/catkin_ws/src/hilserl_wa2/scripts/01_movej_left.py \
  _joint_index:=JOINT_INDEX \
  _delta:=DELTA_RAD \
  _v:=V_RAD_S \
  _acc:=ACC_RAD_S2
```

确认打印的完整目标 8 轴后才执行：

```bash
python /root/catkin_ws/src/hilserl_wa2/scripts/01_movej_left.py \
  _joint_index:=JOINT_INDEX \
  _delta:=DELTA_RAD \
  _v:=V_RAD_S \
  _acc:=ACC_RAD_S2 \
  _execute:=true
```

第一轮使用 `is_async=False`。`True` 只适合已经实现到位监控、总超时和异常停止的控制状态机。

---

## 10. Gate 6：第一次 `movel` 调用

### 10.1 先理解 TCP 格式

左臂 `movel` 的目标格式为：

```text
[x, y, z, qx, qy, qz, qw]
```

- XYZ 是 TCP 在机器人基坐标系中的位置。
- 四元数顺序是 `qx,qy,qz,qw`。
- 目标数组长度始终是 7，与 WA2 左臂有 8 个关节无关。
- `movel` 要求 TCP 沿空间直线运动，控制器内部求解关节运动。

第一次测试推荐使用 `movel_relative_base()`：它读取当前 TCP、只加一个很小的 XYZ 偏移、保持当前姿态四元数，然后调用同一个 `/movel/left_arm` SDK Service。这比手工填写一个未知绝对位姿更不容易发生坐标系和四元数错误，但仍然必须做工作空间、奇异位形和碰撞检查。

### 10.2 `movel` 实验脚本

建议保存为：

```text
/root/catkin_ws/src/hilserl_wa2/scripts/02_movel_left.py
```

代码：

```python
#!/usr/bin/env python3
import math
import sys
import time
import rospy
from naviai_controller import NaviController, ArmGroup, RobotModel


def required(name):
    if not rospy.has_param(name):
        raise ValueError("缺少必填参数 {}".format(name))
    return rospy.get_param(name)


def wait_left_tcp(ctrl, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not rospy.is_shutdown() and time.monotonic() < deadline:
        value = ctrl.get_tcp_rt(ArmGroup.LEFT)
        if value is not None:
            return list(value)
        rospy.sleep(0.05)
    return None


def main():
    rospy.init_node("hilserl_movel_left", anonymous=False)

    delta = [
        float(required("~dx")),
        float(required("~dy")),
        float(required("~dz")),
    ]
    v = float(required("~v"))
    acc = float(required("~acc"))
    execute = bool(rospy.get_param("~execute", False))

    if not all(math.isfinite(x) for x in delta + [v, acc]):
        raise ValueError("dx/dy/dz/v/acc 必须是有限数值")
    if v <= 0.0 or acc <= 0.0:
        raise ValueError("v 和 acc 必须大于 0")

    ctrl = NaviController(model=RobotModel.WA2)
    current = wait_left_tcp(ctrl)
    if current is None or len(current) != 7:
        raise RuntimeError("未取得左臂 TCP")

    target = current.copy()
    target[0] += delta[0]
    target[1] += delta[1]
    target[2] += delta[2]

    print("current TCP:", [round(x, 6) for x in current])
    print("target TCP :", [round(x, 6) for x in target])
    print("base delta:", delta)

    if not execute:
        print("DRY-RUN：未发送命令；检查通过后增加 _execute:=true")
        return 0

    if input("完成工作空间/奇异点/碰撞检查；输入 MOVE：").strip() != "MOVE":
        print("取消")
        return 0

    # 该高层方法内部会重新读取当前 TCP，并调用 ctrl.movel()。
    ok = ctrl.movel_relative_base(
        delta,
        ArmGroup.LEFT,
        v=v,
        acc=acc,
        is_async=False,
    )
    print("movel returned:", ok)
    return 0 if ok else 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, TypeError, ValueError) as exc:
        print("ERROR:", exc)
        sys.exit(1)
```

先 dry-run：

```bash
python /root/catkin_ws/src/hilserl_wa2/scripts/02_movel_left.py \
  _dx:=DX_M _dy:=DY_M _dz:=DZ_M \
  _v:=V_M_S _acc:=ACC_M_S2
```

确认基坐标系方向、完整目标 TCP 和直线路径后才执行：

```bash
python /root/catkin_ws/src/hilserl_wa2/scripts/02_movel_left.py \
  _dx:=DX_M _dy:=DY_M _dz:=DZ_M \
  _v:=V_M_S _acc:=ACC_M_S2 \
  _execute:=true
```

### 10.3 直接调用绝对 `movel`

在已经取得规划器输出并完成验证后，绝对调用形式为：

```python
target_tcp = [x, y, z, qx, qy, qz, qw]
ok = ctrl.movel(
    target_tcp,
    ArmGroup.LEFT,
    v=v,
    acc=acc,
    is_async=False,
)
```

不要把物体位姿、相机坐标系位姿或 HIL-SERL 的归一化 action 直接传给 `movel()`；必须先转换到 SDK 要求的机器人基坐标系，并保证四元数归一化。

---

## 11. `movej` 与 `movel` 应该怎样学习

| 对比项 | `movej` | `movel` |
|---|---|---|
| 输入 | WA2 左臂 8 个绝对关节位置 | TCP 绝对位姿 `[x,y,z,qx,qy,qz,qw]` |
| 路径约束 | 关节空间点到点 | TCP 空间直线 |
| 适合 | 回到已示教关节姿态、较大范围转场 | 靠近工件、沿指定方向直线移动 |
| 主要风险 | TCP 路径不可直观预测、自碰撞 | 奇异位形、不可达目标、直线路径碰撞 |
| 首次目标来源 | 当前关节反馈加受限单轴增量 | 当前 TCP 加受限基坐标系偏移 |

推荐练习顺序：

1. 重复运行只读脚本，能解释每个数组的维度和单位。
2. 不执行动作，只运行 `movej` dry-run，观察修改某个关节后完整 8 轴目标如何变化。
3. 在批准的安全范围内同步执行一次 `movej`。
4. 只运行 `movel` dry-run，理解 BASE X/Y/Z 与左臂运动方向。
5. 同步执行一次经过验证的小位移 `movel`。
6. 读取动作前后状态，自己计算关节误差和 TCP 位移。
7. 最后才学习异步指令、速度控制或循环控制。

---

## 12. 与 `hil-actor` 的正确关系

`hil-actor` Conda 环境只是 Python/JAX/HIL-SERL 的运行环境；`naviai_controller` 依靠 source 后的 ROS Python 路径和 catkin devel space 工作。二者在同一 Python 进程中组合：

```text
hil-actor Python
  ├── JAX / HIL-SERL / Agentlace
  ├── rospy / upperlimb ROS 类型
  └── naviai_controller
```

当前阶段不要在 `train_rlpd.py` 中直接插入 `ctrl.movej()` 或 `ctrl.movel()`。应按以下顺序适配：

1. 在 `catkin_ws/src/hilserl_wa2` 中实现独立的 `WA2Env`。
2. 环境构造时创建一次 `NaviController(model=RobotModel.WA2)`。
3. `reset()` 使用经过审核的 home/reset 过程，不能把 Franka reset HTTP API 照搬过来。
4. `step(action)` 先反归一化、限幅和坐标转换，再调用 WA2 控制接口。
5. 为 ROS 状态新鲜度、Service 超时、到位误差、动作超时和控制器异常定义统一失败语义。
6. 先用 Mock action 验证 observation/action shape，再接入只读真机状态。
7. 真机 action 先采用单步、同步、人工确认模式。
8. 确认控制频率和停止机制后，再接入 Actor 策略与 SpaceMouse 干预。

### 12.1 强化学习动作不应直接映射成无界绝对目标

入门阶段建议的 Env action 语义是有界增量，而不是网络直接输出任意绝对姿态：

```text
策略 action（归一化）
  → clip 到 [-1, 1]
  → 乘单步最大平移/转角
  → 加到当前受控状态
  → 工作空间、关节限位和碰撞检查
  → movel 或后续经过验证的伺服接口
```

`movej`/`movel` 是点到点 Service，更适合入门、复位和低频动作验证；它们未必适合最终 HIL-SERL 的高频 step 控制。最终使用 `movel`、`speedl` 还是 `servoj`，必须依据真机 SDK 支持频率、抢占/停止语义和实测延迟决定，不能仅因接口能调用就直接用于训练。

---

## 13. 每次控制实验前的检查清单

### 软件门禁

- `hilserl` 是唯一动作客户端所在容器。
- HIL-SERL Actor、遥操和其他测试脚本尚未运行。
- `hil-actor`、ROS Noetic、基础 ROS 工作空间和 `/root/catkin_ws/devel/setup.bash` 均已加载。
- `python` 导入的 `naviai_controller` 来自 `/root/catkin_ws`。
- `upperlimb/MoveJ`、`upperlimb/MoveL` 与运行中的 SDK 版本一致。
- `movej/left_arm`、`movel/left_arm` 和状态 topic 都存在。
- 当前关节/TCP 状态不是 `None`，时间戳和内容持续更新。

### 实机门禁

- 物理急停已验证，并有现场人员可以立即触发。
- 明确左臂 8 轴顺序、单位、正方向和软限位。
- 已清空机器人本体、双臂、线缆和工装可能经过的空间。
- 目标、完整路径、负载和末端工具已经检查。
- 第一次调用为同步模式。
- 已先运行 dry-run，并保存当前状态和目标状态。
- 失败时不会自动无限重试，也不会继续发送下一动作。

`resp.success=True` 或高层返回 `True` 不能替代到位反馈检查。

---

## 14. 常见故障定位

### `catkin_init_workspace: command not found`

ROS/catkin 环境没有加载。检查：

```bash
source /opt/ros/noetic/setup.bash
which catkin_init_workspace
```

基础镜像也可能把源码构建的 ROS 放在 `/ros_noetic/catkin_ws/devel`，需要一并 source。

### 全量编译报缺少 `Franka` 或 `franka_hw`

说明没有使用白名单，CMake 尝试编译 HIL-SERL 的 Franka 控制器。重新采用：

```bash
catkin_make \
  -DCATKIN_WHITELIST_PACKAGES=naviai_controller \
  -DPYTHON_EXECUTABLE=/usr/bin/python3
```

不要为 WA2 控制实验安装 Franka 依赖。

### `No module named naviai_controller`

先检查构建是否成功，再加载：

```bash
source /root/catkin_ws/devel/setup.bash
python -c "import naviai_controller; print(naviai_controller.__file__)"
```

### `No module named rospy`

没有加载 ROS Python 路径，或 Conda 与 ROS Python 主版本不兼容。当前二者应都是 Python 3.10。不要从 PyPI 安装一个非官方 `rospy` 替代品。

### `No module named upperlimb`

NaviAI SDK ROS 类型没有安装或没有 source。按 Gate 2 与已工作的 `assembly` 容器比较版本和 MD5，不要任选一个 `.run` 安装。

### `Unable to communicate with master`

检查：

```bash
echo "$ROS_MASTER_URI"
ip addr
rosnode list
```

由于容器使用 host 网络，关键是 Orin 的真实 IP、ROS Master 地址、防火墙和机器人网络路由，而不是 Docker bridge 地址。

### Python 程序卡住但没有报错

`movej()`/`movel()` 当前在内部无超时等待 Service。先用 `rosservice list/type/info` 确认服务存在；工程化时应为封装增加超时或在调用前做有界健康检查。

### `movej` 报长度必须为 8

必须用 `RobotModel.WA2` 和 8 个左臂目标值。不能传 WA1 的 7 轴数据，也不能把 TCP 的 7 维位姿传给 `movej`。

### `movel` 报长度必须为 7

`movel` 接收 TCP 位姿 `[x,y,z,qx,qy,qz,qw]`，不是 8 维关节数组。

### Service 调用失败或返回 `False`

检查 SDK 日志、控制模式、急停、安全状态、目标可达性和控制权。高层接口只保留 `bool`，需要原始 `resp.message` 时可写一个直接 `ServiceProxy` 调试客户端，但不能自动反复发送动作。

### 容器重建后 `hil-actor` 消失

当前环境是在运行中容器的可写层手工创建：

- `docker stop/start`：保留。
- 删除容器或 `docker compose down` 后重建：丢失。

真机链路验证完成后，应按 `Dockerfile.hilserl` 固化环境，不要把当前可写层当作可重复构建结果。

---

## 15. 建议的实际执行顺序

严格按以下 Gate 推进，每一步通过后再进入下一步：

```text
Gate 0  容器、挂载、唯一控制权
  ↓
Gate 1  hil-actor + ROS Python 导入
  ↓
Gate 2  upperlimb 类型、版本和 MD5
  ↓
Gate 3  白名单 catkin_make + 包导入
  ↓
Gate 4  ROS Service/topic 清单 + 只读状态脚本
  ↓
Gate 5  movej dry-run → 人工审核 → 同步单步动作
  ↓
Gate 6  movel dry-run → 人工审核 → 同步单步动作
  ↓
记录关节/TCP反馈、耗时、返回值与异常语义
  ↓
设计 WA2Env 契约，暂不直接启动 HIL-SERL 训练
```

本阶段完成的判据不是“Actor 已经能训练”，而是：在 `hilserl/hil-actor` 中能够稳定导入正确的 ROS 控制包，读取 WA2 左臂状态，并在明确安全门禁下分别完成一次可解释、可复现的 `movej` 和 `movel` 单步调用。之后才具备设计 HIL-SERL WA2 Env 控制层的基础。
