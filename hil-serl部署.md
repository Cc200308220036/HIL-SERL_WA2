# HIL-SERL × NaviAI WA2 部署、复现与后续开发计划

> 文档基线：2026-08-11
>
> Orin 工程：`/home/naviai/hilserl_orin`
>
> 容器工程：`/root/catkin_ws`
>
> 当前容器/环境：`hilserl` / `hil-actor`

## 1. 文档用途与当前结论

本文用于回答三个问题：

1. 当前 HIL-SERL 在 WA2 上实际完成到了哪一步；
2. 从现状到 Actor–Learner 真机训练闭环还缺哪些阶段；
3. 每个阶段如何独立复现、产出什么、达到什么标准才允许进入下一阶段。

当前结论：

```text
Actor 基础运行环境                  已完成
ROS Python 与 naviai_controller     已完成
WA2 状态读取和 ServoL 真机控制       已完成基础验收
SpaceMouse 六轴 + 灵巧手按钮         已完成真机闭环
WA2 Gymnasium Env                  尚未实现
HIL-SERL Intervention Wrapper      尚未实现
相机观测流水线                      尚无本项目验收记录
Actor 单机 transition              尚未复现
Orin Actor ↔ 笔记本 Learner         尚未复现
示范、Reward Classifier、RLPD       尚未复现
最终可重建镜像                      尚未固化
```

因此，项目已经完成“机器人控制与人工遥操基础层”，但还不能表述为“HIL-SERL 已部署完成”。
下一条主线应从 **WA2Env 契约 → Mock Env → ROS Env → Intervention → Actor/Learner** 继续。

## 2. 目标系统架构

### 2.1 最终职责划分

```text
Orin / hilserl / hil-actor
  ├─ ROS1 与 WA2 驱动
  ├─ WA2Env
  ├─ 相机采集与观测预处理
  ├─ Actor 策略推理
  ├─ WA2SpacemouseIntervention
  ├─ 在线 transition / 干预 transition
  └─ Agentlace TrainerClient
                 │
                 │ TCP 5588/5589
                 ▼
Ubuntu 笔记本 / Learner
  ├─ 相同 HIL-SERL 源码与实验配置
  ├─ TrainerServer
  ├─ online replay buffer
  ├─ demo/intervention buffer
  ├─ RLPD/SAC 更新
  ├─ checkpoint / 日志
  └─ 策略参数广播
```

### 2.2 控制权原则

以下两个程序不能同时控制同一机械臂：

```text
手动遥操：spacemouse_wa2_teleop.py
HIL-SERL：Actor → WA2Env.step(action)
```

现有遥操脚本是：

- 六轴方向、比例和 ServoL 安全语义的真机基准；
- 独立人工操作工具；
- 后续 Intervention Wrapper 的逻辑来源。

它不应在 Actor 运行时继续直接发布 ServoL。正式 HIL-SERL 链路必须保证：

```text
SpaceMouse 只产生 Env action
→ Wrapper 决定是否覆盖 policy action
→ WA2Env 是唯一机器人动作出口
```

否则会产生两个 ServoL 发布者和不可控的动作竞争。

## 3. 当前代码、环境与复现边界

### 3.1 目录

```text
宿主机工程：/home/naviai/hilserl_orin
容器工作空间：/root/catkin_ws

catkin_ws/src/hil-serl-main       HIL-SERL 上游快照
catkin_ws/src/hilserl_wa2         WA2 自研适配代码
catkin_ws/src/naviai_controller   WA2 控制封装
catkin_ws/src/joystick_drivers    ROS1 spacenav_node
docker/                           Compose、依赖锁、环境验收脚本
artifacts/wheels/                 修正版 Agentlace wheel
configs/                          后续 Actor/Learner/实验配置
调试日志/                         分阶段实测证据
```

Compose 挂载：

```text
/home/naviai/hilserl_orin/catkin_ws → /root/catkin_ws
```

只有 `catkin_ws` 双向挂载；`docker/`、`docs/`、`configs/` 和 `artifacts/` 默认不在容器内。
需要在容器运行仓库外的验收脚本时，应从 Orin 宿主机通过 `docker exec -i` 输入，或明确新增只读
挂载，不能假设容器能直接访问宿主机路径。

### 3.2 当前环境基线

```text
基础镜像                    ros1_docker:latest
容器                        hilserl
Conda                       hil-actor
Python                      3.10.20
JAX/JAXLIB                  0.4.35
JAX CUDA plugin/PJRT        0.4.35
NumPy                       1.26.4
SciPy                       1.15.3
Flax                        0.10.2
Optax                       0.2.4
Agentlace                   0.1.3（本地修正版 wheel）
Gym                         0.26.2
Gymnasium                   1.2.2
OpenCV                      4.10.0
TensorFlow/TF-Keras         2.21.0
TensorFlow Probability      0.25.0
```

JAX/Orin 当前约束：

```bash
export XLA_FLAGS=--xla_gpu_autotune_level=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.1
```

同一进程中应先导入 JAX，再导入 TensorFlow，避免 XLA protobuf 重复注册。

### 3.3 固定环境加载命令

进入容器后统一执行：

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate hil-actor
source /ros_noetic/catkin_ws/devel/setup.bash
source /opt/ros/noetic/setup.bash --extend
source /root/catkin_ws/devel/setup.bash
cd /root/catkin_ws
```

### 3.4 上游与自研边界

- `hil-serl-main` 尽量保持上游快照；WA2 代码优先放入 `hilserl_wa2`；
- 新实验配置可以在 WA2 包中先开发，接入时再以最小改动注册到
  `examples/experiments/mappings.py`；
- `/home/naviai/ros_docker_test` 始终只作为来源，不在本项目中修改；
- 当前阶段继续使用手工验收的 `hilserl/hil-actor`，尚不构建最终镜像；
- `naviai_controller` 基础提交为 `38db56b`，当前存在 WA2、ServoL 和测试相关未提交改动；
- HIL-SERL 上游 revision 尚未写入项目清单，必须在复现阶段 0 补齐。

## 4. 当前进度矩阵

| 模块 | 状态 | 已有证据 | 剩余问题 |
|---|---|---|---|
| 独立工作空间/Compose | 已完成 | `hilserl` 可运行，catkin_ws 双向同步 | 最终镜像未固化 |
| `hil-actor` 依赖 | 已完成 | `pip check`、核心导入通过 | 需重新生成可复现快照 |
| JAX GPU | 已完成 | `CudaDevice(id=0)`、JIT/卷积规避参数通过 | 最终镜像重建后需复验 |
| Agentlace 本机通信 | 已完成 | transition 上传、参数广播通过 | 跨机尚未完成 |
| ROS Python/类型包 | 已完成 | `rospy/upperlimb/naviai_controller` 可导入 | 消息源码来源需最终固化 |
| WA2 状态读取 | 已完成基础验收 | 关节/TCP/速度/手部状态可读 | 力/急停/错误语义仍需整理 |
| MoveL/ServoL | 已完成基础验收 | MoveL、XYZ、Roll ±2°实机通过 | reset/home、控制权仲裁未定 |
| SpaceMouse 输入 | 已完成 | ROS Joy 六轴和按钮标定 | 丝滑度仍需优化 |
| 六轴遥操 | 已完成 | XYZ+姿态真机可控，stop/clear 正常 | 当前一次只输出主导轴 |
| 灵巧手按钮 | 已完成 | 右键抓握/释放成功 | 即时反馈早于机械到位 |
| WA2Env 契约 | 未完成 | 无 | 下一主任务 |
| Mock WA2Env | 未完成 | 无 | 需离线单元测试 |
| ROS WA2Env | 未完成 | 无 | 需唯一动作出口和超时 |
| 相机观测 | 未复现 | 上游有 RealSense 代码 | 本机序列号/帧率/裁剪未知 |
| Intervention Wrapper | 未完成 | 手动遥操逻辑可复用 | 尚无 `intervene_action` |
| WA2 实验 config | 未完成 | 仅 Franka 示例 config | fake_env 与 action/obs 未定义 |
| Actor 单机 | 未完成 | 核心模块可导入 | 未产生 WA2 transition |
| 笔记本 Learner | 未完成 | 无本项目验收 | 环境、配置、数据均待建 |
| 跨机 Agentlace | 未完成 | 仅 localhost PASS | 防火墙/断线/重连待验收 |
| 示范/分类器 | 未完成 | 上游脚本存在 | WA2 config、相机、任务未定 |
| 小规模训练 | 未完成 | 无 | 必须在前序 Gate 全部通过后进行 |

项目当前处在以下分界点：

```text
环境与机器人控制基础层：完成
HIL-SERL 环境/数据/训练层：从 0 开始接入
```

## 5. 已确认的关键技术决策

### 5.1 WA2 不复用 Franka Robot Server

上游链路为：

```text
FrankaEnv → HTTP → Flask Server → Franka ROS Controller
```

WA2 推荐采用：

```text
WA2Env → NaviController/ROS topic/service → WA2
```

原因：

- 当前 WA2 ROS 控制链已打通；
- Franka 的 compliance、gripper、reset 和控制器语义不适用于 WA2；
- 增加 HTTP 层只会制造新的状态同步和超时问题。

### 5.2 连续控制使用 ServoL，不使用 SpeedL

当前 SDK 1.3.2 的 `speedl` 运行链不可用；已验证的连续控制入口为：

```text
/zj_humanoid/upperlimb/set_servo_params
/zj_humanoid/upperlimb/servol/{left,right}_arm
/zj_humanoid/upperlimb/stop
/zj_humanoid/upperlimb/clear_servo_params
```

固定验收参数：

```text
servo_time = 0.02
servo_gain = 800
控制周期 = 50 Hz
```

### 5.3 HIL-SERL 使用 ROS Joy Intervention，不复用直接 HID 控制链

当前 Orin 已稳定运行：

```text
spacenavd → spacenav_node → /spacenav/joy
```

后续 `WA2SpacemouseIntervention` 应订阅 `/spacenav/joy` 并复用：

- `SpaceMouseInputProcessor`；
- 已标定的 axis map/sign；
- deadman、watchdog、按钮边沿；
- 灵巧手动作编码。

不应同时启动上游 `SpaceMouseExpert` 直接 HID 进程，以免两个进程竞争同一设备。

### 5.4 第一阶段先做单左臂

建议顺序：

1. 左臂 6D、灵巧手固定或人工控制；
2. 左臂 7D（含手部动作）；
3. 单臂 HIL-SERL 闭环；
4. 最后扩展双臂。

直接从 14D 双臂开始会同时放大 reset、动作同步、相机、干预和训练调试复杂度。

## 6. WA2Env 初始契约建议

本节是待评审建议，不代表已经实现。复现阶段 1 必须将其冻结为明确文档和配置。

### 6.1 初始动作空间

第一版建议 6D：

```text
action.shape = (6,)
action.dtype = float32
action range = [-1, 1]

[dx, dy, dz, droll, dpitch, dyaw]
```

Env 内部将归一化动作映射为每步 TCP 增量，建议首个真机 Gate：

```text
单步最大平移：0.5～1.0 mm
单步最大旋转：0.1～0.25°
周期：20 ms / 50 Hz
坐标系：tool 或 base 必须在 config 中固定
```

灵巧手第一阶段可保持预抓握或由 episode 外部控制。6D Env 稳定后再扩展为 7D：

```text
[dx, dy, dz, droll, dpitch, dyaw, hand]
```

`hand` 必须明确是连续目标、三态命令还是边沿事件，不能直接把当前“按钮 toggle”隐式塞入
策略动作。

### 6.2 初始观测空间候选

```text
observation = {
  "state": {
    "tcp_pose":       float32[7],   # m + quaternion xyzw
    "tcp_vel":        float32[6],
    "joint_pos":      float32[8],
    "hand_joints":    float32[6]
  },
  "images": {
    "<policy_camera>": uint8[H,W,3],
    "<wrist_camera>":  uint8[H,W,3]   # 若实际安装
  }
}
```

奇异、状态年龄、cmd_num、错误码和急停更适合作为 `info`/安全门控，而不是直接作为策略输入，
除非后续实验明确需要。

### 6.3 Step/Reset 语义

```text
reset() -> observation, info
step(action) -> observation, reward, terminated, truncated, info
close() -> stop + clear ServoL + 释放资源
```

必须明确：

- `terminated`：任务成功或确定的任务失败；
- `truncated`：最大步数、状态超时、安全边界或人工中止；
- reward：第一阶段可以为 0/外部占位，但不能伪装成已完成分类器；
- reset/home：必须是可验证动作序列，不允许仅把内部计数清零；
- 任一异常路径必须先停止机器人，再抛异常或返回 truncated。

## 7. 全局安全门禁

任何会发送真机动作的阶段均必须满足：

1. 物理急停可触及；
2. 软件 stop 终端已准备；
3. 对应手臂 `is_singular=False`；
4. Joy、TCP、UplimbState 反馈新鲜；
5. 只有一个动作发布者；
6. 工作空间和单步增量在 Env 内二次裁剪；
7. Actor/Env 异常退出会执行 stop/clear；
8. 网络断开不能继续重复旧动作；
9. reset、策略动作和人工干预使用同一安全执行器；
10. 每个新阶段先 dry-run/Mock，再状态只读，最后才是真机。

软件停止：

```bash
rosservice call /zj_humanoid/upperlimb/stop "{}"
```

## 8. 分阶段复现计划总览

| 阶段 | 名称 | 风险 | 预计工作量 | 当前状态 | 是否适合作为单日任务 |
|---|---|---|---|---|---|
| R0 | 基线环境与源码快照复验 | 无运动 | 1～2 h | 待重新记录 | 是 |
| R1 | ROS 接口清单与 WA2Env 契约 | 只读 | 3～6 h | 未完成 | **推荐** |
| R2 | Mock WA2Env 与单元测试 | 无硬件 | 4～8 h | 未完成 | 是，需 R1 |
| R3 | ROS 状态只读 Env | 只读真机 | 3～6 h | 未完成 | 是，需 R2 |
| R4 | 安全 ServoL Action Executor | 小幅运动 | 4～8 h | 未完成 | 是，需 R3 |
| R5 | reset/home 与 episode 生命周期 | 真机运动 | 4～8 h | 未完成 | 视姿态方案 |
| R6 | RealSense 观测流水线 | 设备占用 | 4～8 h | 未复现 | 是，可并行 |
| R7 | WA2 SpaceMouse Intervention | 小幅运动 | 4～8 h | 未完成 | 需 R4 |
| R8 | WA2 实验 Config 与 fake_env | 无硬件 | 4～8 h | 未完成 | 需 R2/R6契约 |
| R9 | Actor 本地 transition 闭环 | 小幅运动 | 4～8 h | 未完成 | 需 R7/R8 |
| R10 | 跨机 Actor–Learner 通信 | 可先无运动 | 3～6 h | 未完成 | 需 R8 |
| R11 | 示范数据采集 | 真机任务 | 1～2 d | 未完成 | 需 R5～R9 |
| R12 | Reward Classifier | 训练/相机 | 1～2 d | 未完成 | 需 R6/R11 |
| R13 | 小规模 HIL-SERL 训练 | 高风险 | 多日 | 未完成 | 不建议立即选择 |
| R14 | 最终镜像与全新容器复现 | 无/后续真机 | 1～2 d | 未完成 | 最后阶段 |

## 9. R0：基线环境与源码快照复验

### 9.1 目标

证明当前结果不是依赖未记录的终端状态，并生成后续每次实验都能引用的基线清单。

### 9.2 操作

Orin 宿主机：

```bash
cd /home/naviai/hilserl_orin
docker ps --filter name=hilserl
(cd artifacts/wheels && sha256sum -c SHA256SUMS)

git -C catkin_ws/src/hil-serl-main rev-parse HEAD
git -C catkin_ws/src/hil-serl-main status --short
git -C catkin_ws/src/naviai_controller rev-parse HEAD
git -C catkin_ws/src/naviai_controller status --short
```

记录 dirty 文件，不要为了得到 clean 状态而执行 reset/checkout。当前 WA2 改动本来就尚未提交。

容器：

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate hil-actor
source /ros_noetic/catkin_ws/devel/setup.bash
source /opt/ros/noetic/setup.bash --extend
source /root/catkin_ws/devel/setup.bash

python --version
python -m pip check
python -c "import rospy, upperlimb, naviai_controller; print('WA2 ROS IMPORT: PASS')"

export XLA_FLAGS=--xla_gpu_autotune_level=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.1
python -c "import jax; print(jax.devices()); assert any(d.platform == 'gpu' for d in jax.devices()); print('JAX GPU: PASS')"

cd /root/catkin_ws
python -m unittest discover -s src/hilserl_wa2/tests/unit -v
```

从 Orin 宿主机执行仓库外的 Agentlace 验收脚本：

```bash
cd /home/naviai/hilserl_orin
docker exec -i hilserl bash -lc \
  'source /opt/conda/etc/profile.d/conda.sh && conda activate hil-actor && python -' \
  < docker/verify_agentlace.py
```

### 9.3 交付物

- `调试日志/<日期>HIL-SERL基线复验.md`；
- HIL-SERL commit；
- naviai_controller base commit 与 dirty 文件清单；
- 容器 ID、镜像 ID；
- `pip check`、JAX GPU、Agentlace、27 项测试输出。

### 9.4 验收标准

- wheel SHA-256：PASS；
- `pip check`：无 broken requirements；
- JAX 至少一个 GPU device；
- Agentlace transition 上传和 network broadcast：PASS；
- 27 项 WA2/SpaceMouse 单元测试：PASS；
- 所有源码 revision 和 dirty 状态已记录；
- 不发生机器人运动。

## 10. R1：ROS 接口清单与 WA2Env 契约

### 10.1 目标

在不发送动作的前提下，将 Env 需要的状态、动作、reset、安全和任务语义写成可评审契约。

### 10.2 操作

只读盘点：

```bash
rostopic list | sort
rosservice list | sort
rosnode list | sort

rostopic info /zj_humanoid/upperlimb/uplimb_state
rostopic info /zj_humanoid/upperlimb/tcp_pose/left_arm
rostopic info /zj_humanoid/upperlimb/tcp_speed/dual_arm
rostopic info /zj_humanoid/upperlimb/joint_states
rostopic info /zj_humanoid/hand/joint_states

rostopic hz /zj_humanoid/upperlimb/uplimb_state
rostopic hz /zj_humanoid/upperlimb/tcp_pose/left_arm
rostopic hz /zj_humanoid/upperlimb/joint_states
```

对关键消息各保存至少 10 组样本，确认：

- shape、dtype、单位和坐标系；
- 时间戳来源和发布频率；
- 四元数顺序；
- WA2 左臂 8 关节布局；
- 灵巧手 6 维布局；
- 奇异、cmd_num、错误/急停字段。

形成：

```text
docs/WA2Env接口契约.md
configs/experiments/wa2_env_contract.yaml
```

契约至少包含：

- observation/action space；
- 50 Hz step 时序；
- action scale 和坐标系；
- 工作空间、关节和单步边界；
- reset/home 候选流程；
- reward/terminated/truncated 初始定义；
- stale、singular、stop、close 和异常策略；
- 灵巧手 6D/7D 分阶段方案；
- 相机键名占位及缺失策略。

### 10.3 验收标准

- 所有策略输入字段都有 shape、dtype、单位、来源和新鲜度要求；
- 所有 action 分量都有归一化范围和物理缩放；
- reset/home 不再是 `TBD`，至少选定安全候选方案和人工确认点；
- 明确唯一动作发布者和 stop/clear 生命周期；
- 明确第一版使用 6D 还是 7D；
- Actor 与 Learner 能仅凭契约构造相同 space；
- 本阶段无机器人动作。

## 11. R2：Mock WA2Env 与单元测试

### 11.1 目标

在没有 ROS、相机和机器人时实现 Gymnasium API、空间、限幅和生命周期。

### 11.2 建议文件

```text
catkin_ws/src/hilserl_wa2/envs/__init__.py
catkin_ws/src/hilserl_wa2/envs/wa2_env.py
catkin_ws/src/hilserl_wa2/envs/contracts.py
catkin_ws/src/hilserl_wa2/ros_adapters/mock_robot.py
catkin_ws/src/hilserl_wa2/tests/unit/test_wa2_env.py
```

### 11.3 必测场景

- `reset()` 返回值属于 observation_space；
- `step()` 五元组符合 Gymnasium；
- action 自动 clip 到 `[-1,1]`；
- 物理增量不超过契约；
- NaN/Inf、shape 错误被拒绝；
- 超过最大步数返回 truncated；
- success 返回 terminated；
- close 可重复调用；
- fake_env 构造不 import rospy、不访问相机和机器人；
- 固定 seed 时 Mock 结果可复现。

### 11.4 验收标准

```text
gymnasium.utils.env_checker.check_env(env) 通过
连续 1000 个随机 action 无异常
所有 observation 均落在 space 内
所有 action 均经过可验证的物理限幅
测试不需要 ROS Master 和硬件
```

## 12. R3：ROS 状态只读 Env

### 12.1 目标

将真实 WA2 状态放入 Env observation，但 `step()` 暂不发送 ServoL。

### 12.2 实现要求

- 抽出 `WA2StateMonitor`，不要把 subscriber 直接散落在 Env；
- 使用 monotonic 时间戳判断 TCP/UplimbState/关节/手部新鲜度；
- 返回副本，避免调用方修改订阅缓存；
- 将 singular、cmd_num、state ages 放入 info；
- 只读模式明确标识，不允许误发动作；
- ROS shutdown/话题停止时能结束，而非永久阻塞。

### 12.3 验收标准

- 机器人保持静止；
- 连续读取至少 10 分钟，无 NaN、shape 变化和内存持续增长；
- observation 每个字段都属于 space；
- 断开一个关键 topic 后，在规定 watchdog 内返回明确 stale/truncated/error；
- 恢复 topic 后的恢复策略与契约一致；
- 左右臂数据不会串侧。

## 13. R4：安全 ServoL Action Executor

### 13.1 目标

让 WA2Env 成为唯一动作出口，复用已经验证的 ServoL、PoseIntegrator 和 stop/clear 语义。

### 13.2 实现要求

建议抽出：

```text
WA2ServoSession
  ├─ start()
  ├─ apply_normalized_action(action, dt)
  ├─ stop()
  ├─ health()
  └─ close()
```

必须做到：

- `set_servo_params(0.02,800,arm)`；
- action → TCP 目标的转换只存在一份；
- 单步限制、episode workspace 和全局安全 box 分层处理；
- TCP/state stale、奇异、异常、KeyboardInterrupt 都 stop/clear；
- policy、SpaceMouse 和 reset 最终都通过同一个执行器；
- 记录 command target、measured TCP、loop dt 和跟踪误差。

### 13.3 真机 Gate

依次执行：

1. 零 action 保持；
2. X +1 mm / -1 mm；
3. Y/Z ±1 mm；
4. Roll/Pitch/Yaw ±2°；
5. action clip；
6. dead process/异常退出停止。

### 13.4 验收标准

- 零 action 无明显漂移；
- 平移方向正确，目标误差满足契约；
- 旋转方向正确，轴外旋转和位置漂移不高于 0810 基线量级；
- 每个动作不超过单步和工作区边界；
- stale/singular/异常后不再发布新目标；
- `stop=True clear=True`；
- 同一时刻只有一个 ServoL 发布者。

## 14. R5：reset/home 与 episode 生命周期

### 14.1 目标

建立可重复 episode，而不是靠人工随意把机器人挪回起点。

### 14.2 需要确定

- 左臂安全 home 8 关节目标；
- 任务 reset TCP/关节目标；
- 灵巧手 reset 目标；
- reset 使用 MoveJ、MoveL 还是分段组合；
- 障碍物/物体摆放的人工确认；
- reset 超时、到位误差和失败恢复；
- episode 最大步数和人工 abort。

### 14.3 验收标准

- 连续执行至少 10 次 reset；
- 每次最终关节/TCP 误差在契约范围内；
- 无碰撞、无奇异、无线缆风险；
- reset 超时会停止且不会进入 episode；
- reset 期间策略和 SpaceMouse 不能抢占；
- episode 结束、Ctrl+C 和异常均能安全 close。

## 15. R6：RealSense 观测流水线

### 15.1 目标

建立 WA2 任务真实图像观测。本项目目前没有相机序列号、安装位姿、裁剪和稳定性验收记录，
不能直接复制 Franka 示例中的序列号与 crop。

### 15.2 操作

- 盘点容器内可见 RealSense 设备与序列号；
- 确认相机归属、USB 带宽和分辨率/帧率；
- 选择 policy camera 与 classifier camera；
- 固定 RGB 格式、resize、crop、dtype；
- 明确时间戳以及与机器人状态的最大允许偏差；
- fake_env 使用同 shape 的零图或样本图，不打开设备。

### 15.3 验收标准

- 每个配置相机序列号唯一；
- 连续采集 10 分钟无崩溃；
- 实测平均 FPS 达到配置要求，丢帧率有记录；
- 图像 shape/dtype 固定并属于 observation_space；
- 相机断开在超时内触发 truncated/安全停止；
- Actor 和其他容器不会同时占用相机；
- policy crop 和 classifier crop 均保存样例图供人工评审。

## 16. R7：WA2 SpaceMouse Intervention

### 16.1 目标

把已验证的 `/spacenav/joy` 转为 Env action，并在干预时写入：

```python
info["intervene_action"]
```

### 16.2 实现要求

- 新建 WA2 专用 Wrapper，不直接使用 Franka 的 HID `SpaceMouseExpert`；
- 复用 axis map/sign、deadman、deadzone、filter 和 watchdog；
- 输出必须与 Env action_space 完全一致；
- 无有效输入时原样返回 policy action；
- 有 deadman+有效输入时覆盖 policy action；
- 手部按钮按 6D/7D 契约编码；
- Wrapper 不能直接调用 ServoL/hand Service；
- 手动遥操脚本和 Actor 互斥运行。

### 16.3 离线/真机验收

- 合成 Joy 12 个方向全部映射正确；
- Joy stale 后不继续干预；
- 无输入时 policy action 不变；
- 有输入时 `intervene_action` 与实际执行 action 相同；
- Actor 普通 transition 进入 `actor_env`；
- 干预 transition 同时进入 `actor_env_intvn`；
- intervention_count/intervention_steps 正确；
- 真机小范围动作和 0810 手动遥操方向一致。

### 16.4 丝滑度并行优化

当前主导轴策略能安全工作，但方向切换有卡顿。示范采集前至少评估：

- 平移组三轴连续输出；
- 旋转组三轴连续输出；
- 组间仍互斥；
- 主轴切换不清空全部滤波状态；
- 交叉淡入淡出或 6×6 解耦。

优化后必须重新跑 27 项离线测试及 2°/20 mm 小范围 Gate。

## 17. R8：WA2 实验 Config 与 fake_env

### 17.1 目标

实现一个能被上游入口加载的 WA2 实验配置，并保证 Learner 构造环境时不接触硬件。

建议新增：

```text
catkin_ws/src/hil-serl-main/examples/experiments/wa2_<task>/config.py
catkin_ws/src/hil-serl-main/examples/experiments/wa2_<task>/run_actor.sh
catkin_ws/src/hil-serl-main/examples/experiments/wa2_<task>/run_learner.sh
```

并最小修改：

```text
examples/experiments/mappings.py
```

### 17.2 fake_env 硬要求

`get_environment(fake_env=True)`：

- 不初始化 rospy；
- 不访问 ROS Master；
- 不打开 SpaceMouse；
- 不打开 RealSense；
- 不请求机器人状态；
- 仍返回与 Actor 完全相同的 observation_space/action_space；
- 能构造 SAC/RLPD agent 和 replay buffer。

### 17.3 验收标准

- 新 `exp_name` 可在 `CONFIG_MAPPING` 查到；
- Actor/learner 两侧 space 序列化后完全一致；
- 笔记本无机器人网络时也可构造 fake env；
- `make_sac_pixel_agent` 能完成初始化；
- Replay Buffer 能插入/采样一条合法 transition；
- 不误加载 Franka server URL、相机序列号或安全边界。

## 18. R9：Actor 本地 transition 闭环

### 18.1 目标

先在 Orin 本机用 dummy TrainerServer 验证真实 WA2Env → Actor transition → Agentlace，不立即跨机。

### 18.2 顺序

1. fake action/no robot 模式产生 transition；
2. TrainerServer 收到 `actor_env`；
3. 人工干预后收到 `actor_env_intvn`；
4. Server 广播一组参数/测试 payload；
5. Actor callback 收到；
6. 最后才允许低幅真机 action。

### 18.3 验收标准

- 连续至少 100 条 transition schema 一致；
- observations/actions/next_observations 无 NaN；
- masks/dones 与 terminated/truncated 语义一致；
- 干预数量与实际操作一致；
- buffer dump 可重新加载；
- Server 停止或 Actor 异常时机器人立即 stop；
- 不依赖 W&B，先使用 `--debug`。

## 19. R10：Orin Actor ↔ 笔记本 Learner

### 19.1 网络约束

```text
TCP 5588：transition/request
TCP 5589：网络参数广播
```

- `--ip` 填 Learner 笔记本 IP，不是 Orin IP；
- Learner 先启动，Actor 后启动；
- 防火墙只开放所需局域网来源；
- 两侧使用相同源码 revision、实验 config、Agentlace 协议和模型结构。

### 19.2 第一轮不连接机器人

先发送 fake transition：

- 1000 条普通 transition；
- 一组 intervention transition；
- 至少一次 network publish；
- 主动断开和重启 Learner；
- 记录 Actor 行为。

### 19.3 验收标准

- Learner 两个 DataStore 数量正确增长；
- Actor 收到至少一次参数广播；
- transition 无丢失/重复，或已明确 Agentlace 的可接受语义；
- 端口监听、时延和吞吐有记录；
- Learner 重启后的恢复方式明确；
- 网络断开期间机器人安全停止或进入人工模式，不继续旧策略；
- 两侧保存同一份 config hash 和源码 revision。

## 20. R11：示范数据采集

### 20.1 前置条件

R5 reset、R6 相机、R7 Intervention、R8 config、R9 transition 必须通过。

入口：

```text
examples/record_demos.py
```

第一轮建议：

- 选择单一、短时、低风险任务；
- 先采 5 条用于 schema 验证；
- 再采至少 20 条成功示范；
- 失败/中止轨迹不得混入成功 demo；
- 每条轨迹保存任务版本、config hash、时间和操作者。

### 20.2 验收标准

- `.pkl` 可在笔记本无硬件环境加载；
- 每条 transition shape/dtype 一致；
- action 是实际执行动作，包括 intervention override；
- 图像、状态、action 时间顺序正确；
- demo 数量和成功数与人工记录一致；
- 随机抽查回放方向正确；
- 数据已复制到笔记本并校验 SHA-256。

## 21. R12：成功/失败数据与 Reward Classifier

入口：

```text
examples/record_success_fail.py
examples/train_reward_classifier.py
```

### 21.1 工作内容

- 固定任务成功定义和相机视角；
- 分别采集成功/失败图像；
- 数据划分 train/validation/test，避免同一轨迹跨集合泄漏；
- 在笔记本训练 classifier；
- 选择阈值并在真实在线图像复验；
- classifier 误判时不得直接触发危险 reset。

### 21.2 验收标准

- 三个数据划分独立；
- 记录 precision、recall、混淆矩阵，而不只看训练 accuracy；
- 测试集达到任务约定指标；
- 真实连续画面上误触发率可接受；
- checkpoint 能在 Orin 加载；
- Actor reward 输出与人工判断抽查一致；
- classifier 输入 key/crop 与训练完全一致。

## 22. R13：小规模 HIL-SERL 训练

### 22.1 前置条件

R0～R12 的相关 Gate 全部通过，尤其是 reset、唯一动作出口、断网停止和 demo buffer。

### 22.2 第一轮限制

- `--debug` 禁用 W&B 外部依赖；
- 低 action scale；
- 短 episode；
- 操作者持续握有急停和 SpaceMouse；
- 先只跑少量 episode 验证数据流，不以成功率为目标；
- 每次策略版本更新都记录；
- 出现异常动作立即回退到固定 checkpoint 或停止。

### 22.3 验收标准

- online/demo buffer 均持续采样；
- Learner loss/gradient/参数无 NaN/Inf；
- Actor 周期性收到新参数；
- checkpoint 可保存、恢复并继续；
- reward、episode、intervention 统计一致；
- 网络、Actor 或 Learner 异常均不会导致机器人继续旧动作；
- 全程没有越界、奇异、碰撞和无法恢复故障。

## 23. R14：最终镜像与全新容器复现

只有真机闭环稳定后才执行。

### 23.1 固化内容

- 基础镜像 digest；
- Python/Conda/系统包；
- JAX ARM64 安装来源；
- `actor-requirements.txt`；
- Agentlace wheel 和 SHA-256；
- HIL-SERL/WA2/naviai_controller revision；
- ROS 消息类型包来源；
- entrypoint 和环境变量；
- 相机/USB/ROS 网络配置模板；
- 构建期无硬件测试和运行期硬件 Gate。

### 23.2 验收标准

- 删除旧容器不是第一步；先保留旧容器作为回退；
- 从 Dockerfile 构建版本化镜像；
- 从新镜像创建全新容器；
- 不复制旧容器 Conda 目录；
- R0 全部通过；
- Mock/fake_env、JAX、Agentlace、ROS import 通过；
- 相机、SpaceMouse、WA2 状态与小幅动作 Gate 通过；
- Actor–Learner 和小规模训练至少复验一次；
- 文档中的命令可由另一人从零执行。

## 24. 数据、日志与恢复策略

建议：

```text
Orin:
  runs/<exp>/<timestamp>/
    config.yaml
    source_versions.txt
    buffer/
    demo_buffer/
    videos/
    logs/

Learner:
  runs/<exp>/<timestamp>/
    config.yaml
    checkpoints/
    demos/
    classifier_ckpt/
    logs/
```

Agentlace 只同步实时 transition、请求和参数，不自动同步：

- checkpoint；
- 历史 buffer；
- demo 文件；
- classifier；
- 视频和调试日志。

每次同步都应保存 SHA-256。每次实验至少记录：

```text
日期/操作者
任务名和 exp_name
HIL-SERL/WA2/naviai_controller revision
dirty diff 摘要
容器/镜像 ID
依赖版本
config 文件及 hash
相机序列号
机器人起始状态
验收结果和异常
```

## 25. 已知风险与当前阻塞

### 25.1 reset/home 尚未定义

这是 WA2Env 真机 episode 的主要阻塞。没有可重复 reset，就不能安全采示范或训练。

### 25.2 相机未完成本项目验收

上游 Franka config 的相机序列号、crop 和 exposure 不能复用。视觉 Actor、示范和 classifier
均依赖 R6。

### 25.3 当前 SpaceMouse 是手动控制器，不是 HIL Intervention

真机遥操成功不等于 `info["intervene_action"]`、`actor_env_intvn` 和 demo buffer 已打通。

### 25.4 SpaceMouse 丝滑度

当前主导轴安全策略会在换轴时卡顿。它不阻塞 R1/R2/R3，但应在 R7/R11 之前优化，否则示范
动作质量受影响。

### 25.5 灵巧手反馈滞后

Service 返回快于机械到位。7D Env 或 episode 条件若依赖手部状态，必须加入延迟确认和超时。

### 25.6 fake_env 必须硬件隔离

Learner 不应初始化 ROS、机器人、SpaceMouse 或相机。不能直接继承 FrankaEnv 的硬件初始化缺陷。

### 25.7 源码状态尚未冻结

`naviai_controller` 有未提交 WA2 改动；HIL-SERL revision 尚未记录。R0 必须先保存版本证据，
但不得为了“干净”而丢弃用户改动。

### 25.8 当前环境位于容器可写层

误删/重建 `hilserl` 会丢失手工 Conda 环境。在 R14 前使用 stop/start，不使用 `docker rm` 或
`docker compose down` 重建服务。

## 26. 今天可选择的阶段任务

### 选项 A：R0 基线环境复验

适合目标：先确保后续所有开发都有可靠版本和环境基线。

当天产物：

- 一份完整基线日志；
- 源码 revision/dirty 清单；
- 环境、JAX、Agentlace、27 项测试输出。

风险：无机器人运动。工作量：约 1～2 小时。

### 选项 B：R1 WA2Env 接口契约（推荐）

适合目标：立即推进 HIL-SERL 主线，又不进行真机动作。

当天产物：

- `docs/WA2Env接口契约.md`；
- `configs/experiments/wa2_env_contract.yaml`；
- ROS topic/service 频率和字段样本；
- 第一版 6D/7D、observation 和 reset 决策。

风险：只读。工作量：约半天到一天。

### 选项 C：R2 Mock WA2Env

适合目标：如果 R1 契约已经人工确认，可直接进入代码实现。

当天产物：

- Mock Env；
- Gymnasium check_env；
- action clip、终止/截断和 fake_env 单元测试。

风险：无硬件。工作量：约一天。

### 选项 D：SpaceMouse 丝滑度优化

这是并行支线，不替代 WA2Env 主线。适合先提高示范质量。

当天产物：

- 组内多轴或切换淡入淡出方案；
- 扩展离线标定测试；
- dry-run 和低速真机对比日志。

风险：需要小范围真机复验。工作量：约半天到一天。

建议选择顺序：

```text
如果今天强调“复现”      → A
如果今天强调“推进主线”    → B（推荐）
如果 R1 已经评审确认       → C
如果示范手感是当前最高优先 → D
```

## 27. 最终完成定义

只有以下条件全部满足，才可称为“HIL-SERL 已在 WA2 上完成部署”：

- 全新环境可以按锁定依赖复建；
- WA2Env observation/action/reset/close 完整；
- 相机观测稳定；
- 策略和 SpaceMouse 共用唯一安全执行器；
- `intervene_action` 和双 Replay Buffer 数据正确；
- Actor 单机稳定；
- 笔记本 Learner 可独立 fake_env 构造；
- 跨机 transition 和参数广播稳定；
- demo 与 classifier 可复现；
- 小规模真机训练闭环通过；
- 网络中断、进程退出、状态 stale、奇异和越界均安全停止；
- 最终镜像从零创建的新容器通过完整回归。

## 28. 相关文档

- `调试日志/0805调试日志.md`：Actor 环境、JAX、Agentlace；
- `调试日志/0806调试日志.md`：ROS、WA2 状态和 MoveL；
- `调试日志/0807调试日志.md`：ServoL、SpaceMouse XYZ；
- `调试日志/0810调试日志.md`：六轴、旋转、灵巧手和卡顿分析；
- `docs/SpaceMouse使用指南.md`：当前六轴真机启动与参数；
- `docs/solution/SpaceMouse设计方案.md`：SpaceMouse 设计细节；
- `docker/actor-requirements.txt`：Actor 依赖基线；
- `docker/verify_hil_actor.py`：环境/GPU 验收；
- `docker/verify_agentlace.py`：Agentlace 本机通信验收。
