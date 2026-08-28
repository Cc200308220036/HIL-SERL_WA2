# HIL-SERL 论文与源码笔记

> 论文：Jianlan Luo, Charles Xu, Jeffrey Wu, Sergey Levine, *Precise and Dexterous Robotic Manipulation via Human-in-the-Loop Reinforcement Learning*（2024）  
> 源码：当前 `hil-serl-main` 仓库  
> 阅读目的：对齐论文框架、算法逻辑、关键代码和数据流，并给出 Franka 实验的完整使用方法。

## 0. 先澄清：本仓库不是 Franka 仿真项目

当前仓库提供的是**真实 Franka/Panda/FR3 机械臂实验栈**：Gym 环境通过 HTTP 请求连接 Flask 机器人服务器，服务器再通过 ROS 连接真实机械臂控制器；视觉来自 RealSense，相机/人类接管来自 SpaceMouse。

仓库中没有 MuJoCo、Isaac Sim、Gazebo、PyBullet 等仿真器，也没有可直接运行的仿真场景或仿真机器人模型。源码里的 `fake_env=True` 主要用于 Learner 创建 observation/action space，并不模拟动力学、图像、奖励或机械臂交互；它不能用于体验完整策略 rollout。并且当前 `FrankaEnv.__init__()` 在检查 `fake_env` 前会调用 `_update_currpos()`，仍可能访问机器人服务器。因此：

- 有真实 Franka、RealSense、SpaceMouse 和 ROS 控制栈：可按第 7 节体验完整 HIL-SERL。
- 没有硬件：可阅读代码、训练已有离线数据、检查网络/回放池逻辑，但不能用本仓库独立完成真实的采集—干预—控制闭环。
- 若目标必须是仿真复现，需要另行实现兼容 Gymnasium 接口的模拟环境，并替换 `TrainConfig.get_environment()`；这不是仓库现成能力。

## 1. 一句话理解 HIL-SERL

HIL-SERL 用少量成功示范启动一个基于 SAC/RLPD 的视觉离策略强化学习器，在真实机器人自行探索时允许人用 SpaceMouse 临时接管纠错；离线示范和在线经验各占训练 batch 的一半，成功与否由视觉分类器给出稀疏奖励，Actor 与 Learner 异步运行，从而在较短实机时间内学到比单纯模仿人更可靠、更快的策略。

它相对 SERL 的关键增量不是更换一个全新的 RL 公式，而是把**在线人类纠错数据**有效纳入 RLPD 训练闭环。论文消融显示：只靠 RL 从零探索失败；把离线示范从约 20 条扩大到 200 条但没有在线纠错，仍显著弱于 HIL-SERL。

## 2. 论文核心框架

### 2.1 问题建模

任务建模为 MDP：状态/观测 `s`、动作 `a`、转移、稀疏奖励 `r`、折扣因子 `γ`。观测不是纯机器人状态，而是：

- 一个或多个 RGB 相机图像；
- 末端位姿、速度、力/力矩、夹爪状态；
- 动态任务还可使用关节位置 `q` 和关节速度 `dq`。

策略通常输出高斯分布的连续动作，目标是最大化折扣累计奖励。任务奖励通常只有成功时为 1，其余为 0；成功由离线训练的视觉分类器判断。

### 2.2 三个系统主体

1. **Actor 进程**：以 10 Hz 读取环境观测、执行最新策略、接受 SpaceMouse 接管、生成 transition，并发给 Learner。
2. **Learner 进程**：保存回放数据，等量采样 demo 和在线数据，更新 Actor/Critic/温度参数，再周期性广播新策略。
3. **两个 Replay Buffer**：
   - demo buffer：初始装入约 20–30 条专家示范，并继续接收人类干预 transition；
   - RL buffer：保存所有在线 transition，包括策略动作和被人替换后的实际干预动作。

论文图 2 的“Replay Buffer 位于 Learner 内部”在源码中对应 `TrainerServer` 注册的两个 data store。两进程使用 AgentLace/ZeroMQ 异步传输，不要求采样和梯度更新锁步。

### 2.3 感知与策略网络

每路 `128×128×3` RGB 图像分别通过共享设计的 ImageNet 预训练 ResNet-10 特征提取器；图像 embedding 拼接后，再与展平的本体状态融合，送入 MLP：

- 连续 Actor：输出 tanh-squashed Gaussian 分布；
- 连续 Critic：双 Q ensemble，输入观测编码和连续动作；
- 温度模块：自动调整熵正则权重；
- 可选 Grasp Critic：对离散夹爪动作做 DQN，单臂输出 `open/close/stay` 三个 Q 值；双臂对应 3×3 种组合。

源码默认策略与 critic MLP 是 `256×256`，图像训练使用 padding 4 的随机裁剪。策略编码处对视觉 encoder 使用 `stop_gradient=True`；预训练 ResNet 配置也是 frozen，重点训练后续视觉压缩层和策略/价值网络。

### 2.4 RLPD/SAC 更新逻辑

源码把论文所称 RLPD 实现为“高 UTD 的 SAC + prior data 等比例混合”：

1. 在线池抽 `batch_size/2`；demo 池抽 `batch_size/2`；沿 batch 维拼接。
2. 计算 SAC Bellman target：目标 critic 对下一动作估值，双 Q 取较小值。
3. 最小化 critic 的均方 Bellman error。
4. Actor 最大化 `Q(s,a) - α log π(a|s)`。
5. 自适应更新温度 `α`，并通过 Polyak averaging 更新 target critic。
6. `cta_ratio` 控制 critic-to-actor 更新比：默认每轮先做 `cta_ratio-1` 次仅 critic 更新，再做一次 critic、actor、temperature 联合更新。
7. 每 `steps_per_update` 个 Learner step 向 Actor 广播参数。

对带夹爪任务，连续运动和离散夹爪拆开学习：Actor 产生 6D/12D 末端运动，Grasp Critic 用 Double-DQN 风格 target 选择 `argmax` 离散动作，最后拼成 7D/14D 发送给环境。额外的 `grasp_penalty` 抑制无意义的重复开合。

### 2.5 人类干预逻辑

Actor 先取得策略动作；`SpacemouseIntervention` 随后读取 SpaceMouse：

- 无有效输入：执行策略动作；
- 有平移/旋转或按钮输入：用专家动作替换策略动作，并在 `info["intervene_action"]` 返回实际执行动作。

Actor 构造 transition 前会用 `intervene_action` 覆盖原策略动作。因此存下的是**真实执行动作**，不是被否决的动作。普通在线 transition 写入 RL buffer；干预 transition 同时写入在线池和 intervention/demo 池。这样 demo 池持续吸收“当前策略会犯错的状态”上的专家纠正。

推荐的人机节奏是：早期让策略探索 20–30 步后，把机器人引回有效区域，并较频繁地帮助轨迹获得成功奖励；策略开始偶尔自主成功后显著减少干预，只对重复错误和罕见恢复场景进行短纠正。人类不是全程遥操作，否则策略得不到足够自主状态分布。

### 2.6 奖励、相对坐标和底层安全控制

**视觉奖励**：正负图像各自进入 replay buffer，训练 ResNet-10 + 两层分类头；源码每个 batch 正负各半，使用 sigmoid binary cross entropy。推理时不同任务使用约 `0.7–0.9` 的阈值，并可叠加本体状态条件减少假阳性。Egg Flip 使用三分类，判断物体是否从初始朝向翻到另一朝向。

**相对/自我中心表示**：每回合 reset 位姿作为参考系，TCP pose 表示为相对初始位姿；速度转换到当前末端坐标系；策略输出的 twist 也从当前末端坐标系转换回机器人基坐标系。这使策略不易记住绝对位置，并提高对初始位置扰动和中途外力扰动的适应能力。

**控制双时间尺度**：上层策略/环境 10 Hz；接触任务的笛卡尔阻抗控制器 1 kHz。参考位姿增量和工作空间边界受限，避免随机探索产生过大接触力。Egg Flip 等动态任务不用位姿阻抗，而是 10 Hz 给出末端前馈 wrench，1 kHz 控制器通过 `Jᵀw` 转为关节力矩；该模式风险明显更高。

## 3. 论文模块与源码位置对照

| 论文模块 | 源码实现 | 关键职责 |
|---|---|---|
| 总训练入口 | `examples/train_rlpd.py` | Actor/Learner 主循环、网络同步、双池采样、checkpoint |
| 实验注册 | `examples/experiments/mappings.py` | 将 `exp_name` 映射到具体 `TrainConfig` |
| 通用训练参数 | `examples/experiments/config.py` | batch、discount、UTD、容量、同步周期等默认值 |
| 任务配置 | `examples/experiments/<task>/config.py` | 相机、位姿、安全边界、wrapper 顺序、模型类型、奖励阈值 |
| 任务 reset | `examples/experiments/<task>/wrapper.py` | RAM 重抓、USB 多阶段 reset、双臂 reset、Egg Flip 动作裁剪 |
| SAC/RLPD | `serl_launcher/.../agents/continuous/sac.py` | Actor、双 Q、温度、target update |
| 混合动作 | `sac_hybrid_single.py`、`sac_hybrid_dual.py` | 连续运动 SAC + 离散夹爪 DQN |
| 网络组装 | `serl_launcher/.../utils/launcher.py` | 网络宽度、预训练 encoder、随机裁剪、AgentLace 端口 |
| Actor/Critic 网络 | `serl_launcher/.../networks/actor_critic_nets.py` | Policy、Critic、GraspCritic |
| 视觉骨干 | `serl_launcher/.../vision/resnet_v1.py` | ResNet-10 与空间特征压缩 |
| 奖励分类器 | `networks/reward_classifier.py`、`examples/train_reward_classifier.py` | 分类器创建、训练、恢复和推理 |
| 数据采集 | `record_success_fail.py`、`record_demos.py` | 奖励样本和成功示范录制 |
| 回放池 | `serl_launcher/.../data/memory_efficient_replay_buffer.py` | 图像相邻帧打包、低内存存储、batch iterator |
| 观测整理 | `SERLObsWrapper`、`ChunkingWrapper` | 展平 proprio、图像移到顶层、增加时间维 |
| 人类接管 | `serl_robot_infra/.../envs/wrappers.py` | 单/双臂 SpaceMouse 动作替换和 intervention 标记 |
| 坐标变换 | `serl_robot_infra/.../envs/relative_env.py` | 相对 TCP 状态、body↔base 动作变换 |
| Gym 实机环境 | `serl_robot_infra/.../envs/franka_env.py` | 10 Hz step/reset、相机、HTTP 请求、安全 box |
| 双臂环境 | `dual_franka_env.py` | 两臂并行 step/reset、合并状态与图像 |
| 机器人服务 | `serl_robot_infra/robot_servers/franka_server.py` | Flask REST ↔ ROS topic/service |
| 动态控制 | `franka_wrench_env.py`、`franka_eggflip_server.py`、`egg_flip_controller/` | 前馈 wrench 控制链 |

## 4. 模块间数据传输

### 4.1 单个环境 step 的数据形态

基础 `FrankaEnv` 返回：

```text
observation = {
  "state": {
    "tcp_pose":     (7,) = xyz + quaternion,
    "tcp_vel":      (6,),
    "gripper_pose": (1,),
    "tcp_force":    (3,),
    "tcp_torque":   (3,)
  },
  "images": {camera_key: (128, 128, 3), uint8}
}
action = (7,) = Δxyz(3) + Δrotvec(3) + gripper(1), normalized to [-1,1]
```

依次经过 `RelativeFrame → Quat2Euler → SERLObsWrapper → ChunkingWrapper(obs_horizon=1)` 后，交给网络的是：

```text
obs = {
  "state": (1, D),
  camera_key_1: (1, 128, 128, 3),
  camera_key_2: (1, 128, 128, 3), ...
}
```

`1` 是 observation history 维，不是 batch 维。RAM 固定夹爪后 action 为 6D；USB 是 7D；双臂为 14D；Egg Flip 经 wrapper 只保留 `x、z、绕 y` 三个 wrench 分量。

每条 RL transition 为：

```text
{
  observations,
  actions,
  next_observations,
  rewards,       # 0/1；带夹爪任务的 penalty 单独保存
  masks,         # 1 - done，用于 bootstrap
  dones,
  grasp_penalty  # 可选
}
```

### 4.2 端到端数据流

```text
RealSense ──RGB──┐
Robot server ──pose/vel/force/gripper──> Gym env → wrappers → obs s_t
SpaceMouse ──expert action──────────────┘                 │
                                                         ▼
                                  Actor policy π(s_t) → proposed a_t
                                                         │
                    no intervention ─────────────────────┤
                    intervention → replace with a_human ─┘
                                                         │ HTTP pose/gripper/wrench
                                                         ▼
                                              Flask → ROS → controller → Franka
                                                         │
                                   next obs + classifier reward + done
                                                         ▼
                     transition → AgentLace queued data store → Learner RL buffer
                              └─ if intervention ───────→ Learner demo buffer

Learner: 128 online + 128 demo（默认 batch=256）→ SAC/DQN gradients
       → every steps_per_update → broadcast params → Actor callback replaces params
```

机器人侧 HTTP 内容主要是 JSON：环境向 `/pose` 发送 `{ "arr": [x,y,z,qx,qy,qz,qw] }`，向 `/wrench` 发送 wrench；`/getstate` 返回 pose、vel、force、torque、Jacobian、`q/dq`、gripper position。Actor/Learner 侧传输的是序列化 transition、统计信息和 JAX 参数树。

### 4.3 四个开源示例的配置差异

| `exp_name` | 硬件/动作 | 策略图像 | 奖励图像 | 特点 |
|---|---|---|---|---|
| `ram_insertion` | 单臂，6D twist，夹爪固定闭合 | 两个 wrist | 两个 wrist | 最完整教程；随机 reset；可按 F1 触发重抓 |
| `usb_pickup_insertion` | 单臂，6D twist + 离散夹爪 | side_policy + 两个 wrist | side_classifier | 学习抓取和插入；夹爪 penalty；脚本 reset |
| `object_handover` | 双臂，12D twist + 两个离散夹爪 | 左/右 wrist + 左 side | 左 side_classifier | 两个 Franka server；源码存在，官方 walkthrough 尚未补全 |
| `egg_flip` | 单臂，3D 前馈 wrench | wrist + side | wrist（三分类） | 动态高风险任务；官方 walkthrough 尚未补全 |

论文评测还包括 SSD、线缆卡扣、IKEA、仪表板、同步带、Jenga 等任务，但当前公开仓库没有这些任务的完整实验目录，不能仅靠现有配置直接复现全部论文实验。

## 5. 一次完整训练按什么顺序发生

1. 设计工位、相机视角、crop、动作尺度、绝对安全边界和 reset 轨迹。
2. 遥操作收集成功/失败图像，失败样本应覆盖“看起来很像成功”的困难负例。
3. 训练奖励分类器，先重点消除假阳性；假阳性会让 RL 学会欺骗分类器。
4. 用 SpaceMouse 收集约 20 条成功示范，期间也检验 classifier 和 reset。
5. Learner 将示范装入 demo buffer，创建 SAC/混合动作 agent，并等待在线池达到 `training_starts=100`。
6. Actor rollout 并上传 transition；Learner 从两池各采一半持续更新。
7. 操作者在危险、无效或局部最优行为出现时短暂接管；干预数据进入两个池。
8. Learner 定期广播参数，Actor 无需重启即可使用新策略。
9. 成功率提高、周期时间和干预率下降后，保存 checkpoint 并固定 checkpoint 做多回合评估。

## 6. 安装和硬件前提

### 6.1 硬件与系统

- Franka Panda 或 FR3，已配置网络和 Franka Desk/FCI；双臂任务需两套独立服务。
- 合适的 Franka/Robotiq gripper；RAM 示例是预抓取、固定夹爪策略。
- Intel RealSense，相机序列号需要写入任务 config。
- 3Dconnexion SpaceMouse，负责 demo 和在线干预。
- 带 NVIDIA GPU 的 Learner 主机更实际；Actor 与 Learner 可同机或分机。
- ROS、`libfranka`、`franka_ros`，以及仓库外部依赖 `serl_franka_controllers`。

这套系统会让 RL 在真实机械臂上探索，必须有实体急停、清空工作空间、合理负载标定和人工全程监护。不要把源码中的示例 IP、位姿、安全边界或增益原样用于另一台机器人。

### 6.2 Python 环境

以下命令从仓库根目录开始。JAX 安装命令应按本机 CUDA/驱动匹配；这里保留仓库 README 的 Python 3.10 与 JAX 0.4.35 组合：

```bash
conda create -n hilserl python=3.10
conda activate hilserl

# GPU 示例；CPU 可安装 jax[cpu]，但不适合实机在线视觉训练
pip install --upgrade "jax[cuda12_pip]==0.4.35" \
  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

cd serl_launcher
pip install -e .
pip install -r requirements.txt

cd ../serl_robot_infra
pip install -e .
cd ..
```

随后按 `serl_robot_infra/README.md` 安装与当前 Franka 固件/ROS 发行版匹配的 `libfranka`、`franka_ros` 和 `serl_franka_controllers`。仓库自带 launch shell 含作者机器的绝对路径、IP 和关节值，必须编辑后才能使用。

## 7. 推荐实操：RAM insertion 全流程

RAM 是最适合首次体验的示例，因为仓库文档、reset、分类器和训练脚本最完整。

### 7.1 上电前安全检查

1. 固定主板、RAM holder、腕部相机和所有线缆，确认机器人全范围内没有人和障碍物。
2. 在 Franka Desk 标定末端工具和腕部相机的质量、质心、惯量。
3. 设置实体急停，并安排一人始终观察机器人。
4. 先用低风险手动动作确认工作空间边界，再允许随机策略运行。
5. 解锁机器人并启用 FCI；FR3 还需进入 execution mode。

### 7.2 配置机器人服务器

编辑 `serl_robot_infra/robot_servers/launch_right_server.sh`：

- catkin workspace 的 `devel/setup.bash`；
- `robot_ip`；
- `gripper_type` 与可选 `gripper_ip`；
- `reset_joint_target`；
- Flask bind 地址与 ROS port。

启动并只做只读/低风险连通检查：

```bash
cd serl_robot_infra/robot_servers
bash launch_right_server.sh

# 另一终端；把 URL 换成实际地址
curl -X POST http://127.0.0.2:5000/getstate
curl -X POST http://127.0.0.2:5000/getpos_euler
```

确认状态正常后，轻推末端检查阻抗顺应性。`stopimp/startimp`、`jointreset` 和位姿命令会改变机器人状态，不应作为随意的连通测试。

### 7.3 修改任务配置

编辑 `examples/experiments/ram_insertion/config.py`：

- `SERVER_URL`：必须以 `/` 结尾；
- `REALSENSE_CAMERAS`：改成实际序列号；
- `IMAGE_CROP`：让图像集中在 RAM、插槽和接触区域；
- `TARGET_POSE`：RAM 完全插入时的末端位姿；
- `GRASP_POSE`：从 holder 抓 RAM 的位姿；
- `RESET_POSE`：每回合开始位姿；
- `ABS_POSE_LIMIT_LOW/HIGH`：真实安全 box；
- `ACTION_SCALE`、控制器参数和随机化范围：先保守验证。

将机械臂手动引导到各目标位姿时，用下面的命令读取 `[x,y,z,roll,pitch,yaw]`，再人工填写配置：

```bash
curl -X POST http://<FRANKA_SERVER>:5000/getpos_euler
```

### 7.4 收集并训练奖励分类器

从任务目录执行，确保相对路径和输出目录正确：

```bash
cd examples/experiments/ram_insertion
python ../../record_success_fail.py --exp_name=ram_insertion --successes_needed=200
```

脚本默认记录失败帧；按住空格时记录成功帧。建议失败样本至少是成功样本的 2–3 倍，并包含：错误插槽、仅插入一侧、半插入、靠近但未插入、手持悬空、遮挡和各 reset 随机位置。

```bash
python ../../train_reward_classifier.py --exp_name=ram_insertion
```

数据保存在 `classifier_data/`，checkpoint 保存在 `classifier_ckpt/`。训练准确率不能代替实机验证：重新运行采集/演示界面，逐个测试最危险的困难负例。RAM 配置默认还要求 classifier 概率大于 `0.85` 且状态条件成立。

### 7.5 收集 20 条示范

```bash
python ../../record_demos.py --exp_name=ram_insertion --successes_needed=20
```

策略动作此时为零，SpaceMouse 输入通过 intervention wrapper 实际控制机器人。每个成功 episode 的 transition 会保存到 `demo_data/`。如果出现 classifier 假阳性/假阴性，暂停示范，补采对应分类数据并重训；不要把错误成功轨迹放进 demo buffer。

### 7.6 配置异步训练

编辑任务目录中的 `run_actor.sh` 和 `run_learner.sh`：

- 两者 `checkpoint_path` 指向本次 run；
- Learner 的 `--demo_path` 指向刚生成的 `.pkl`，有多个文件可重复传 flag；
- Actor 与 Learner 分机时，Actor 增加 `--ip=<LEARNER_IP>`，并开放 TCP 5588/5589；
- `XLA_PYTHON_CLIENT_MEM_FRACTION` 按 GPU 调整。

也可不用 shell 占位符，直接运行：

```bash
# 终端 A：Learner；先启动，它会等待在线池达到 100 条
cd examples/experiments/ram_insertion
python ../../train_rlpd.py \
  --exp_name=ram_insertion \
  --checkpoint_path=first_run \
  --demo_path=demo_data/<实际示范文件>.pkl \
  --learner

# 终端 B：Actor
cd examples/experiments/ram_insertion
python ../../train_rlpd.py \
  --exp_name=ram_insertion \
  --checkpoint_path=first_run \
  --actor \
  --ip=localhost
```

Actor 每 1000 step 把在线和干预 transition 分块保存；Learner 每 5000 step 保存 checkpoint。RAM 默认 `batch_size=256`、`discount=0.97`、`cta_ratio=2`、同步周期 50 Learner step、最大 episode 100 step、环境 10 Hz。

### 7.7 干预和收敛判断

- 初期允许短暂随机探索，再把 RAM 引向插槽附近，并适度帮助完成任务，使奖励能向前传播。
- 对越界趋势、重复远离目标、危险碰撞立即接管；安全优先于采样纯度。
- 策略能偶尔自主成功后减少干预；否则人类动作会掩盖真实策略水平。
- 用 F1 触发 RAM 重抓流程，增加抓取初态多样性。
- 关注成功率、episode return、cycle time、`intervention_count` 和 `intervention_steps`。目标不仅是训练轨迹成功，还应是 intervention rate 接近 0。

论文报告 RAM 在相应工位和随机化下约 1.5 小时达到 100%，但这不是新硬件、不同相机或不同装配公差下的保证。

### 7.8 评估固定 checkpoint

先停止 Learner，确保评估期间参数不再变化；Actor 加入：

```bash
python ../../train_rlpd.py \
  --exp_name=ram_insertion \
  --checkpoint_path=first_run \
  --actor \
  --eval_checkpoint_step=<checkpoint编号> \
  --eval_n_trajs=20 \
  --save_video
```

评估应覆盖训练随机化范围、困难初态和可控外部扰动，分别记录成功率、成功回合耗时、失败模式和是否需要人工接管。源码评估仍随机采样策略动作（`argmax=False`），因此应做足够多回合，而不是只展示一次成功。

## 8. 其他三个示例如何迁移

### USB pickup + insertion

沿用 RAM 的“配置—分类器—20 demos—双进程训练—评估”顺序，但策略学习夹爪动作。需要设置一个 side camera 的两个逻辑 crop（policy/classifier），避免同一物理相机重复初始化；reset 会先取回 USB，再随机放置。默认 episode 120 step、`discount=0.98`、classifier 阈值 0.7，并对不必要开合施加 `-0.02` penalty。官方经验约 2.5 小时，但同样仅作参考。

### Object handover

需要同时启动左右两个 Franka server，使用不同 Flask 地址、ROS master port，必要时使用各自兼容固件的 catkin workspace。`DualFrankaEnv` 用两个线程并行 step/reset，合并为带 `left/`、`right/` 前缀的观测；动作 14D。双臂同时运动会放大碰撞风险，必须重新设计两臂各自和相互之间的安全约束。仓库只有源码配置，walkthrough 明确标注说明尚未完成，应先做代码级和低速硬件验证。

### Egg Flip

需要把 `egg_flip_controller` 复制到 catkin workspace 并编译，启动 `franka_eggflip_server.py`。策略输出前馈 wrench，而非安全性更高的位姿增量。仓库 README 明确警告该控制器只是任务专用参考实现，启动时机器人可能立即移动到 reset pose；周围上下前后必须完全清空。没有控制/动力学经验时不要运行。该任务应在完成静态阻抗任务、独立审查 controller 增益/限幅/急停后再尝试。

## 9. 无硬件时能做的验证

1. 阅读论文和本笔记，追踪 `TrainConfig → wrappers → train_rlpd.actor/learner → SAC.update`。
2. 对源码做语法检查：`python -m compileall examples serl_launcher serl_robot_infra`；这不验证 ROS、相机、网络或运行时依赖。
3. 若已获得兼容的 `classifier_data`，可离线训练 reward classifier；环境构造仍可能访问 robot server，必要时需先修正/Mock 硬件初始化。
4. 若已有 demo/online pickle，可研究 replay、batch 和 learner 更新；当前入口仍根据环境空间构造 agent，需提供兼容的 mock env。
5. 若要真正仿真，应实现与最终 wrapper 前后契约一致的环境：相同 observation keys/shapes、normalized action、10 Hz step、reset、binary reward 和 `info["intervene_action"]`，再在 `CONFIG_MAPPING` 注册新的 config。仅把 `fake_env=True` 当仿真使用会得到错误结论。

## 10. 常见故障与排查顺序

| 症状 | 优先检查 |
|---|---|
| 环境初始化卡住/Connection refused | `SERVER_URL`、末尾 `/`、Flask bind IP、端口、防火墙、server 是否已启动 |
| Learner 一直等待 buffer | Actor 是否连到正确 Learner IP；5588/5589 是否通；Actor 是否在 step |
| 相机冻结 | USB 带宽/供电、RealSense serial、是否重复打开同一相机、crop 是否越界 |
| 一开始 episode 就结束 | classifier 假阳性、阈值太低、state gate 索引/量纲错误 |
| 真成功却一直不给奖励 | crop 不含关键区域、光照变化、困难正例不足、阈值过高 |
| 机器人总撞安全 box | reset/target/action scale/坐标变换或 ABS limits 不适配当前工位 |
| 夹爪反复开合 | gripper 状态方向、开合阈值、sleep、grasp penalty、示范动作标签 |
| 恢复训练报错 | Actor/Learner checkpoint path、buffer 文件名、checkpoint 与 config/shape 是否一致 |
| GPU OOM | 降低 XLA 内存比例、batch 或相机路数；确认 Actor/Learner GPU 分配 |
| 策略“骗过”奖励分类器 | 立即停训，补充该失败模式的强负例，重训 classifier 后再继续 |

## 11. 阅读源码时值得注意的实现细节

- `train_rlpd.py` 的 `config = CONFIG_MAPPING[FLAGS.exp_name]()` 出现在合法性 assert 之前，错误的 `exp_name` 会先触发字典异常。
- 任务 shell 中的 `checkpoint_path=first_run` 和 `demo_path=...` 只是占位值；USB shell 还保留 debug 路径，不能不检查就运行。
- 当前目录不是完整 Git worktree（没有可用的仓库元数据），实验前应自行建立版本记录，保存每次 config、数据和 checkpoint 的对应关系。
- 论文奖励分类器描述为 Adam `3e-4`、100 iterations；当前源码 `create_classifier()` 使用 Adam `1e-4`，训练脚本默认 150 epochs。复现实验时应区分“论文参数”和“开源实现默认值”。
- 论文称干预片段同时写入 demo/RL 两池；源码确实如此，但保存的是 wrapper 替换后的实际动作。分析日志时不要把这些 transition 当作自主策略成功。
- `MemoryEfficientReplayBuffer` 会把相邻图像帧打包以减少传输/存储；agent 更新时 `_unpack()` 恢复 `obs/next_obs`，本体状态则正常分别保存。
- 官方 walkthrough 只完整覆盖 RAM 和 USB；Object Handover 与 Egg Flip 明确写着说明待补充。因此后二者应视为研究代码，而不是开箱即用教程。

## 12. 总结

HIL-SERL 的效果来自系统组合而非单一技巧：预训练视觉特征稳定图像 RL，RLPD 的 50/50 prior-online sampling 提高样本效率，在线人类纠错把探索导向可恢复和高价值区域，视觉稀疏奖励避免手工 shaping，相对坐标增强空间泛化，安全阻抗控制使真实机器人探索可行，异步 Actor/Learner 则让采样和 GPU 更新并行。

若要在当前仓库真正体验算法全流程，最佳路线是从单臂 RAM insertion 开始，先把机器人服务、相机、奖励分类器、reset 和安全 box 分别验证，再收集 20 条示范并启动 Actor/Learner；不能把 `fake_env` 当作现成 Franka 仿真器。完整复现的主要工程难点通常不在 SAC 公式，而在可靠奖励、任务 reset、安全控制、相机布局和干预质量。
