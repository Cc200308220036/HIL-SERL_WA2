# HIL-SERL-WA2 项目工程代码梳理

> 文档日期：2026-08-26  
> 工程根目录：`/home/cyw/orin_hilserl`  
> Actor 运行根：Orin 容器 `/root/catkin_ws`，Conda 环境 `hil-actor`  
> Learner 运行根：笔记本 `/home/cyw/orin_hilserl/HILSERL_Learner`，Conda 环境 `hil-learner`  
> 分析依据：HIL-SERL 论文、`hil-serl-main` 论文源码、WA2 R1～R13 代码、方案文档与阶段验收日志。  
> 本文只标注文件属性和清理建议，不执行删除、移动、归档或重命名。

---

## 1. 文档目的和分类口径

本工程是在 HIL-SERL 论文实现基础上，将原 Franka 机器人环境替换为 WA2 的 ROS1、ServoL、相机、灵巧手和 SpaceMouse 接口，并拆分为 Orin Actor 与笔记本 Learner 两个运行端。

交付前不能简单按“最近有没有运行过”判断文件是否可删。很多底层模块不会被命令行直接启动，但会通过 `env_factory`、Agent、Wrapper 或动态 import 间接加载。本文采用以下分类：

| 标记 | 含义 | 交付建议 |
|---|---|---|
| **P0 核心** | 正式训练、评测或安全控制的必需代码 | 必须保留并纳入版本和哈希管理 |
| **P1 运维** | 建 manifest、采 demo、训练分类器、恢复缓存、环境激活等 | 建议保留在交付开发包；可不放最小运行镜像 |
| **T 阶段测试** | R1～R12 的 Gate、dry-run、dummy server、历史验证入口 | 正式验收后可移出运行包，建议归档而非直接销毁 |
| **A 训练资产** | demo、分类器、策略 checkpoint、Buffer 快照、manifest | 不能按普通日志删除；应按“基线/历史”分级保留 |
| **G 生成物** | `__pycache__`、`.pyc`、临时日志、可重建 build 结果 | 通常可删除 |
| **C 条件删除** | 上游未用算法、旧脚本、旧数据、镜像 tar 等 | 满足本文列出的前置条件后才可删除 |

“可删除”在本文中表示可从最终运行包或当前工作目录清理，不等同于可以在没有备份和回归测试的情况下立即执行删除。

---

## 2. 论文架构与 WA2 工程对应关系

### 2.1 论文主闭环

HIL-SERL 的核心不是单独一个 SAC 脚本，而是以下异步闭环：

```text
                     Learner（笔记本 GPU）
  初始成功 Demo ───────> Demo Buffer ───────┐
  在线人工干预 ─────────> Demo Buffer        │ 50%
                                            ├─ 合并 batch
  Actor 所有 transition ─> RL Buffer ───────┘ 50%
                              │
                              ▼
                 Continuous SAC/RLPD 更新
                 Discrete Grasp Critic 更新
                              │
                       发布最新网络参数
                              ▼
                     Actor（WA2 Orin）
  相机/状态 → 策略动作 → 人工可覆盖 → WA2 → 奖励分类器
                              │
                  实际执行 transition 上传
```

论文与当前代码的关键映射如下：

| 论文模块 | WA2 实现 |
|---|---|
| Actor process | `catkin_ws/src/hilserl_wa2/scripts/r13_actor_train.py` |
| Learner process | `HILSERL_Learner/src/hilserl_wa2/scripts/r13_learner_train.py` |
| 连续 SAC/RLPD | `serl_launcher/agents/continuous/sac_hybrid_single.py` |
| 离散 Grasp Critic | 同上，`grasp_critic_loss_fn` |
| RL Buffer | Learner 的 `MemoryEfficientReplayBufferDataStore`，注册名 `actor_env` |
| Demo Buffer | 同类 Buffer，注册名 `actor_env_intvn`；启动时先载入初始 demo |
| 50/50 抽样 | Learner 分别抽 `batch_size/2` 后 `concat_batches` |
| 人工干预覆盖 | `WA2SpacemouseIntervention` + `transition.executed_action` |
| 稀疏奖励 | Actor 侧 `WA2RewardClassifierWrapper` |
| learned gripper | `WA2GraspActionWrapper` + Hybrid SAC Grasp Critic |
| 参数广播 | Agentlace `TrainerServer.publish_network` → Actor callback |

### 2.2 WA2 动作不是 7D 全连续动作

Replay Buffer 统一保存 7D 实际执行动作：

```text
[dx, dy, dz, dRx, dRy, dRz, grasp]
 └────── 6D Continuous SAC ──────┘  └ Discrete DQN ┘
```

- Continuous Actor 输出 6D。
- Continuous Critic 只读取 `actions[..., :-1]`。
- Grasp Critic 只读取状态，输出 `Q(s,-1)、Q(s,0)、Q(s,+1)`。
- WA2 定义 `+1=grasp、0=hold、-1=release`。
- 推理时才把 6D 连续动作和 1D 离散动作拼成 7D。
- Learner 显式传入 `target_entropy=-3.0`，对应 6 个连续维度的当前工程规则。

当前 2026-08-25 以后代码还包含两项工程修正：

- `grasp_penalty` 默认已从早期 `-0.02` 调整为可配置的 `-0.002`，原因是稀疏奖励下 `-0.02` 曾推动 Grasp Critic 塌缩为长期 hold。
- HIL 训练默认 `grasp_eps=0.15`，只在离散开/合中进行 epsilon 探索；脱手评测固定 `grasp_eps=0`。

因此，早期方案、聊天记录或日志中出现的 `-0.02` 不是当前默认运行值。正式交付时应把最终值冻结在配置或启动脚本中，避免口径再次漂移。

### 2.3 transition 分流语义

正式 transition 核心字段为：

```python
{
    "observations": s_t,
    "actions": executed_action,
    "next_observations": s_t1,
    "rewards": sparse_reward,
    "masks": 0.0_or_1.0,
    "dones": terminated,
}
```

Learner 入 Buffer 时再统一补充 `grasp_penalty`。分流规则：

| 数据来源 | RL Buffer | Demo Buffer |
|---|---:|---:|
| 普通策略 transition | 写入 | 不写 |
| 人工干预 transition | 写入 | 同时写入 |
| 初始成功 demo | 不经 Actor 在线上传 | Learner 启动时载入 |

必须保存实际执行动作，而不是干预前策略原动作。该语义由 `experiments/transition.py` 统一负责，不能在交付清理时用自行拼字典替代。

---

## 3. 工程总目录与边界

```text
/home/cyw/orin_hilserl/
├── catkin_ws/                 # Actor 工作区；映射到容器 /root/catkin_ws
│   ├── src/hilserl_wa2/      # WA2 Actor 和共享语义代码
│   ├── src/hil-serl-main/    # HIL-SERL 上游算法源码副本
│   ├── src/naviai_controller/# WA2 机器人控制 Python/ROS 接口
│   ├── src/joystick_drivers/ # spacenav_node/SpaceMouse ROS 驱动
│   └── runs/                 # Actor demo、分类器、日志等运行资产
├── HILSERL_Learner/           # 笔记本 Learner 独立工作根
│   ├── src/hilserl_wa2/      # Learner 与共享语义代码副本
│   ├── src/hil-serl-main/    # Learner 使用的算法源码副本
│   ├── artifacts/            # ResNet10、Agentlace wheel
│   └── runs/                 # checkpoint、Buffer、demo、metrics
├── docker/                    # Actor 容器固化草案和依赖验收
├── artifacts/                # 离线 wheel 等公共交付资产
├── configs/                  # 早期/仓库级配置镜像
├── docs/                     # 接口、方案、本文档
├── 调试日志/                 # 开发过程和阶段验收证据
└── dustynv-jax-*.tar.gz      # 大型基础镜像/安装缓存
```

边界必须保持清晰：

- Actor 不负责梯度更新，不运行 `r13_learner_train.py`。
- Learner 不连接 ROS 和真实机器人，不运行 Actor、ServoL、reset 或 demo 采集脚本。
- `hilserl_wa2/configs`、`envs`、`experiments`、`wrappers`、`serl_launcher` 和 `examples/experiments/wa2` 属于两端握手哈希范围，修改后必须同步两端并重建 manifest。
- `interventions` 和两端专属 `scripts` 当前不全部进入 source tree hash，但仍可能是安全关键代码，不能因为“不在 hash 中”就删除。

---

## 4. Actor 端代码梳理

### 4.1 Actor 正式入口

| 文件 | 等级 | 作用 | 清理结论 |
|---|---|---|---|
| `scripts/r13_actor_train.py` | P0 | 正式 HIL 训练 Actor；握手、采样、人工覆盖、分类器、上传、fail-closed | 必须保留 |
| `scripts/r13_actor_eval.py` | P0 | 冻结 checkpoint 的脱手评测；关闭干预和夹爪 epsilon 探索 | 必须保留 |
| `scripts/record_r13_demos.py` | P1 | 采集原生 7D 成功 demo | 建议保留 |
| `scripts/start_spacemouse_joy.sh` | P0/P1 | 启动 `spacenavd` 与 ROS `spacenav_node` | 当前硬件路线必须保留 |
| `scripts/build_r13_manifest.py` | P1 | 生成/比较两端 R13 manifest | 必须随交付开发包保留 |
| `scripts/split_r13_demo_pkl.py` | P1 | 将大 demo 拆成逐 episode pickle，避免 Learner 内存峰值 | 建议保留 |
| `scripts/verify_r13_demo_load.py` | P1 | 训练前验证 7D demo 与 Buffer 可加载性 | 建议保留 |

### 4.2 Actor 主调用链

```text
r13_actor_train.py
  ├─ load_task / task YAML
  ├─ make_wa2_environment
  │   ├─ WA2Env
  │   │   ├─ WA2StateMonitor
  │   │   ├─ WA2ImageMonitor / WA2Cameras
  │   │   ├─ WA2ServoSession
  │   │   └─ WA2ResetExecutor / naviai_controller
  │   ├─ SERLObsWrapper / ChunkingWrapper
  │   ├─ WA2SpacemouseIntervention
  │   ├─ WA2GraspActionWrapper
  │   └─ WA2RewardClassifierWrapper
  ├─ make_sac_pixel_agent_hybrid_single_arm
  ├─ Agentlace handshake / network callback
  ├─ sample 6D SAC + 1D Grasp action
  ├─ build_actor_transition / route_transition
  ├─ actor_env + actor_env_intvn 本地队列
  └─ actor_upload_queue 异步上传
```

### 4.3 Actor P0 模块

#### `envs/`

| 文件 | 作用 | 结论 |
|---|---|---|
| `envs/wa2_env.py` | Gymnasium Env；fake/read-only/live 三模式；组合状态、相机、ServoL、reset、hand | P0 |
| `envs/contracts.py` | WA2 observation/action/安全阈值契约 | P0 |
| `envs/scene_config.py` | 场景 reset 位姿、工作空间和容差 | P0 |

#### `ros_adapters/`

| 文件 | 作用 | 结论 |
|---|---|---|
| `state_monitor.py` | 订阅 TCP、速度、关节、手状态和控制器状态 | P0 |
| `image_monitor.py` | 双相机缓存、帧龄/FPS/同步状态 | P0 |
| `image_preprocess.py` | ROS 图像解码、裁剪、resize 到策略输入 | P0 |
| `wa2_cameras.py` | 相机接口封装 | P0 |
| `servo_session.py` | 唯一 ServoL 执行路径；50 Hz、闩锁、奇异/陈旧/故障 stop+clear | P0 安全关键 |
| `reset_executor.py` | 安全停止 ServoL、MoveJ、手爪和场景 reset | P0 安全关键 |
| `mock_robot.py`、`mock_cameras.py` | fake_env 和无硬件单测 | P1；建议保留测试包，可不进最小 live 镜像 |

#### `interventions/`

| 文件 | 作用 | 结论 |
|---|---|---|
| `wa2_spacemouse_intervention.py` | 将 `/spacenav/joy` 转成 `intervene_action`；干预会话优先 | P0 |
| `spacemouse_input.py` | 死区、轴选择、归一化和 motion intent | P0 |
| `spacemouse_config.py` | SpaceMouse YAML 校验 | P0 |
| `joy_watchdog.py` | Joy 新鲜度和断连保护 | P0 安全关键 |
| `pose_integrator.py` | 时间积分/动作平滑 | P0 |
| `actor_upload_queue.py` | 异步上传、游标对齐、背压、gap、确认 | P0 数据可靠性关键 |
| `end_effector.py` | 灵巧手适配抽象 | P0/P1；当前 hand 路径使用 |
| `spacemouse_wa2_teleop.py` | 独立诊断 teleop，不是 R13 Actor 主控制入口 | T/P1；可移出最小运行包，保留维护包 |

#### `wrappers/`

| 文件 | 作用 | 结论 |
|---|---|---|
| `grasp_action.py` | 6D→7D agent space；离散化；边沿触发 `request_hand`；记录实际夹爪动作 | P0 |
| `reward_classifier.py` | 加载 R12 classifier，产生稀疏 reward 和成功 terminated | P0 |

#### `experiments/`

| 文件 | 作用 | 结论 |
|---|---|---|
| `env_factory.py` | 统一构造 Wrapper 栈和 space signature | P0 |
| `task_config.py` | task YAML 加载、校验、hash、space hash | P0 |
| `transition.py` | 实际动作、mask/done、离散夹爪校验、双写路由 | P0 |
| `r10_protocol.py` | 网络配置、source tree hash、session guard、manifest 基础 | P0，R13 仍复用 |
| `r13_protocol.py` | R13 握手、动作缩放、NaN 检查、server 状态 | P0 |
| `actor_safety.py` | 启动门禁、网络 watchdog、运动预算、fail-closed、参数签名 | P0 安全关键 |
| `demo_io.py` | R11/R13 demo 流式读取、分 episode、校验、写 bundle | P0/P1 |
| `classifier_io.py` | R12 数据集、阈值、bundle、指标 | P1；若交付支持分类器再训练则保留 |
| `demo_grasp.py` | 历史 R11 6D demo 离线增广为 7D | C；当前 R13 明确使用原生 7D demo 后可归档 |

### 4.4 Actor 外部依赖源码

| 目录 | 作用 | 结论 |
|---|---|---|
| `catkin_ws/src/naviai_controller` | `NaviController`、`ArmGroup`、`HandType`、ServoL/MoveJ/hand 接口 | P0，不能删除 |
| `catkin_ws/src/joystick_drivers/spacenav_node` | SpaceMouse ROS1 节点 | P0，当前输入方案不能删除 |
| `catkin_ws/src/joystick_drivers` 其他子包 | PS3/Wii 等未使用驱动 | C；可在重新构建只保留 `spacenav_node` 后裁剪 |
| `naviai_controller/scripts/test*.py` | 控制器厂商/早期 ServoL 测试 | T；正式封装后可移出运行包 |
| `naviai_controller/third_party/*.run` | 不同版本类型包安装器 | C；确定目标固件/类型版本并完成镜像固化后只留实际版本或外部归档 |

---

## 5. Learner 端代码梳理

### 5.1 Learner 正式入口和职责

| 文件 | 等级 | 作用 | 清理结论 |
|---|---|---|---|
| `scripts/r13_learner_train.py` | P0 | 双 Buffer、50/50 RLPD、Hybrid SAC 更新、参数广播、checkpoint/Buffer 恢复 | 必须保留 |
| `scripts/activate_hil_learner.sh` | P0/P1 | 激活 `hil-learner`，隔离 ROS2 overlay，设置 PYTHONPATH/JAX | 建议保留 |
| `scripts/build_r13_manifest.py` | P1 | Learner manifest 生成与两端比较 | 必须保留 |
| `scripts/verify_r13_demo_load.py` | P1 | 训练前验证 demo 和 7D Buffer | 建议保留 |
| `scripts/build_source_sha256s.py` | P1 | 固化源码校验和 | 交付建议保留 |
| `scripts/verify_hil_learner_dependencies.py` | P1 | 环境依赖验收 | 建议保留 |
| `scripts/verify_hil_learner_gpu.py` | P1 | JAX/CUDA 验收 | 建议保留 |

### 5.2 Learner 主调用链

```text
r13_learner_train.py
  ├─ fake WA2 Env 构造 observation/action space
  ├─ make_sac_pixel_agent_hybrid_single_arm
  │   ├─ Continuous Actor
  │   ├─ Double Continuous Critic
  │   ├─ Temperature α
  │   └─ Discrete Grasp Critic
  ├─ 载入 R13 episode demo → Demo Buffer
  ├─ 恢复 demo_buffer_cache / online_buffer_cache
  ├─ TrainerServer
  │   ├─ actor_env → RL Buffer
  │   └─ actor_env_intvn → Demo Buffer
  ├─ RL 半 batch + Demo 半 batch
  ├─ cta_ratio 更新
  ├─ temperature floor / resume kick
  ├─ publish_network(params)
  └─ checkpoint + metrics + Buffer snapshot
```

### 5.3 Learner 当前核心训练语义

- 默认 batch 256：128 Online/RL + 128 Demo/Intervention。
- 默认 `cta_ratio=2`：第一批只更新 `critic + grasp_critic`，第二批更新 `critic + grasp_critic + actor + temperature`。
- 连续 Critic 使用任务 `reward`；Grasp Critic 使用 `reward + grasp_penalty`。
- 当前默认 `grasp_penalty=-0.002`，命令行可覆盖。
- `target_entropy=-3.0`，对应 6D 连续 Actor。
- 当前增加 `min_temperature=0.01` 和 resume kick `0.05`，防止历史 checkpoint 的 α 下降到约 `1e-4` 后策略近乎确定。
- target network 使用 Polyak 软更新。
- 图像使用随机 crop；预训练 ResNet10 参数来自 `artifacts/models/resnet10_params.pkl`。
- Memory Efficient Replay Buffer 使用环形覆盖和均匀随机抽样，不是 prioritized replay。
- 旧 Buffer 快照加载后会按当前 penalty 规则重算 `grasp_penalty`。

### 5.4 Learner P0/P1 模块

Learner 的 `hilserl_wa2/envs`、`experiments`、`wrappers` 虽然不控制真机，但用于：

- 构造与 Actor 完全一致的 observation/action space；
- 计算 `space_hash`、`source_tree_sha256`；
- 校验 transition 和 demo；
- 载入 reward classifier 训练/评测工具。

因此不能只留下 `r13_learner_train.py`。至少必须保留：

```text
HILSERL_Learner/src/hilserl_wa2/
├── configs/
├── envs/
├── experiments/
├── wrappers/
├── ros_adapters/mock_robot.py
├── ros_adapters/mock_cameras.py
└── scripts/r13_learner_train.py
```

真实 ROS adapters 只存在于 Actor 端是合理的；Learner 不应引入 `rospy` 或机器人控制依赖。

### 5.5 Reward Classifier 生命周期文件

| 文件 | 用途 | 分类 |
|---|---|---|
| `train_r12_reward_classifier.py` | 训练奖励分类器 | P1；支持新任务/再训练时保留 |
| `eval_r12_splits.py` | 离线 split 指标 | P1 |
| `reselect_r12_threshold.py` | 不重训重选阈值 | P1 |
| `merge_r12_classifier_bundles.py` | 合并 success/fail bundle | P1 |
| Actor `record_r12_success_fail.py` | 采集正负样本 | P1；只交冻结模型时可移出运行包 |
| Actor `eval_r12_live.py` | 实机分类器回归 | P1；建议保留维护包 |

如果最终交付目标仅为“冻结 bottle_pick 模型运行”，以上脚本可不进入最小运行镜像，但分类器 checkpoint、阈值 JSON 和 Actor `reward_classifier.py` 仍为 P0。

---

## 6. `hil-serl-main` 论文源码梳理

### 6.1 当前 WA2 实际依赖的上游核心

| 上游目录/文件 | 当前用途 | 结论 |
|---|---|---|
| `agents/continuous/sac_hybrid_single.py` | 6D SAC + 1D Grasp Critic | P0 |
| `agents/continuous/sac.py` | Hybrid Agent 复用的 SAC 基础结构 | P0/间接依赖 |
| `agents/continuous/bc.py` | `utils/launcher.py` 顶层 import | 当前即使不训练 BC 也不能直接删，除非先重构 launcher |
| `agents/continuous/sac_hybrid_dual.py` | `utils/launcher.py` 顶层 import | 同上；不能未经重构直接删 |
| `data/replay_buffer.py` | 环形 Buffer | P0 Learner |
| `data/memory_efficient_replay_buffer.py` | 图像低冗余 Buffer | P0 Learner |
| `data/data_store.py` | Agentlace DataStore 包装与锁 | P0 Learner |
| `networks/*` | Actor/Critic/MLP/Lagrange/Classifier | P0 |
| `vision/resnet_v1.py`、`spatial.py`、`data_augmentations.py` | ResNet10、空间编码、随机裁剪 | P0 |
| `common/*` | TrainState、编码、优化器、类型 | P0 |
| `utils/launcher.py` | 构造 Hybrid Agent | P0 |
| `utils/train_utils.py` | 合并 batch 等训练工具 | P0 Learner |
| `wrappers/chunking.py`、`serl_obs_wrappers.py` | observation 包装和 frame stack | P0 两端 |
| `examples/experiments/config.py`、`mappings.py`、`wa2/` | WA2 TrainConfig 和 source hash | P0/握手依赖 |

### 6.2 论文参考入口但不是 WA2 正式入口

| 文件/目录 | 说明 | 清理建议 |
|---|---|---|
| `examples/train_rlpd.py` | 论文通用 Actor/Learner 示例；WA2 借鉴其结构但不能直接跑真机 | 建议源码参考保留；可不进最小运行包 |
| `examples/record_demos.py` | Franka 通用 demo 采集 | C |
| `examples/record_success_fail.py` | Franka 分类器数据采集 | C |
| `examples/train_bc.py`、`train_hgdagger.py` | 非当前 R13 路径 | C |
| `serl_robot_infra/franka_env` | Franka ROS/HTTP 环境 | WA2 不使用；完成依赖闭包测试后可从 WA2 运行包移除 |
| Franka 各实验目录 | RAM/USB/handover 等论文任务 | 论文参考代码；可归档，不属于 WA2 运行必需 |

不能直接手工只挑 `sac_hybrid_single.py` 一个文件，因为它递归依赖多个 common、network、vision 和 optimizer 模块。若要制作精简算法包，应先建立 import-closure 测试，再裁剪。

### 6.3 两份上游源码的同步要求

以下文件在 Actor 和 Learner 各有一份：

```text
catkin_ws/src/hil-serl-main/...
HILSERL_Learner/src/hil-serl-main/...
```

当前 Hybrid Agent 两份副本必须字节一致，尤其包括 `grasp_eps` 修改。交付时建议改为以下二选一：

1. 保持两份 vendored source，但用 SHA256/CI 强制一致；或
2. 建立一个版本化共享包，在 x86_64 Learner 和 aarch64 Actor 分别安装同一源码版本。

不建议继续依赖人工复制且没有 CI 检查。

---

## 7. R1～R13 阶段代码与最终属性

| 阶段 | 主要产出 | 最终属性 |
|---|---|---|
| R1 | ROS 接口盘点、Env 契约、Docker 验证 | 契约文档/配置保留；`verify_r1_contract.py` 为 T |
| R2 | Mock Env、隔离硬件 | `mock_*` 保留测试；`verify_r2_mock_env.py` 为 T |
| R3 | 只读真实状态 | 状态监控进入 P0；`r3_soak_readonly.py`、`verify_r3_*` 为 T |
| R4 | ServoL dry-run、hold、真实 Gate | `servo_session.py` 进入 P0；`r4_*`、`verify_r4_*` 为 T |
| R5 | 场景 reset | `scene_config.py`、`reset_executor.py` 进入 P0；`verify_r5_reset.py` 作为训练前单次 reset 运维入口保留 |
| R6 | 双相机与预处理 | image adapters 进入 P0；inventory/soak/verify 脚本为 T |
| R7 | SpaceMouse 干预 | intervention/watchdog 进入 P0；`verify_r7_*` 为 T |
| R8 | task 配置、fake env、Agent/Buffer | task/env factory 和 Buffer 进入 P0；`verify_r8_*` 为 T/P1 |
| R9 | transition、dummy server、fail-closed | transition/safety 进入 P0；dummy/verify 脚本为 T |
| R10 | 跨机 Agentlace、manifest、协议 | protocol/manifest 进入 P0/P1；R10 server/verify 为 T |
| R11 | 20 条成功 demo | demo IO 进入 P1；旧 6D demo/脚本转历史资产 |
| R12 | Reward Classifier | wrapper 和冻结模型进入 P0；采集/训练/阈值脚本为 P1 |
| R13 | Hybrid SAC、在线 HIL、脱手评测、Buffer 恢复 | 当前正式框架 P0 |

### 7.1 明确属于阶段测试的 Actor 脚本

以下文件在 R13 正式训练/评测路径中不是入口：

```text
scripts/r3_soak_readonly.py
scripts/r4_hold_zero_action.py
scripts/r4_visible_move_demo.py
scripts/r6_inventory_cameras.py
scripts/r6_soak_images.py
scripts/r9_dummy_trainer_server.py
scripts/verify_r2_mock_env.py
scripts/verify_r3_readonly_env.py
scripts/verify_r4_servo_dryrun.py
scripts/verify_r4_servo_gates.py
scripts/verify_r6_image_live.py
scripts/verify_r6_image_offline.py
scripts/verify_r7_intervention_live.py
scripts/verify_r7_intervention_offline.py
scripts/verify_r8_agent_buffer.py
scripts/verify_r8_config.py
scripts/verify_r9_actor_local.py
scripts/verify_r9_fault_stop.py
scripts/verify_r9_transition_offline.py
scripts/verify_r10_actor_remote.py
```

它们可在最终验收后移出生产运行包，但建议作为 `tests/hardware_gates/` 归档，因为机器人接口或镜像升级后仍有回归价值。

### 7.2 明确属于阶段测试的 Learner 脚本

```text
scripts/r10_learner_server.py
scripts/verify_r10_learner_loopback.py
scripts/verify_r8_agent_buffer.py
scripts/verify_r8_config.py
scripts/verify_r9_transition_offline.py
scripts/verify_r11_demo_load.py
scripts/augment_r11_demo_grasp.py
```

- `r10_learner_server.py` 只验证 server/广播，不执行当前 R13 梯度训练。
- `augment_r11_demo_grasp.py` 是旧 6D demo 迁移工具，当前原生 R13 7D demo 基线建立后可归档。

### 7.3 单元测试是否可以删除

`src/hilserl_wa2/tests/unit/` 不属于线上运行依赖，但不建议从交付源码包删除。推荐：

- 最小运行镜像可不复制测试目录；
- 源码交付包和 CI 包必须保留；
- 安全、transition、协议、Buffer、grasp、reward classifier 测试属于长期回归，不应作为“无用代码”永久删除。

---

## 8. 数据、日志和大文件梳理

当前磁盘占用主要不是源码：

| 路径 | 盘点时规模 | 内容 | 处理建议 |
|---|---:|---|---|
| `catkin_ws/runs/wa2_bottle_pick` | 约 30 GB | R9～R13 Actor demo、图像、分类器和日志 | 按 run 分类，不可整目录直接删 |
| `HILSERL_Learner/runs/wa2_bottle_pick` | 约 32 GB | checkpoint、demo、两类 Buffer cache、metrics | 训练资产，需先定基线 |
| `dustynv-jax-r36.4.0-arm64.tar.gz` | 约 4.2 GB | 未采用的 ARM64 JAX 基础镜像 tar | 2026-09-01 已从工作区删除 |
| `catkin_ws/failed` | 约 226 MB | 历史 R11 失败 episode | 2026-09-01 已按项目决策永久删除，不保留 JSON/PKL |
| `调试日志` | 约 1.6 MB | 阶段证据 | 不进运行镜像；建议随工程交付归档 |
| `catkin_ws/build`、`devel` | 数 MB | catkin 生成物 | 可重建；但当前容器运行前不要删 `devel`，除非验证重建流程 |

### 8.1 R13 必须优先保护的资产

```text
Actor:
  R13 原生 7D demo bundle
  R12/R13 正式 reward classifier checkpoint + threshold
  最终 actor/learner manifest

Learner:
  最终策略 checkpoint
  与其匹配的 metrics.jsonl
  demo_buffer_cache
  online_buffer_cache（若要求可续训）
  R13 demo 副本及 SHA256
```

策略 checkpoint 不能脱离以下内容单独交付：源码 hash、任务 YAML、space hash、demo hash、Agentlace 版本、分类器版本和启动参数。

### 8.2 历史 run 的建议级别

| 数据 | 建议 |
|---|---|
| R9/R10 fake/联调 run | Gate 结束后可从生产包删除，验收日志保留 |
| R11 5 fake/5 live/旧 6D demo | 当前 R13 不再训练使用；保留一份归档后可清理工作副本 |
| R11 20 success 6D demo | 若已确认 R13 原生 7D demo 完整，可转历史归档 |
| R12 原始 success/fail 图像 | 若需要未来修正分类器则保留；只交冻结模型时可外部归档 |
| R13 多个中间 checkpoint | 选择最终、最佳和必要恢复点后，其余可条件删除 |
| R13 Buffer 快照 | 若交付要求续训必须保留；只做推理部署可不交付 |
| classifier live dumps | hard-negative/false-positive 排查结束后可筛选归档，其余可删 |

---

## 9. 可以删除或移出运行包的文件清单

### 9.1 可直接重新生成的 G 类

以下通常可删除，不影响源码：

```text
所有 __pycache__/
所有 *.pyc
.pytest_cache/（若存在）
临时 *.tmp
空的 HILSERL_Learner/jax 文件
空的根目录 SHA256SUMS（已于 2026-09-01 删除；`artifacts/wheels/SHA256SUMS` 仍保留）
```

`catkin_ws/build/` 和 `catkin_ws/devel/` 也是生成物，但现有容器可能依赖 `devel/setup.bash`。只有在最终 Dockerfile/构建脚本能够从源码完整重建并通过 R13 回归后，才适合从源码交付包排除。

### 9.2 验收后可移出最小运行包的 T 类

- 第 7 节列出的所有 R2～R10 `verify_*`、soak、dry-run、dummy server。
- `naviai_controller/scripts/test*.py`。
- `src/hilserl_wa2/tests/unit/`：可不进运行镜像，但应保留在源码/CI 交付包。
- `docs/solution/R1_方案.md`～`R12_方案.md`、阶段验收 raw 文件：不进运行镜像，转工程归档。
- `run_spacemouse_teleop_from_yaml.py` 和独立 teleop：若最终运维流程明确只允许 R13 Actor 控臂，可移出最小运行包。

### 9.3 条件删除的 C 类

| 候选 | 删除前置条件 |
|---|---|
| `serl_robot_infra/franka_env` | WA2 import closure、Agent 初始化、Actor/Eval/Learner smoke 全通过 |
| Franka examples | 已保留论文源码 tag/commit 或外部归档；WA2 manifest 不再纳入 |
| `demo_grasp.py`、`augment_r11_demo_grasp.py` | 所有正式 demo 均为原生 7D，且不再迁移旧包 |
| R11 recorder/verify | R13 recorder 和 demo 校验覆盖完整，旧 6D 数据已归档 |
| R12 训练工具 | 明确交付只支持冻结分类器推理，不支持再训练 |
| joystick_drivers 非 spacenav 子包 | catkin 白名单构建与 SpaceMouse 回归通过 |
| `naviai_controller/third_party` 旧安装器 | 明确实际依赖版本、镜像中已固化并留有校验副本 |
| `dustynv-jax-*.tar.gz` | 未采用该镜像方案，根目录副本已于 2026-09-01 删除 |
| 历史 checkpoint | 已定义 final/best/resume 三类保留点并验证 final 可加载 |
| 历史 demo/分类器数据 | 已有不可变归档和 checksum，且确认不再用于 hard-negative 修正 |

### 9.4 明确禁止删除

```text
Actor r13_actor_train.py / r13_actor_eval.py
Learner r13_learner_train.py
envs/、experiments/、wrappers/ 的核心文件
Actor ros_adapters/ 的真实硬件实现
Actor interventions/ 的安全与上传实现
hil-serl-main/serl_launcher 当前 import 闭包
naviai_controller 实际 Python 包
spacenav_node 当前输入驱动
task/scene/camera/network 配置
Agentlace 固定 wheel 及 SHA256
ResNet10 预训练参数
正式 demo、classifier、最终 checkpoint、manifest
```

---

## 10. 当前工程中需要在封装前解决的结构问题

### 10.1 两端源码重复

`hilserl_wa2` 和 `hil-serl-main` 在 Actor/Learner 各有副本。当前 source tree manifest 只保护部分共享目录，`interventions`、专属 scripts 和部分 tests 不在 hash 中。

建议交付前建立 CI：

1. 对 hash 范围强制字节一致；
2. 对 `sac_hybrid_single.py` 强制一致；
3. 允许 Actor-only 与 Learner-only 文件白名单差异；
4. manifest 构建后执行 Actor/Learner compare；
5. 任何核心变更都重建 manifest。

### 10.2 配置存在多份镜像

根目录 `configs/` 与 `catkin_ws/src/hilserl_wa2/configs/` 有重复配置；目前核心文件 SHA256 一致，但长期容易漂移。

建议最终确定单一配置源：

- 源码包内 `hilserl_wa2/configs` 为权威；
- 部署时复制/挂载，不再人工维护根目录镜像；或
- 根目录为权威，构建时生成包内配置。

不能继续两处手工修改。

### 10.3 Docker 尚未完全固化

当前 `docker/dockerfile` 是草案，Compose 仍依赖已有容器可写层中的 `hil-actor`。这不是可交付的最终状态。

封装前应：

- 固定 Ubuntu/JetPack/ROS/CUDA/JAX 版本；
- 固定 aarch64 Python wheel 和 Agentlace wheel；
- 固定 `naviai_controller` 依赖；
- 从空镜像重建 `hil-actor`；
- 运行 R1、SpaceMouse、fake Agent、R13 Actor 加载和真机安全回归；
- 生成版本化镜像和 SHA256/SBOM。

### 10.4 当前参数口径需冻结

当前代码相对早期 R13 方案已经变化：

| 参数 | 当前代码 |
|---|---:|
| continuous target entropy | `-3.0` |
| grasp penalty | 默认 `-0.002`，CLI 可改 |
| train grasp epsilon | 默认 `0.15` |
| eval grasp epsilon | `0` |
| SAC min temperature | 默认 `0.01` |
| resume temperature kick | 默认 `0.05` |

最终交付不应只依赖 argparse 默认值。建议生成一个版本化 experiment YAML/launch script，记录所有参数并纳入 manifest。

### 10.5 网络与长任务可靠性

Actor/Learner LAN 使用 Agentlace，与外部 Codex/代理无关。但正式训练仍需：

- 固定 Learner IP、5588/5589 端口；
- Actor 单端重启后执行 upload cursor align；
- 正常退出 Learner，等待 checkpoint 和 Buffer snapshot 完成；
- 保留 `actor_status.json`、`status.json`、metrics 和 manifest；
- 不以“终端仍在输出”代替 server count/confirm 检查。

---

## 11. 建议的最终交付包结构

```text
HIL-SERL-WA2-release/
├── actor/
│   ├── hilserl_wa2/              # Actor P0 源码
│   ├── hil-serl-main/            # 经验证的最小算法依赖或固定源码包
│   ├── naviai_controller/
│   ├── spacenav_node/
│   ├── configs/
│   ├── scripts/
│   │   ├── r13_actor_train.py
│   │   ├── r13_actor_eval.py
│   │   ├── record_r13_demos.py
│   │   ├── start_spacemouse_joy.sh
│   │   └── build_r13_manifest.py
│   └── Dockerfile / compose / lock
├── learner/
│   ├── hilserl_wa2/
│   ├── hil-serl-main/
│   ├── scripts/r13_learner_train.py
│   ├── configs/
│   ├── requirements-lock/
│   └── artifacts/resnet10_params.pkl
├── models/
│   ├── reward_classifier/
│   └── policy_checkpoint/
├── demos/
│   └── baseline_7d_demo/
├── manifests/
│   ├── actor.json
│   ├── learner.json
│   └── SHA256SUMS
├── tests/
│   ├── unit/
│   ├── fake_smoke/
│   └── hardware_gates/
└── docs/
    ├── README.md
    ├── 部署手册.md
    ├── 操作与安全手册.md
    ├── 模型与数据说明.md
    └── 故障排查.md
```

源码交付包、运行镜像和训练资产应分开。尤其不要把 60 GB `runs/` 整体复制进 Docker 镜像。

---

## 12. 清理执行前的验收清单

后续真正删除或归档前，至少完成：

- [ ] 冻结 Actor/Learner 源码版本和 source tree hash。
- [ ] 冻结当前任务 YAML、scene、camera、network 配置。
- [ ] 冻结正式 R13 7D demo SHA256。
- [ ] 冻结 reward classifier checkpoint、threshold 和数据说明。
- [ ] 选择 final/best/resume checkpoint 并逐一验证可加载。
- [ ] 明确是否需要交付“继续训练”能力；若需要则保留两类 Buffer cache。
- [ ] 从干净环境重建 Actor Docker 和 Learner Conda 环境。
- [ ] 运行无硬件单测和 fake smoke。
- [ ] 运行 SpaceMouse、相机、状态、ServoL、reset 安全 Gate。
- [ ] 完成一次短 HIL 闭环和一次 `grasp_eps=0` 脱手评测。
- [ ] 对准备删除的训练资产先做只读归档和 checksum。
- [ ] 删除后再次重建 manifest，并确认 Actor/Learner compare 通过。

---

## 13. 最终结论

当前工程已经形成完整的 HIL-SERL-WA2 R13 框架，真正的生产主线只有两条：

```text
Actor：r13_actor_train.py / r13_actor_eval.py
  → WA2 Env + SpaceMouse + Grasp + Classifier + Transition + Agentlace

Learner：r13_learner_train.py
  → Demo/RL Buffer + 50/50 RLPD + Hybrid SAC + Checkpoint/Publish
```

R1～R12 并非全部“废代码”：各阶段中形成的 Env、ServoL、reset、相机、干预、协议、transition、demo 和 classifier 模块已经沉淀为 R13 核心；只有以 `verify_*`、soak、dry-run、dummy server 为主的入口属于阶段验收代码。

可以优先清理的是 Python 缓存、临时文件和已确认可重建的生成物。占用最大的 `runs/`、checkpoint、demo、Buffer 和 classifier 数据必须先确定交付能力边界、建立归档和 SHA256 后再处理。上游 Franka 代码和未用算法可以精简，但必须在重构 `launcher.py` 顶层 import 并完成 import-closure、Agent 初始化、两端握手和 R13 smoke 后进行，不能按文件名直接删除。

对外封装时，建议保留完整源码/测试开发包，同时另做最小 Actor 运行镜像和 Learner 训练环境；不要把“运行包精简”演变成“永久丢失测试、论文参考代码和训练可复现资产”。
