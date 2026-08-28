# HIL-SERL-WA2 全流程代码体检报告（Codex）

> 检查日期：2026-08-26  
> 检查方式：静态代码审查，不进入 `hil-actor` / `hil-learner` 环境，不连接机器人，不启动 ROS、Agentlace、JAX 训练或真机动作。  
> 修改范围：本次只新增本报告，未修改任何 Actor、Learner、算法、配置或部署代码。

## 1. 结论摘要

当前工程已经形成一条基本完整的 HIL-SERL-WA2 链路：Actor 负责真机观测、混合动作执行、人工干预、奖励分类与数据上传；Learner 负责示教/在线 buffer、RLPD 采样、6D 连续 SAC 与 1D 离散夹爪 critic 更新、参数发布和检查点保存。混合动作拆分、50/50 RLPD 采样、干预动作落盘等算法主干总体与论文及上游源码一致。

但按“可安全上真机、可长时间训练、可封装交付”的标准，当前 R13 主链路仍不建议直接作为最终交付基线。静态检查发现以下最需要优先处理的问题：

1. **Actor 在第一次收到 Learner 参数之前即可 reset、预热并进入真机 step**。此时 Actor 使用 `seed=0` 的本地随机初始化参数，而 Learner 默认使用 `seed=42`，首次发布又要等待在线 buffer 达到 100；因此开局动作不是 Learner 的策略。这是当前最高优先级的真机安全与数据质量问题。
2. **Actor 退出时先设置上传线程停止标志、后发送 flush**，上传线程可能在处理 flush 前退出。现有一次真实 R13 状态中仍有 `PENDING_ENV=498`、`PENDING_INTVN=328`，与该静态问题一致。
3. **R13 数据入口没有强制“先握手后写入”，也没有锁定唯一 session**；相比 R10 已有逻辑发生了协议保障退化，旧 Actor、错误客户端或跨 session 数据存在混入 buffer 的可能。
4. **夹爪执行失败没有传播到训练 transition**。底层返回 `ok=False` 时，上层仍可把夹爪命令记录成已执行动作，直接污染 1D 夹爪训练标签。
5. **奖励分类器推理异常会被静默降级成 reward=0 并继续运行**。持续异常时，系统可能在没有硬故障的情况下采集大量零奖励数据。
6. **源码/参数一致性门禁覆盖不完整**：manifest 不覆盖 R13 主入口、ROS adapter、干预及上传代码，实际运行 manifest 的 `params_tree_signature` 为空，关键代码或参数树漂移仍可能握手通过。

现有运行状态同时说明主干并非完全不可用：Learner 已完成 2,042 次更新、夹爪 critic 有更新、未出现 NaN/Inf、发布参数 41 次。但这些结果不能消除上述开局策略、数据完整性和会话隔离风险。

## 2. 检查范围、依据与限制

### 2.1 工程边界

| 端 | 主机/环境边界 | 本仓库工作目录 | R13 正式入口 | 主要责任 |
|---|---|---|---|---|
| Actor | WA2 Orin 的 `hilserl` Docker，Conda `hil-actor` | `catkin_ws/`，容器内对应 `/root/catkin_ws` | `catkin_ws/src/hilserl_wa2/scripts/r13_actor_train.py` | ROS/相机/状态读取、ServoL、复位、SpaceMouse、分类器推理、7D 动作执行、transition 构造与上传 |
| Learner | 本地笔记本，Conda `hil-learner` | `HILSERL_Learner/` | `HILSERL_Learner/src/hilserl_wa2/scripts/r13_learner_train.py` | demo/online/intervention buffer、RLPD 混合采样、hybrid SAC 更新、checkpoint/snapshot、参数广播 |
| 上游参考 | 论文官方实现 | 两端各自的 `src/hil-serl-main/` | 上游 `train_rlpd.py` 与 `serl_launcher` | HIL-SERL/RLPD/SAC 基线算法与网络实现 |

Actor 与 Learner 都保留了一份 `hilserl_wa2` 共享语义代码以及一份上游 `hil-serl-main`。这有利于离线部署，但形成了“双份源码同步”的维护风险。当前上游 `serl_launcher` 主体在两端一致；WA2 共享代码已有少量漂移，例如 Actor 的 SpaceMouse stale-session 安全逻辑未同步到 Learner 副本。

### 2.2 检查内容

- 论文 HIL-SERL/RLPD 数据流与项目 hybrid SAC 实现的对应关系。
- R13 Actor、Learner 正式入口及其直接依赖。
- demo、online、intervention buffer 的写入、采样、缓存与恢复。
- 6D 连续动作、1D 离散夹爪动作的生成、执行和训练路径。
- Agentlace 握手、参数广播、数据上传、重连及 fail-closed 逻辑。
- 分类器奖励、episode 终止、复位、ServoL 与 SpaceMouse 安全路径。
- Docker、配置、manifest、checkpoint、评估入口和现有运行产物。
- 阶段方案和 R1–R12 验收日志，用于区分已验收基础模块与 R13 未闭环内容。

### 2.3 本次未做事项

- 未激活虚拟环境，未 import JAX/ROS/Agentlace 应用模块。
- 未运行单元测试、训练、推理、ROS topic/service 或真机动作。
- 未验证相机延迟、ServoL 实际频率、GPU 显存、网络抖动和夹爪真实反馈。
- 未删除、移动、格式化或归档任何工程文件。

因此，报告中的“确定问题”均可由代码控制流或现有状态文件直接证明；标为“待动态验证”的项目不能仅靠本次静态检查下最终结论。

## 3. 论文/源码与当前数据流对齐

### 3.1 正常设计链路

```text
Actor / WA2 Orin
  ROS状态 + head/wrist图像
              │
              ▼
       WA2Env / wrappers
              │ obs
              ▼
  hybrid policy: 6D arm + 1D grasp
              │
      SpaceMouse可覆盖执行动作
              │
              ▼
     ServoL + 灵巧手实际执行
              │
              ▼
 transition(obs, executed_action, reward,
            next_obs, mask, done, metadata)
          │                 │
          │普通样本          └─人工干预样本
          ▼                           ▼
     actor_env                   actor_env_intvn
          └──────── Agentlace upload ────────┘
                              │
                              ▼
Learner / Laptop
  online replay buffer     demo/intervention buffer
             └──── 50% + 50% batch ────┘
                              │
                              ▼
      6D continuous SAC + 1D grasp critic
                              │
                   publish_network(params)
                              │
                              └──────────────► Actor
```

### 3.2 已正确对齐的关键点

1. **6D 与 1D 是同一 transition 内的混合动作，但训练头已拆分。**  
   `sac_hybrid_single.py` 的连续 critic 使用 `actions[..., :-1]`，夹爪 critic 单独使用最后一维；采样时再拼成 7D。项目不是把两类动作拆成两个完全独立训练任务，而是共享视觉/状态编码和 replay transition、分别计算连续与离散目标。这符合当前 WA2 hybrid SAC 设计。

2. **`grasp_penalty=-0.02` 已进入夹爪 critic 的奖励目标。**  
   非零夹爪动作会得到轻微代价，用来抑制无意义的反复开合；它不是任务失败惩罚，也不应替代成功奖励。该值对夹爪策略有实际影响，但量级相对成功奖励较小。

3. **RLPD 采样比例实现为在线 50% + demo/intervention 50%。**  
   Learner 分别用 `batch_size // 2` 从 replay 和 demo buffer 取样，再拼接后更新。这个结构和论文“离线示教 + 在线经验联合更新”的主逻辑一致。

4. **人工干预记录执行动作而非原策略动作。**  
   transition 构造会优先采用 `info["intervene_action"]`，并把干预样本同时送入普通在线流和 intervention/demo 流，符合 HIL-SERL 对人类修正动作再利用的思路。

5. **终止 mask 基本符合 bootstrap 语义。**  
   真正 `terminated` 时 mask 为 0，时间截断 `truncated` 可继续 bootstrap；这与常见 SAC/RLPD 处理一致。

## 4. 问题总表

严重度定义：

- **P0 阻断**：可能直接造成真机非预期动作、安全链路失真或核心训练数据不可相信，上线前应处理。
- **P1 高**：长时间运行、数据完整性、协议一致性或可恢复性存在明显风险。
- **P2 中**：不会必然立即失败，但会降低训练效果、复现性或交付可靠性。
- **P3 低**：工程治理、CLI 易用性或验证覆盖问题。

| ID | 严重度 | 归属 | 问题摘要 | 判定 |
|---|---|---|---|---|
| A-01 | P0 | Actor/跨端 | 首次 Learner 参数到达前 Actor 即可执行本地随机策略 | 确定 |
| A-02 | P0 | Actor | 夹爪执行 `ok=False` 未上抛，transition 仍记录为已执行 | 确定 |
| A-03 | P1 | Actor | 分类器推理异常静默变成 reward=0，训练继续 | 确定 |
| A-04 | P1 | Actor/跨端 | 上传队列满时允许丢弃未确认 transition 并跨越序号缺口 | 确定 |
| A-05 | P1 | Actor | 退出顺序可能跳过最终 flush，现有状态残留 826 条待上传 | 确定且有运行证据 |
| A-06 | P1 | Actor | 分类器 sanity 的失败样本检查不可达 | 确定 |
| A-07 | P1 | Actor | 手动 episode cap 可 reset 而不标 transition 为 truncated | 条件触发 |
| A-08 | P2 | Actor | 无最低控制频率/超时门禁，慢循环只打印不故障 | 确定，影响待真机验证 |
| L-01 | P1 | Learner/协议 | R13 不强制先握手后写 datastore，且 session 可被覆盖 | 确定 |
| L-02 | P1 | Learner | demo manifest/hash 与实际加载的 episode 文件不一致 | 确定 |
| L-03 | P1 | Learner | demo episode 仅做 transition 级校验，缺少成功/终止/夹爪边沿整体验证 | 确定 |
| L-04 | P1 | Learner | `--resume` 找不到 checkpoint 时会静默从头开始 | 确定 |
| L-05 | P1 | Learner | checkpoint 与 buffer snapshot 没有原子一致的同一步恢复约束 | 确定 |
| L-06 | P2 | Learner | 异步 buffer 快照允许 torn snapshot，替换过程先删旧缓存 | 确定 |
| L-07 | P2 | Learner/算法 | temperature floor 每次回写并重建优化器，实际等价于长期固定 α=0.05 | 确定且有运行证据 |
| X-01 | P0 | 跨端 | manifest 源码哈希漏掉 R13 入口、interventions、ROS adapters | 确定 |
| X-02 | P1 | 跨端 | `params_tree_signature` 默认为空且未形成强制门禁 | 确定且有运行证据 |
| X-03 | P1 | 跨端 | `allowed_actor_cidr` 在 R13 中未用于实际客户端身份/网段约束 | 确定 |
| X-04 | P1 | 评估 | eval 随机采样、checkpoint 恢复成功未验证、无 manifest 门禁 | 确定 |
| D-01 | P1 | 部署 | compose 仍运行基础镜像，Dockerfile 的“独立源码快照”不含 WA2 主包 | 确定 |
| D-02 | P2 | 入口治理 | R13 正式入口与上游 WA2 `run_*.sh` 并存，后者仍指向旧 `train_rlpd.py` | 确定 |
| D-03 | P3 | 配置 | `store_true, default=True` 的布尔选项无法从 CLI 关闭 | 确定 |
| D-04 | P3 | 验收 | 有 R1–R12 正式验收，未见独立 R13 阶段验收报告 | 确定 |

## 5. Actor 端详细体检

### A-01：首次网络参数前就可能执行本地随机策略（P0）

**证据链：**

- Actor 在 `r13_actor_train.py:95-107` 用固定 `seed=0` 创建本地 hybrid agent。
- Learner 在 `r13_learner_train.py:702-711` 用 CLI seed，默认值为 42，创建另一套初始参数。
- Actor 握手后立即注册回调、执行 `env.reset()`、JIT 预采样，并进入 `env.step()`；见 Actor `r13_actor_train.py:458-484,582-601`。
- Learner 直到 online replay 达到 `--training-starts`（默认 100）才首次 `publish_network()`；见 Learner `r13_learner_train.py:1049-1067`。
- Actor 的网络 watchdog 初始 disabled，只有参数回调到达后才启用；`--require-network-update` 也在主循环结束后才检查，见 Actor `r13_actor_train.py:764-770`。

**影响：** 开局约 100 条数据以及对应真机运动可能来自与 Learner 无关的随机初始化策略；不仅影响安全，也会把低质量 transition 写入 online replay。

**建议（本次未实施）：** 把“收到并验证至少一次 Learner 参数”设为 `env.reset/step` 前硬门禁；握手响应应返回当前参数或 Learner 启动后立即发布一次；网络 watchdog 从允许真机运动的时刻起启用。

### A-02：夹爪失败被记录为成功执行（P0）

`WA2Env.request_hand()` 在 `wa2_env.py:364-389` 明确返回 `{"ok": bool, "command": ...}`。但 `WA2GraspActionWrapper._fire_hand()` 在 `grasp_action.py:71-76` 只读取 `command`，忽略 `ok`；随后 `step()` 在 `grasp_action.py:93-115` 更新 `_last_nonzero` 并将非零 `grasp_command` 写入 7D 执行动作。

**影响：** 服务调用失败、手部未到位或动作被拒绝时，replay 仍把 `+1/-1` 当作已执行动作。夹爪 critic 的 `(s,a,r,s')` 物理语义被破坏，人工干预数据也可能错误。

**建议：** `ok=False` 应触发安全 fault 或至少把 executed grasp 记为 0 并标注失败；更理想的是使用手部状态反馈确认完成，而不是只依赖 service 返回。

### A-03：奖励分类器异常静默降级（P1）

`wrappers/reward_classifier.py:336-365` 捕获宽泛异常后将 `prob=0.0`，仅记录一次错误并继续。Actor 主循环检查状态 stale、奇异位形、Servo fault、NaN 等，但没有分类器连续异常 watchdog。

**影响：** 模型、图像 shape、设备或推理线程一旦持续失败，任务奖励会长期为 0，系统仍可继续采集和训练，造成难以察觉的数据污染。

**建议：** 区分“合法预测为 0”和“推理失败”；连续失败达到阈值后 fail-closed；状态文件记录异常计数、最近成功推理时间和模型 hash。

### A-04/A-05：上传队列允许未确认丢失，退出 flush 顺序错误（P1）

- `actor_upload_queue.py:44-54` 在容量满时直接丢最旧数据并增加 `dropped_unacked`。
- `actor_upload_queue.py:60-100` 默认 `allow_gap=True`，允许跳过缺口并继续推进服务端 cursor。
- Actor 只打印 `UPLOAD_DROP_UNACKED`，没有因此停止真机；见 `r13_actor_train.py:547-552`。
- 正常退出段先 `upload_stop.set()`，之后才 `put_nowait("flush")`；而 worker 条件是 `while not upload_stop.is_set()`，见 `r13_actor_train.py:553-565,753-760`。线程可在消费 flush 前退出。

现有运行文件 `catkin_ws/runs/wa2_bottle_pick/20260825_224215_r13/actor_status.json` 记录：

```text
ONLINE_N=8006
PENDING_ENV=498
PENDING_INTVN=328
fault_reason=sigint
```

即退出时共有 826 条本地待确认记录。该文件不能单独证明它们最终永久丢失，但证明“退出前未排空”已经真实发生。

**建议：** 停止采集后先同步 drain 并等待 ack，再停止 worker；落盘 WAL/队列并在重启后恢复；队列满或 gap 应触发 fail-closed，尤其 intervention 数据不得静默跳过。

### A-06：分类器 sanity 失败样本检查不可达（P1）

`r13_actor_train.py:152-160` 中，failure false-positive 检查缩进在 success 阈值失败的 `raise` 之后。正常 success 检查通过时不会进入该块；失败时又先抛异常，因此 failure 检查永远不会执行。此外，找不到 `success.pkl` 时整个 sanity 会跳过。

**建议：** 将 success 与 failure 两项校验改为并列门禁，明确要求完整 sanity bundle，并在状态文件记录校验结果。

### A-07：Actor 外层 episode cap 的终止语义不完整（P1，条件触发）

Actor 在 transition 构造之后单独判断 `hit_cap` 并 reset；若 CLI `--max-steps-per-episode` 小于 `WA2Env.max_steps`，该条 transition 本身可能没有 `truncated=True` 或 episode-end 元数据。当前默认值都为 4000，因此默认配置下通常不会触发分歧，但一旦独立调整 CLI 就会产生跨 episode 的 next-observation/边界风险。

### A-08：控制周期只有上限 sleep，没有最低频率门禁（P2）

主循环只在运行快于目标周期时 sleep；运行过慢时仅打印 Hz，不会告警或停止。Servo session 有短时 latch/watchdog 语义，若 JAX、图像或网络造成长尾延迟，真实动作可能呈现 stop-go，而 transition 的固定折扣仍把每一步视为同等时间间隔。

该问题需在真机上记录 p50/p95/p99 step latency、Servo 实际频率和超时次数后确定影响等级。

## 6. Learner 端详细体检

### L-01：R13 会话与数据入口保障相对 R10 退化（P1）

R13 `r13_learner_train.py:967-980` 的 datastore callback 只更新计数，没有检查 `handshake_accepted`；`r13_learner_train.py:989-1001` 每次收到成功握手都会覆盖 `accepted_session_id`。

对比 R10 `r10_learner_server.py:225-256`：

- datastore 先于握手会把 schema 标为失败；
- 第一个 session 被接受后，后续不同 session 会被拒绝。

R13 Agentlace 数据 batch 本身也没有 session id 绑定。因此，仅有 request 握手并不能证明随后写入的 batch 来自该 session。

**建议：** 恢复并加强 R10 保障：数据写入前必须握手；单次训练锁定 Actor/session；每批数据携带并校验 session、source hash、递增序号；切换 session 必须显式结束旧会话。

### L-02：demo 哈希对象与实际训练对象不一致（P1）

`build_r13_manifest.py` 对 bundle 根目录的 `demo.pkl` 计算 hash，但 Learner 实际遍历加载 `episodes/epXXX.pkl`（`r13_learner_train.py:286-315`）。demo buffer cache key 也沿用 manifest 中的 `demo.pkl` hash。

**影响：** episode 文件发生修改而 `demo.pkl` 不变时，握手/缓存仍可能判定一致；反之 demo 索引变化也无法证明每个 episode 内容未变。缓存可能命中旧数据。

**建议：** 对实际加载的所有 episode 文件建立排序后的逐文件 SHA-256 manifest，并把该聚合 hash 作为 cache key；验证 `demo.pkl` 与 episode bundle 的等价关系。

### L-03：demo 只做 transition 级检查，episode 语义校验不足（P1）

`_insert_demo_episodes()` 校验单条 transition 和 7D action，并统计总非零夹爪动作；但没有逐 episode 强制验证：

- 成功 episode 最后一条 reward/terminated 是否正确；
- observation 与 next_observation 是否连续；
- 每个成功 episode 是否包含所需 grasp/release 边沿；
- sidecar 元数据、episode 数、transition 数与 manifest 是否完全对应；
- 是否混入失败/中止采集片段。

工程中已有 `validate_success_episode`、`validate_r13_grasp_edges` 等能力，但 R13 Learner 主入口没有把它们作为 demo ingest 硬门禁。

### L-04/L-05/L-06：恢复与快照一致性风险（P1/P2）

1. `--resume` 请求后若找不到 latest checkpoint，代码没有失败分支，会继续使用新 agent，从零开始训练。
2. checkpoint 与 online/demo buffer snapshot 分别保存，恢复时没有强制它们来自同一个 learner step；online cache 元数据中的 `learner_step` 没被用作恢复一致性门禁。
3. 后台 snapshot 明确不在完整写入期间持有全局锁，ring buffer 正在写入/覆盖时可能产生轻微 torn snapshot。
4. cache 替换过程先删除旧目录再 rename，新写入或进程崩溃时可能同时失去旧的可用快照。

**建议：** resume 找不到目标必须失败；checkpoint、两个 buffer 和 manifest 使用共同 generation/step；先写临时 generation、fsync/校验后原子切换 `LATEST` 指针，并保留至少一个上一代快照。

### L-07：temperature floor 实际抑制自动熵调节（P2）

`ensure_min_temperature()` 在 α 低于 0.05 时调用 `set_sac_temperature()`，后者重建 temperature optimizer。若优化器每次更新都试图把 α 降到阈值下方，就会每步回写并重置其优化状态。

现有 Learner 状态：

```text
learner_step=2042
_temp_floor_hits=2042
temperature=0.05
```

说明 2,042 个 learner step 每一步都命中 floor，当前行为实际上接近固定 α=0.05，而不是带下界的平滑自动调节。这与 `grasp_penalty=-0.02` 是两个独立问题：前者影响连续策略的熵温度，后者只进入离散夹爪 critic 奖励。

**建议：** 若目标就是固定温度，应明确关闭 temperature 学习并配置固定 α；若要保留自动调节，应在参数化/损失层实现可微下界，不要在每步替换参数并重建优化器。

## 7. 跨端协议、评估与部署体检

### X-01/X-02：一致性门禁没有覆盖真正的全流程代码（P0/P1）

`experiments/r10_protocol.py:161-196` 的 `source_tree_manifest()` 只覆盖：

- `configs/`、`envs/`、`experiments/`、`wrappers/`；
- 上游 `serl_launcher`；
- 上游 WA2 experiment 和少量配置文件。

它没有覆盖：

- `scripts/r13_actor_train.py` 与 `scripts/r13_learner_train.py`；
- `interventions/`，包括 SpaceMouse 与上传队列；
- `ros_adapters/`，包括 ServoL、状态、图像与 reset；
- Docker/requirements/entrypoint。

因此两端即使安全、上传或主循环代码不同，也可得到相同 source hash。`build_r13_manifest.py` 又默认允许空 `params_tree_signature`，manifest compare 明确忽略该字段；Actor 也只在它非空时校验。现有两端 R13 manifest 的该字段均为空。

**建议：** 构建“运行闭包 manifest”，覆盖正式入口及所有直接依赖；禁止空参数签名；Learner 创建 agent 后校验实际签名，Actor 在接受广播参数时也校验结构签名。代码 hash、模型 hash、demo hash、wheel hash、task config 应共同进入 session provenance。

### X-03：`allowed_actor_cidr` 仅是配置语义，R13 未真正约束客户端（P1）

R13 会将网络配置纳入 hash，但没有像 R10 preflight 那样验证 CIDR，也没有根据连接来源执行身份认证。Agentlace 服务使用 host 网络时，CIDR 字段不能代替防火墙或应用层认证。

**建议：** 至少落实主机防火墙/绑定接口和单 Actor 白名单；更完整的做法是为握手和 batch 增加认证材料，不能只依赖 manifest 内容相同。

### X-04：评估结果不可充分复现（P1）

`r13_actor_eval.py` 使用 `argmax=False` 进行连续动作随机采样；恢复 checkpoint 后无条件打印已加载，没有验证 checkpoint 是否存在、参数是否实际变化，也没有 task/source/classifier/demo provenance 门禁。

**影响：** 同一 checkpoint 多次评估可能产生不同轨迹；错误路径或缺失 checkpoint 可能被误认为已恢复；结果无法充分证明评估对应哪套完整代码和模型。

**建议：** 正式验收默认 deterministic/argmax，单独保留 stochastic evaluation；恢复后验证 checkpoint step 和参数摘要；输出完整 manifest 与 classifier hash。

### D-01：Docker 交付链不自包含（P1）

- `docker/docker-compose.hilserl.yml:3-5` 使用 `ros1_docker:latest`，没有使用 Dockerfile 构建的 `hilserl:actor-v1`。
- compose 依赖固定宿主机路径 `/home/naviai/hilserl_orin/catkin_ws:/root/catkin_ws`。
- `docker/dockerfile:98-105` 声称保存独立源码快照，但只复制 `catkin_ws/src/hil-serl-main`，没有复制 `hilserl_wa2`、机器人 ROS package、SpaceMouse package 等正式 Actor 依赖。

**影响：** 当前开发机可通过 bind mount 工作，但交付到新 Orin 后，镜像本身无法提供完整 R13 Actor；compose 也没有消费已构建的固定环境镜像，运行结果仍依赖基础镜像可写层和宿主目录状态。

### D-02/D-03：入口和配置存在歧义（P2/P3）

- 正式 R13 入口是 `hilserl_wa2/scripts/r13_*`，但两端上游 `examples/experiments/wa2/run_actor.sh` / `run_learner.sh` 仍调用旧 `train_rlpd.py`。交付人员若误用这些脚本，会绕过 R13 的 7D、握手和安全扩展。
- Actor/Learner/eval 的 `--end-episode` 以及部分 `--debug` 使用 `action="store_true", default=True`，从命令行无法关闭，help 语义与实际能力不一致。
- task YAML 与入口 argparse 同时维护一批训练/安全默认值，部分值由代码再次限制，配置的唯一真源尚未形成。

### D-04：R13 尚缺独立正式验收闭环（P3）

`调试日志/阶段验收日志/` 中有 R1–R12 正式验收记录，R13 有方案和运行产物，但未见独立的 R13 验收报告。现有运行状态只证明部分训练曾运行，不等价于 Actor/Learner 全流程、断网、重启、flush、恢复、确定性评估和交付镜像均通过。

另外，仓库根部 `.git` 是空的只读目录，当前工作副本无法提供正常 Git commit 追溯。若这是交付副本而非工具挂载限制，应在封装前补齐版本号、commit 或不可变源码归档 hash。

## 8. 现有运行证据解读

本次只读取状态文件，没有启动或续跑任务。

### 8.1 Learner 状态

文件：`HILSERL_Learner/runs/wa2_bottle_pick/20260825_224215_r13/status.json`

| 指标 | 值 | 解读 |
|---|---:|---|
| `DEMO_FILE_N` | 15,250 | demo 文件侧 transition 计数 |
| `DEMO_BUFFER_N` | 43,490 | demo 基线 + 在线干预后的总量 |
| `INTVN_N` | 28,240 | 运行期进入 intervention/demo buffer 的数量 |
| `ONLINE_N` | 28,818 | Learner online replay 数量 |
| `learner_step` | 2,042 | 已发生实际更新 |
| `PUBLISH_COUNT` | 41 | 已广播参数 |
| `GRASP_CRITIC_UPDATE` | true | 夹爪 critic 更新路径已被触发 |
| `NAN_OR_INF` | false | 该状态未记录数值异常 |
| `_temp_floor_hits` | 2,042 | 每个 learner step 均触发温度下界 |

### 8.2 Actor 状态

文件：`catkin_ws/runs/wa2_bottle_pick/20260825_224215_r13/actor_status.json`

| 指标 | 值 | 解读 |
|---|---:|---|
| `steps_executed` | 8,006 | 本 session 真机 step 数 |
| `network_update_count` | 6 | Actor 接收过 Learner 参数 |
| `succeed_episodes` | 7 | Actor 记录的成功 episode 数 |
| `PENDING_ENV` | 498 | 退出时未确认普通样本 |
| `PENDING_INTVN` | 328 | 退出时未确认干预样本 |
| `fault_reason` | `sigint` | 因人工中断进入 fail-closed 状态 |

Actor 的 8,006 与 Learner 的 28,818 不应直接要求相等，因为 Learner 状态可能跨 Actor session/恢复累积；但当前状态格式没有完整 generation/session 分解，正说明需要加强 provenance 和逐 session 对账。

## 9. 静态质量与已有安全基础

值得保留和继续沿用的实现包括：

- ServoL session 有命令锁存、watchdog、fault 状态和退出停止处理。
- 状态/图像 stale、奇异位形、保护性停止等检查已纳入 Actor 主循环。
- reset executor 对复位模式和资源仲裁做了较清晰的封装。
- SpaceMouse 干预动作会覆盖策略动作并以执行动作进入 transition。
- 上游 hybrid SAC 对 7D action 有 shape 断言，连续与夹爪更新路径清晰。
- Learner 对 NaN/Inf、grasp critic 是否更新、publish count 等已有状态观测。
- Actor/Learner 的上游 `serl_launcher` 源文件当前未发现字节级漂移，网络配置副本和 Agentlace wheel hash 也保持一致。

静态语法扫描中，HIL-SERL/WA2 主体 Python 文件可被 Python 3 AST 解析，相关 shell 入口通过 `bash -n`。`catkin_ws/src/joystick_drivers/wiimote/` 下发现两份历史脚本不能被 Python 3 AST 解析，位置为 `nodes/wiimote_node.py:128` 和 `src/wiimote/WIIMote.py:189`；它们属于第三方旧 wiimote 包，不在 R13 SpaceMouse 主链路。本项不应直接算作 HIL-SERL 算法故障，但交付时应确认它们是否仍需保留以及目标 ROS Python 版本。

## 10. 建议修复与验收优先级（仅建议，未修改）

### 第一批：再次允许真机在线训练前

1. Actor 必须在首次成功接收、验证 Learner 参数后才允许 reset/step。
2. 夹爪 `ok=False` 必须进入故障或明确的未执行动作语义。
3. 分类器推理异常必须可计数并在持续失败时 fail-closed。
4. 修正退出 drain/flush 顺序；干预和普通 transition 均实现可恢复 WAL，退出必须对账为 0 pending。
5. R13 恢复“先握手后写入”和 session lock，每批数据绑定 session/序号。
6. manifest 覆盖 R13 运行闭包，并强制非空参数树签名。

### 第二批：长时间训练与断点恢复前

1. 实际 episode demo 文件逐文件 hash，并做 episode 级语义校验。
2. checkpoint + online buffer + demo/intervention buffer 建立同 generation 原子快照。
3. `--resume` 目标不存在时硬失败；恢复后核对 step、buffer count、参数摘要。
4. 明确 temperature 策略：固定 α 或真正带下界的自动调节，二选一。
5. 对 Actor/Learner 按 session 做 upload count、ack、duplicate、gap 和 replay insert 对账。

### 第三批：封装交付前

1. compose 改为使用正式构建镜像；镜像包含完整 Actor 运行包或明确声明并校验外部只读源码包。
2. 只保留一个受支持的 R13 启动入口，旧上游脚本明确标为参考/禁用。
3. 统一 task YAML、CLI 和部署环境变量的配置真源，修正不可关闭的布尔参数。
4. 建立可追溯版本：Git commit、镜像 digest、wheel hash、源码 manifest、模型/demo hash。
5. 新增 R13 验收：首次参数门禁、断网、Learner 重启、Actor 重启、队列满、SIGINT drain、checkpoint 恢复、分类器故障、夹爪失败、20 次确定性评估和长时间 soak。

## 11. 最终判定

| 维度 | 判定 | 说明 |
|---|---|---|
| 论文算法主干对齐 | 基本通过 | RLPD 50/50、6D+1D hybrid、干预动作回放总体正确 |
| Learner 核心训练路径 | 有条件通过 | 已实际更新且无 NaN，但 demo 完整性、温度和恢复一致性需处理 |
| Actor 真机安全闭环 | 不通过 | 首次参数前可动作、夹爪失败语义与分类器异常策略不满足最终上线标准 |
| 跨端数据完整性 | 不通过 | 会话隔离、队列丢失、退出 flush、manifest 覆盖存在缺口 |
| 评估可复现性 | 不通过 | stochastic eval、checkpoint/provenance 校验不足 |
| Docker/交付可复现性 | 不通过 | compose 与构建镜像脱节，镜像不含完整 WA2 主链路 |
| 阶段基础模块 | 较好 | R1–R12 具有较完整方案和验收记录，可继续复用 |

**总体结论：当前工程适合作为已跑通主要训练链路的研发版本，但尚不适合作为无需额外约束即可交付和长期无人值守运行的最终版本。应先解决 P0，再进行 R13 全流程动态验收；P1 中的数据完整性、恢复和部署问题完成前，不建议将现有训练结果视为完全可复现的正式基线。**

---

本报告只进行检查和记录。除新增本 Markdown 文件外，未修改任何项目代码、配置、模型、buffer、运行日志或虚拟环境。
