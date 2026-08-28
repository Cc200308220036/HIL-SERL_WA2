# WA2Env 接口契约（R1 冻结）

> version: **0.1.1**（R6：腕相机启用）  
> 日期：2026-08-12  
> 机器可读副本：`configs/experiments/wa2_env_contract.yaml`  
> 风险：契约；真机图像见 R6；动作仍仅经 WA2Env

---

## 1. 范围

| 项 | 冻结值 |
|---|---|
| 臂 | **left** |
| action | **6D**（7D 仅附录） |
| 控制周期 | 50 Hz / 0.02 s |
| 连续控制后端 | ServoL（`servo_time=0.02`, `servo_gain=800`） |
| 图像键 | `head` + `wrist`（双键固定） |
| 唯一动作出口 | **WA2Env**（R4 起强制；R1 只写入契约） |

---

## 2. Action

```text
shape = (6,)
dtype = float32
range = [-1, 1]
semantics = [dx, dy, dz, droll, dpitch, dyaw]
```

物理缩放（每步，Env 内二次裁剪）：

| 分量 | 映射 |
|---|---|
| 平移 | `delta_xyz_m = action[:3] * 0.001`（最大 1.0 mm/step） |
| 旋转 | `delta_rpy_deg = action[3:] * 0.25`（最大 0.25°/step） |

坐标系（与已验收 SpaceMouse 遥操对齐）：

| 通道 | frame |
|---|---|
| 平移增量 | **base**（加到 TCP xyz） |
| 旋转增量 | **tool**（右乘局部旋转） |

NaN/Inf/错误 shape → 拒绝该步，不发动作。

### 附录：7D（未启用）

```text
[dx, dy, dz, droll, dpitch, dyaw, hand]
```

`hand` 语义（连续 / 三态 / 边沿）待 6D 稳定后单独冻结；禁止沿用按钮 toggle 隐式语义。

---

## 3. Observation

### 3.1 state

| 键 | shape | dtype | 单位 | 来源 | 实测 hz |
|---|---|---|---|---|---|
| `tcp_pose` | `[7]` | float32 | m + quat **xyzw** | `/zj_humanoid/upperlimb/tcp_pose/left_arm` (`upperlimb/Pose`) | ~125 |
| `tcp_vel` | `[6]` | float32 | m/s + rad/s | `/zj_humanoid/upperlimb/tcp_speed/dual_arm` 字段 `left_arm` | ~125 |
| `joint_pos` | `[8]` | float32 | rad | `/zj_humanoid/upperlimb/joint_states` 左臂 8 关节 | ~125 |
| `hand_joints` | `[6]` | float32 | rad | `/zj_humanoid/hand/joint_states` 左手 6 维 | ~200 |

左臂关节名顺序：

```text
Chest_Z_L, Shoulder_Y_L, Shoulder_X_L, Shoulder_Z_L,
Elbow_L, Wrist_Z_L, Wrist_Y_L, Wrist_X_L
```

左手关节名顺序：

```text
THUMB_MP_LEFT, THUMB_CMC_LEFT, INDEX_MCP_LEFT,
MIDDLE_MCP_LEFT, RING_MCP_LEFT, LITTLE_MCP_LEFT
```

安全/诊断进 `info`（默认不进策略观测）：

```text
is_singular (= left_arm_is_singular), state_age, image_age, image_ages,
cmd_num, cmd_name, iddp_status
```

新鲜度门控：`state_max_age_s = 0.2`，`image_max_age_s = 0.2`（头/腕均适用）。

### 3.2 images

| 键 | topic | enabled | raw | obs shape | 说明 |
|---|---|---|---|---|---|
| `head` | `/zj_humanoid/sensor/realsense_head/color/image_raw` | **true** | 720×1280×3，`rgb8`，~30 Hz | `uint8[128,128,3]` | ROS 订阅；crop+resize；obs 固定 RGB |
| `wrist` | `/zj_humanoid/sensor/left_wrist/image_raw` | **true** | 720×1280×3，`bgr8`，~30 Hz | `uint8[128,128,3]` | R6 启用；适配器转 RGB |

`fake_env` / 未订流占位：`missing_policy = zero_image`。  
真机路径：`head` 或 `wrist` stale → `truncated=True`（运动模式另 stop+clear）。

---

## 4. Step / Reset / Close

```text
reset() -> observation, info
step(action) -> observation, reward, terminated, truncated, info
close() -> stop + clear_servo_params + 释放资源（可重复调用）
```

| 符号 | 冻结定义 |
|---|---|
| `reward` | 恒 0（占位；未接 classifier） |
| `terminated` | 任务成功/失败占位，第一版恒 `False` |
| `truncated` | `max_steps` / 状态超时 / **头或腕**图超时 / 安全 / 人工中止 |
| 异常路径 | 先 `stop`，再抛异常或返回 truncated |

### reset（选定方案 C）

- 策略：`manual_pose_tolerance`（人工摆到 episode 起始姿态，Env 只做容差校验）
- 容差：关节 `0.05 rad`；TCP 位置 `0.01 m`；姿态 `5°`
- 必须人工确认：急停可达、线缆、双臂间隙、`is_singular=False` 后再开始 episode
- 附录自动化候选（R5）：固定关节 home + MoveJ  
- 失败/超时：`stop`，不进入 episode

---

## 5. 控制与安全生命周期

```text
set_servo_params  /zj_humanoid/upperlimb/set_servo_params
servol            /zj_humanoid/upperlimb/servol/left_arm
stop              /zj_humanoid/upperlimb/stop
clear             /zj_humanoid/upperlimb/clear_servo_params
```

1. 正式链路中 **仅 WA2Env** 发布 ServoL（禁止与遥操脚本并行）。
2. SpaceMouse → Env action / `intervene_action`，不得直接 ServoL。
3. `close()` / 异常 / Ctrl+C：`stop` → `clear_servo_params`。
4. stale / singular：不发新动作。
5. 网络断开不得重复旧动作（R9/R10 细化）。

---

## 6. fake_env（Learner）

- 不 import/初始化 rospy，不打开相机
- `images/head` 与 `images/wrist` 同为 `uint8[128,128,3]` 零图
- observation/action space 与真实 Env **逐字段一致**

---

## 7. Gymnasium space 摘要（供 Actor/Learner 对齐）

```text
action_space = Box(-1, 1, (6,), float32)

observation_space = Dict({
  "state": Dict({
    "tcp_pose":    Box(..., shape=(7,), dtype=float32),
    "tcp_vel":     Box(..., shape=(6,), dtype=float32),
    "joint_pos":   Box(..., shape=(8,), dtype=float32),
    "hand_joints": Box(..., shape=(6,), dtype=float32),
  }),
  "images": Dict({
    "head":  Box(0, 255, (128, 128, 3), uint8),
    "wrist": Box(0, 255, (128, 128, 3), uint8),
  }),
})
```

---

## 8. 证据索引（清理后最小集）

| 内容 | 路径 |
|---|---|
| topic 总清单 | `调试日志/阶段验收日志/R1_ROS接口盘点.md` |
| 状态/手/速度样本 | `调试日志/阶段验收日志/r1_samples/{tcp_pose_left_arm,tcp_speed_dual_arm,joint_states,uplimb_state,hand_joint_states}.txt` |
| hz 摘要 | `调试日志/阶段验收日志/r1_samples/hz_summary.md` |
| 头相机 | `调试日志/阶段验收日志/r1_samples/head/{head_image_meta.txt,head_sample.png}` |
| 手腕未启动 | `调试日志/阶段验收日志/r1_samples/wrist/{STATUS.md,info_left_wrist.txt}` |
| 验收记录 | `调试日志/阶段验收日志/2026-08-11_R1验收.md` |

样本采集时 `left_arm_is_singular=True` 仅反映当时静止姿态，不改变契约字段定义；真机 step 前仍须门控 `is_singular=False`。
