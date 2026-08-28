# SpaceMouse 六自由度末端与灵巧手控制设计方案

## 1. 文档目的

本文给出 WA2 双臂机器人 SpaceMouse 遥操作的下一阶段设计。目标是在 0807 已完成的
“SpaceMouse → ROS Joy → ServoL → 左臂 BASE XYZ 平移”基础上，实现：

- 机械臂末端位置和姿态的六自由度连续控制；
- 平移、旋转指令的平滑处理，改善当前控制不流畅的问题；
- 通过 SpaceMouse 另一个按钮控制同侧灵巧手抓握和释放；
- 左、右机械臂可配置，但单次运行只控制其中一侧；
- 为后续更换末端执行器保留独立适配接口；
- 通过 dry-run、单轴、小角度、组合动作逐级完成实机验收。

本文是设计和验收方案，不代表旋转控制已经过真机验证。旋转轴的 Joy 下标、方向、ServoL
姿态跟随、奇异状态和碰撞风险必须按本文 Gate 顺序验收，不能直接执行大角度翻转。

## 2. 已有基础与当前限制

### 2.1 已验证链路

```text
SpaceMouse USB
  → spacenavd
  → spacenav_node
  → /spacenav/joy (sensor_msgs/Joy)
  → spacemouse_wa2_teleop.py
  → NaviController.servol()
  → /zj_humanoid/upperlimb/servol/{left,right}_arm
```

0807 已确认：

- SDK 1.3.2 没有可用的 `enable_speedl` / `speedl` 运行接口；
- 连续笛卡尔控制采用 ServoL 绝对位姿目标；
- `set_servo_params(time=0.02, gain=800, arm_type=1/2/3)` 可用；
- 左臂在非奇异位下完成 X、Y、Z 毫米级运动；
- Joy 频率通常约 50 Hz，偶尔约 30 Hz；
- 左键 `buttons[1]` 已作为 deadman；
- 右键 `buttons[0]` 当前未使用。

已标定的平移映射：

| 输出 | Joy轴 | sign |
|---|---:|---:|
| BASE X（前后） | `axes[0]` | `-1` |
| BASE Y（左右） | `axes[1]` | `-1` |
| BASE Z | `axes[2]` | `+1` |

旋转轴已于后续物理标定中确认：

| SpaceMouse 动作 | 主轴 | 原始方向 | 峰值样例 |
|---|---:|---:|---:|
| 左倾斜 | `axes[3]` | 正 | `+0.9766` |
| 右倾斜 | `axes[3]` | 负 | `-0.9766` |
| 前倾斜 | `axes[4]` | 负 | `-0.9766` |
| 后倾斜 | `axes[4]` | 正 | `+0.9766` |
| Z轴顺时针扭转 | `axes[5]` | 负 | `-0.9766` |
| Z轴逆时针扭转 | `axes[5]` | 正 | `+0.9766` |

因此设备侧旋转轴顺序确定为：

```text
Roll  ← axes[3]
Pitch ← axes[4]
Yaw   ← axes[5]
```

若先把“左倾、前倾、顺时针”定义为遥操输入的正方向，则 dry-run 初始符号可写为：

```text
angular_axis_map  = [3, 4, 5]
angular_axis_sign = [+1, -1, -1]
```

这里的符号只是 SpaceMouse 人机方向归一化，不代表 WA2 工具坐标系的最终正方向。最终 Rx/Ry/Rz
符号仍必须通过 ServoL 正负 2° 小角度实机 Gate 确认。

本次标定还发现明显的机械耦合/操作串扰：

- 左倾斜时 `axes[5]≈+0.31`；
- 后倾斜时 `axes[5]≈-0.35`；
- 前倾斜时 `axes[0]≈+0.26`；
- 右倾斜时 `axes[1]≈+0.39`；
- Z轴顺时针扭转时 `axes[2]≈-0.62`；
- Z轴逆时针扭转时 `axes[2]≈-0.89`。

其中扭转与 Z 平移的耦合已经远大于普通 deadzone，不能仅靠增加死区解决。第一版六自由度控制
必须采用“旋转优先 + 旋转主轴锁定”：检测到明确旋转意图时暂时抑制 XYZ，并只保留最强的旋转轴；
待完整收集正负 XYZ 操作样本后，再评估全六轴同时混控或标定矩阵解耦。

后续正负 XYZ 标定结果为：

| SpaceMouse 动作 | 主轴 | 主轴值 | 最大串扰样例 |
|---|---:|---:|---:|
| 左平移 | `axes[1]` | `-0.8809` | `axes[2]=-0.6602` |
| 右平移 | `axes[1]` | `+0.9766` | `axes[2]=-0.8477`、`axes[3]=+0.4531` |
| 前平移 | `axes[0]` | `-0.9766` | `axes[1]=+0.1465` |
| 后平移 | `axes[0]` | `+0.9766` | `axes[2]=-0.6289` |
| Z轴上平移 | `axes[2]` | `+0.9766` | `axes[5]=-0.2305` |
| Z轴下平移 | `axes[2]` | `-0.9766` | 其余轴较小 |

因此本机设备的六轴顺序和初始人机方向归一化确定为：

```text
axis_map  = [0, 1, 2, 3, 4, 5]
axis_sign = [-1, -1, +1, +1, -1, -1]

正X = 前平移
正Y = 左平移
正Z = 上平移
正Roll  = 左倾斜
正Pitch = 前倾斜
正Yaw   = 顺时针扭转
```

上述符号用于 SpaceMouse 操作习惯归一化；机器人 BASE/TOOL 最终正方向仍须通过小位移和正负 2°
ServoL 实机 Gate 确认。

平移标定进一步表明：平移组三轴之间也有明显串扰。左平移的 Z 串扰约为主轴的 75%，右平移
约为 87%，后平移约为 64%。因此第一版不仅旋转组需要主轴锁定，平移组也必须只保留最强主轴。
在完成多样本标定矩阵前，不开放斜向平移和真正的六轴同时混控。

### 2.2 当前脚本不足

现有 `spacemouse_wa2_teleop.py` 有以下限制：

1. 强制禁止旋转轴，`angular_scale` 必须为零；
2. 每周期只积分 XYZ，姿态始终复制 deadman 按下时的初始四元数；
3. 只有一个固定的左臂控制目标；
4. 没有灵巧手按钮状态机；
5. 只有 deadzone，没有低通滤波、响应曲线和加速度限制；
6. `tcp_watchdog` 参数存在，但控制器没有记录 TCP 回调时间，实际上不能判断反馈是否陈旧；
7. 没有在发送 ServoL 前程序化检查 `left/right_arm_is_singular`；
8. ServoL 与灵巧手 Service 如果放在同一循环同步执行，手部 Service 最长可能阻塞 5 秒；
9. 松开 deadman 后执行 `stop()`，再次按下时是否需要重新调用 `set_servo_params()` 尚需明确并验收；
10. 旋转后缺少相对角度限制、自碰撞约束和翻转过程的分段验证。

因此本阶段不建议只在现有 `_map_axes()` 后面简单增加三个角速度并立即实机运行，而应先拆分输入、
位姿积分、ServoL 会话和末端执行器控制职责。

## 3. 总体控制方案

### 3.1 单 SpaceMouse 控制一侧机械臂

WA2 是双臂机器人，但一只 SpaceMouse 只有六轴和两个按钮。本方案一次启动只控制一侧：

```text
~arm_side:=left   → 左臂 + 左手
~arm_side:=right  → 右臂 + 右手
```

原因：

- 两个按钮分别用于 deadman 和灵巧手，已经没有可靠的机械臂切换按钮；
- 运行中切换左右臂容易因目标位姿未重新初始化造成跳变；
- `servol/dual_arm` 的 DualPose 未使用字段语义尚未单独验证；
- 单臂逐侧验收更容易隔离轴映射、奇异和碰撞问题。

如以后确实需要在线切换左右臂，应增加独立键盘、脚踏开关或 ROS Service，并且只允许在 deadman
松开、两臂停止后切换；不能用 SpaceMouse 两键的组合或双击承担这一安全功能。

### 3.2 按钮定义

| 输入 | 行为 |
|---|---|
| 左键 `buttons[1]` 按住 | deadman；允许同侧机械臂六自由度控制 |
| 左键松开 | 立即停止 ServoL；清空滤波速度；下一次按下从最新实测 TCP 重新初始化 |
| 右键 `buttons[0]` 上升沿 | 在抓握与释放之间切换 |

右键命令必须满足：

- 左键 deadman 已松开；
- 六个 SpaceMouse 轴均处于中位区；
- 机械臂已经执行 `stop()`；
- 距离上一次手部指令超过冷却时间；
- 当前手部关节反馈有效，或配置明确指定了初始手部状态。

这样不允许“机械臂连续运动的同时调用灵巧手 Service”。虽然将来可能需要边移动边抓握，但当前
`joint_switch` 是同步 Service，最长等待 5 秒，直接在 50 Hz ServoL 主循环调用会破坏机械臂指令周期。

### 3.3 控制状态机

```text
INIT
  └─ 状态、Joy、TCP、手部反馈就绪 → IDLE

IDLE
  ├─ deadman 上升沿且安全检查通过 → ARM_ACTIVE
  ├─ 手部键上升沿且手柄回中 → HAND_COMMAND
  └─ 状态失联/奇异/错误 → FAULT

ARM_ACTIVE
  ├─ 50 Hz 积分并发布 ServoL
  ├─ deadman 松开/Joy超时/TCP超时 → stop → IDLE
  └─ 奇异/错误/状态失联 → stop + clear → FAULT

HAND_COMMAND
  ├─ 执行同侧抓握或释放 Service
  ├─ 成功后更新手部期望状态 → IDLE
  └─ 失败或超时 → FAULT 或 IDLE（按错误级别处理）

FAULT
  ├─ 保持机械臂停止，拒绝所有动作
  └─ 状态恢复后需要显式复位，不自动重新进入控制
```

## 4. 六自由度位姿积分

### 4.1 输出动作

SpaceMouse 经标定、死区、响应曲线和滤波后得到：

```text
v = [vx, vy, vz]       # m/s
w = [wx, wy, wz]       # rad/s
```

每个周期使用单调时钟计算 `dt`：

```text
delta_p = clip(v * dt, max_linear_step)
delta_r = clip(w * dt, max_angular_step)
```

位置更新：

```text
p_target = p_target + delta_p
```

姿态必须用旋转向量和四元数复合，不允许直接对四元数四个分量相加，也不建议用欧拉角做连续积分。

```python
dR = Rotation.from_rotvec(delta_r)

# 默认：绕当前末端自身坐标轴旋转
R_target = R_target * dR

# 可选：绕 BASE 固定坐标轴旋转
R_target = dR * R_target

q_target = R_target.as_quat()  # [qx, qy, qz, qw]
```

每次发布前应：

- 将四元数归一化；
- 检查全部元素为有限数；
- 保持四元数符号连续，若 `dot(q_new, q_old) < 0`，令 `q_new = -q_new`；
- 检查单周期角度和单次 deadman 累计角度；
- 通过 `NaviController.servol()` 的四元数合法性检查。

### 4.2 参考坐标系

推荐默认采用混合模式：

```text
平移：BASE frame
旋转：TOOL/TCP local frame
```

理由：

- 现有 XYZ 已按 BASE 方向标定，继续使用可降低迁移风险；
- 用户旋转 SpaceMouse 帽体时，让末端绕自身轴转动通常更符合“拧、翻、倾斜”的直觉；
- 对灵巧手抓住物体后的姿态调整，工具坐标旋转更容易理解。

仍应保留参数：

```text
~translation_frame:=base
~rotation_frame:=tool   # tool | base
```

只有完成两种模式的 dry-run 和小角度验证后，才允许修改运行参数。

### 4.3 翻转动作

“翻转”不是一个额外的离散动作，而是持续输入某个或多个角速度轴，使目标姿态逐渐复合。第一阶段按
以下顺序开放：

1. 单轴正负 2°；
2. 单轴正负 10°；
3. 单轴正负 30°；
4. 单轴 90° 分段翻转；
5. 确认关节余量、自碰撞和线缆安全后，再评估 180°。

第一版累计角度限制建议为每次 deadman 最多 30°。完成 30° 和 90° Gate 后再将配置上限放宽，
不能初始就把上限设为 180°。累计限制使用初始姿态与目标姿态之间的四元数测地角，而不是欧拉角差。

## 5. 平滑控制设计

当前“不流畅”可能来自 Joy 噪声、30/50 Hz 抖动、死区边缘跳变、每周期 dt 波动和目标积分方式。
建议按以下顺序处理。

### 5.1 连续死区映射

保留当前连续 deadband，使超过死区后的输出从零连续增长：

```text
u_deadband = sign(u) * (abs(u) - deadzone) / (1 - deadzone)
```

平移和旋转应允许不同死区：

```text
linear_deadzone  = 0.10 ~ 0.15
angular_deadzone = 0.12 ~ 0.20
```

旋转通常更容易受平移操作串扰，因此旋转死区应稍大。

### 5.2 响应曲线

对 deadband 后的归一化输入使用线性与三次曲线混合：

```text
u_shaped = (1 - k) * u + k * u^3
```

建议初值：

```text
linear_curve_mix  = 0.25
angular_curve_mix = 0.45
```

这样小输入更精细，满推仍能达到配置的最大速度。旋转比平移使用更强的曲线，方便小角度对准。

### 5.3 一阶低通滤波

使用与实际 `dt` 相关的一阶低通：

```text
alpha = 1 - exp(-dt / tau)
y = y + alpha * (x - y)
```

建议初值：

```text
linear_filter_tau  = 0.06 s
angular_filter_tau = 0.10 s
```

不要用固定窗口平均，因为 Joy 频率在 30 Hz 和 50 Hz 之间变化时，固定窗口会改变真实延迟。

### 5.4 速度变化率限制

在滤波后增加速度变化率限制，防止手柄突然满推导致目标速度阶跃：

```text
|dv/dt| <= max_linear_accel
|dw/dt| <= max_angular_accel
```

建议保守初值：

```text
max_linear_accel  = 0.10 m/s²
max_angular_accel = 1.0 rad/s²
```

deadman 松开、Joy 超时或 Fault 时不应等待滤波慢慢归零，而应立即停止 ServoL 并清空滤波状态。

### 5.5 固定频率目标流

ServoL 发布循环保持 50 Hz。Joy 回调只更新最新输入和时间戳，不在回调中发布机器人命令。

```text
Joy callback：保存 axes/buttons/stamp
50 Hz loop：读取快照 → 安全检查 → 滤波 → 积分 → 发布
```

即使 Joy 临时从 50 Hz 降到 30 Hz，只要没有超过 watchdog，ServoL 仍按 50 Hz 发布最新计算的目标；
超过 watchdog 则立即停止，不能继续沿用旧输入。

### 5.6 旋转意图判定与串扰抑制

根据本机标定结果，第一版不能把六个原始轴简单同时乘比例后输出。建议增加：

```text
rotation_strong_enter    = 0.65
group_exit_threshold     = 0.20
translation_enter        = 0.35
group_switch_hysteresis  = 0.15
dominant_axis_hysteresis = 0.10
```

处理逻辑：

1. 分别计算 `t_max=max(abs(axes[0:3]))` 和 `r_max=max(abs(axes[3:6]))`；
2. 当 `r_max >= 0.65` 时进入 `ROTATION_INTENT`。实测所有明确旋转动作主轴约为 `0.9766`，
   而平移产生的最大旋转串扰为 `0.4531`，该阈值能够分开现有样本；
3. 未满足旋转强触发且 `t_max >= 0.35` 时进入 `TRANSLATION_INTENT`；
4. `ROTATION_INTENT` 中将 XYZ 输出置零，只保留 `axes[3:6]` 中最强的旋转轴；
5. `TRANSLATION_INTENT` 中将旋转输出置零，只保留 `axes[0:3]` 中最强的平移轴；
6. 从旋转直接切向平移时，平移主峰必须至少比旋转主峰大 `0.15`，避免扭转产生的 Z 串扰
   错误抢占；
7. 组内主轴切换采用迟滞：新轴必须比当前轴至少大 `0.10` 才允许接管。右平移实测 Y 主轴
   比 Z 串扰大约 `0.129`，因此不能使用原先过大的 `0.15` 组内迟滞；
8. 当前组主轴低于 `0.20` 后退出意图状态，避免阈值附近快速切换；
9. 模式切换和 deadman 松开时清空滤波状态。

这套规则对当前12个正负方向标定样本均能作出正确的平移/旋转分组，并能抑制已观察到的最大
串扰。阈值仍需结合静止状态连续样本确定最终裕量。

该策略优先保证翻转和姿态调整安全，代价是第一版每次只输出一个主轴，不能斜向平移或同时平移
和旋转。完成多轮、不同力度的六轴样本采集后，可再比较以下升级路线：

- 保留意图分离，但允许旋转组三轴混合；
- 使用完整 6×6 标定矩阵做线性解耦；
- 增加独立模式开关，在平移模式和旋转模式之间显式切换。

## 6. 灵巧手抓握与释放

### 6.1 当前固定目标

参考 `naviai_controller/scripts/test_hand.py`：

```python
GRASP_TARGET = [0.1, 1.5, 1.2, 1.2, 1.2, 1.2]
RELEASE_TARGET = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
```

同侧映射：

```text
arm_side=left  → ArmGroup.LEFT  + HandType.LEFT
arm_side=right → ArmGroup.RIGHT + HandType.RIGHT
```

抓握调用：

```python
ctrl.grasp_hand(selected_hand, grasp_target)
```

释放调用：

```python
ctrl.release_hand(selected_hand)
```

### 6.2 右键切换逻辑

只在 `buttons[0]` 从 0 变成 1 的上升沿触发一次，禁止按住时重复调用 Service。

```text
当前期望为 RELEASED → 发送抓握目标 → 成功后记为 GRASPED
当前期望为 GRASPED  → 发送释放目标 → 成功后记为 RELEASED
```

建议参数：

```text
~hand_button:=0
~hand_debounce:=0.30
~hand_cooldown:=1.00
~hand_neutral_threshold:=0.08
~grasp_target:="[0.1,1.5,1.2,1.2,1.2,1.2]"
~release_target:="[0,0,0,0,0,0]"
```

Service 返回失败时不能切换内部状态。日志应记录：侧别、命令、目标、调用耗时、返回值和调用后的
手部关节反馈。

### 6.3 启动时手部状态

启动时读取 `get_hand_joints(selected_hand)`，分别计算它到抓握目标和释放目标的距离：

```text
d_grasp  = ||q_hand - q_grasp||
d_release = ||q_hand - q_release||
```

- 明显更接近释放目标：下一次按键执行抓握；
- 明显更接近抓握目标：下一次按键执行释放；
- 处于中间状态或反馈无效：拒绝按钮切换，要求通过参数明确 `initial_hand_state`，或先单独执行释放。

不能默认认为启动时手一定是张开的，否则脚本重启后第一次按键可能重复抓握。

### 6.4 末端执行器可更换设计

不要把 `grasp_hand()` 直接散落在 SpaceMouse 主循环中。定义独立接口：

```python
class EndEffectorAdapter:
    def wait_until_ready(self, timeout): ...
    def open(self) -> bool: ...
    def close(self) -> bool: ...
    def get_state(self): ...
```

当前实现：

```text
DexterousHandAdapter
  ├─ open()  → release_hand()
  └─ close() → grasp_hand(fixed_target)
```

以后更换夹爪时只新增适配器，例如 `ParallelGripperAdapter`，SpaceMouse 状态机和按钮逻辑不变。

## 7. 软件结构建议

建议逐步将现有单文件拆为：

```text
hilserl_wa2/interventions/
├── spacemouse_input.py          # Joy缓存、按钮边沿、轴标定
├── pose_integrator.py           # deadband、曲线、滤波、SE(3)积分和限幅
├── end_effector.py              # 灵巧手/未来夹爪统一适配器
├── spacemouse_wa2_teleop.py     # 状态机、参数、日志、生命周期
└── tests/
    ├── test_pose_integrator.py
    ├── test_button_state.py
    └── test_end_effector_fake.py
```

机器人控制基础层还应补充：

```text
hilserl_wa2/ros_adapters/
├── wa2_state_monitor.py         # TCP/关节/uplimb_state及时间戳
└── wa2_servo_session.py         # ServoL进入、50 Hz发布、stop、clear
```

主要职责：

| 模块 | 不应该承担的职责 |
|---|---|
| `SpaceMouseInput` | 不调用机器人 Service/Publisher |
| `PoseIntegrator` | 不依赖 ROS，不读取硬件 |
| `WA2StateMonitor` | 不修改目标、不发运动命令 |
| `WA2ServoSession` | 不解释按钮，不控制灵巧手 |
| `EndEffectorAdapter` | 不发送 ServoL |
| `SpaceMouseWA2Teleop` | 只编排状态机和生命周期 |

这种结构也方便后续 HIL-SERL intervention Wrapper 复用输入映射和动作整形，而不复用实机遥操循环。

## 8. 参数建议

### 8.1 首次旋转实机档

| 参数 | 建议初值 | 硬上限建议 |
|---|---:|---:|
| `publish_rate` | `50 Hz` | 固定为已验证周期 |
| `linear_scale` | `0.010 m/s` | `0.03 m/s` |
| `angular_scale` | `0.15 rad/s` | 初期 `0.30 rad/s` |
| `max_linear_step` | `0.0005 m` | `0.002 m` |
| `max_angular_step` | `0.003 rad` | 初期 `0.01 rad` |
| `translation_workspace` | `0.03 m` | `0.10 m` |
| `rotation_workspace` | `10°` | 初期 `30°` |
| `joy_watchdog` | `0.25 s` | 不建议放宽 |
| `tcp_watchdog` | `0.25 s` | 必须真实生效 |

### 8.2 日常六自由度档（完成 Gate 后）

```text
linear_scale       = 0.015 m/s
angular_scale      = 0.30 rad/s
linear_deadzone    = 0.12
angular_deadzone   = 0.16
max_linear_step    = 0.001 m
max_angular_step   = 0.008 rad
translation_limit  = 0.05 m per deadman
rotation_limit     = 30 deg per deadman
translation_frame  = base
rotation_frame     = tool
```

90°/180° 翻转不应通过单纯加大 `angular_scale` 实现，而应在低角速度下逐步扩大累计角度限制，并持续
监控关节、奇异、自碰撞和线缆状态。

## 9. 必须补齐的安全门禁

六自由度开放前至少完成：

1. 在 `ArmController` 或独立 StateMonitor 中记录 TCP、关节、`UplimbState` 的最后回调时间；
2. `tcp_watchdog`、state watchdog 真正按时间戳判断；
3. 根据 `arm_side` 检查对应的 `left/right_arm_is_singular`；
4. 校验当前 TCP、目标 TCP、四元数和手部目标均为有限数；
5. 限制单周期平移、单周期旋转、累计平移和累计旋转；
6. deadman 松开、Joy超时、TCP超时、ROS shutdown、异常时调用 `stop()`；
7. 严重 Fault 时再调用 `clear_servo_params()`，并要求人工复位；
8. 遥操节点启动前确认没有其他 ServoL、MoveL、Actor 或 `assembly` 控制源；
9. 实机测试时物理急停可达，先清空灵巧手和机械臂周围空间；
10. 旋转验收期间不抓持重物，先使用空手或轻质软物体；
11. 没有可靠碰撞检测前，不在身体、桌面和另一条机械臂附近做大角度翻转；
12. 手部按钮触发时禁止机械臂动作，Service失败不自动重试。

## 10. 分阶段开发与验收

### Gate 0：纯软件数学测试

不连接 ROS 和机器人：

- 六轴长度、NaN、Inf、非法参数检查；
- deadband 在阈值处连续；
- 滤波在 30 Hz/50 Hz 输入下结果接近；
- 零角速度不会改变四元数；
- 单轴正负旋转方向符合定义；
- 连续积分后四元数范数保持 1；
- 180°附近没有欧拉角跳变；
- base/tool 左乘与右乘结果符合预期；
- 单步和累计角度限制有效；
- deadman 松开清空滤波状态；
- 右键按住只产生一次手部事件；
- Service失败不改变手部期望状态。

### Gate 1：物理 SpaceMouse 六轴标定（无机器人动作）

只读取 `/spacenav/joy`：

- 单独扭动/倾斜每个旋转方向；
- 记录实际 `axes[3]`、`axes[4]`、`axes[5]` 对应关系；
- 确定每个轴正方向；
- 测量静止噪声和操作时轴间串扰；
- 据此复核已确定的 `angular_axis_map`、初始 `angular_axis_sign`，并确定 `angular_deadzone` 和串扰抑制阈值。

旋转和平移轴顺序及操作方向已经确认；下一步还需记录：

```text
静止状态连续 5~10 组样本
```

用于确定各轴静止噪声、最终 deadzone 和意图退出阈值。现有方向标定结果必须写入调试日志，
不能只保存在启动命令历史中。

### Gate 2：六自由度 dry-run

`execute=false`，输出：

```text
raw_axes
filtered_linear_velocity
filtered_angular_velocity
target_xyz
target_quaternion
relative_translation_mm
relative_rotation_deg
deadman/hand_button/state
```

验收：

- 不按 deadman，目标不变化；
- 平移操作不会引起超过死区的旋转；
- 旋转操作不会引起超过死区的平移；
- 松手后立即冻结目标；
- 右键上升沿只输出一次 GRASP/RELEASE 预览，不调用手部 Service。

### Gate 3：ServoL 零动作保持

保持当前实测 TCP 姿态，六轴输入均为零，以 50 Hz 发布 1 秒：

- TCP 无明显漂移；
- `cmd_num=14`；
- 对应手臂非奇异；
- stop 后无继续运动；
- 重复两次“进入 ServoL → stop → 再进入”以验证是否每次需要重新设置伺服参数。

### Gate 4：单旋转轴小角度实机

一次只开放一个旋转轴，平移全部关闭：

1. 正向约 2°；
2. 反向回到起点附近；
3. 反向约 2°；
4. 对 Rx、Ry、Rz 分别执行；
5. 计算实测 TCP 相对旋转角和旋转轴，不只依靠目测。

通过后逐步扩大到 10°、30°。任何轴方向错误、明显耦合、抖动或奇异都应立即停止并回到 dry-run。

### Gate 5：三旋转轴与六自由度组合

- 先启用三个旋转轴，不启用平移；
- 再启用 XYZ + RxRyRz；
- 验证斜向平移、倾斜、绕工具轴旋转；
- 逐步调节滤波时间常数和响应曲线；
- 每组参数记录 Joy 频率、目标变化、实测 TCP 和主观手感。

### Gate 6：灵巧手独立按钮

机械臂停止、空手测试：

- 根据 `arm_side` 选择同侧手；
- 第一次右键发送固定抓握目标；
- 验证 Service 返回和6维手部反馈；
- 第二次右键释放为六个零；
- 按住右键不得重复触发；
- 快速连按受 cooldown 限制；
- deadman 按下或手柄不在中位时，右键命令被拒绝并打印原因。

### Gate 7：机械臂与灵巧手组合流程

推荐验收序列：

```text
空手移动到目标附近
  → 松开 deadman，机械臂 stop
  → 右键抓握
  → 检查手部反馈
  → 按住 deadman，低速移动/倾斜
  → 松开 deadman
  → 右键释放
```

先使用轻质软物体；在组合流程稳定前不执行 90° 以上翻转。

### Gate 8：翻转验收

依次进行 30°、60°、90°，每阶段检查：

- 左右臂关节余量；
- 对应手臂奇异状态；
- 灵巧手抓握是否滑移；
- 腕部线缆是否缠绕；
- 是否接近机器人身体、桌面或另一机械臂；
- 松开 deadman 后是否立即停止。

只有 90° 全流程重复稳定后，才讨论 180° 翻转和更高角速度。

## 11. 建议实施顺序

1. 增加机器人状态时间戳和真实 `tcp_watchdog`；
2. 暴露并检查左右臂奇异状态；
3. 抽出纯 Python `PoseIntegrator`，完成四元数、滤波和限幅单元测试；
4. 增加 `arm_side`，先保持 XYZ 功能不变，分别验证左右臂；
5. 完成物理 SpaceMouse 旋转轴 dry-run 标定；
6. 开放单旋转轴、小角度 ServoL；
7. 完成三轴旋转和六自由度平滑控制；
8. 实现 `EndEffectorAdapter` 和右键边沿状态机；
9. 完成灵巧手独立与组合 Gate；
10. 最后开展分段翻转和参数手感优化。

## 12. 本阶段完成标准

- 左、右臂可通过参数选择，启动后不能隐式切换；
- 六个 SpaceMouse 轴均经过物理标定；
- XYZ 和姿态控制连续、无明显抖动或突然跳变；
- 姿态使用四元数/旋转向量复合，不使用欧拉角直接积分；
- deadman、Joy watchdog、TCP watchdog、奇异检查真实生效；
- 右键每次只触发一次同侧灵巧手抓握或释放；
- 灵巧手 Service 不阻塞 50 Hz ServoL 循环；
- 抓握/释放固定角度可通过参数配置；
- 任一失联、异常或退出路径都能停止机械臂；
- 30° 旋转和机械臂—灵巧手组合流程可重复通过；
- 90°/180° 翻转只有在对应 Gate 通过后才开放。

## 13. 相关文件

- 调试基线：`调试日志/0807调试日志.md`
- 当前遥操：`catkin_ws/src/hilserl_wa2/interventions/spacemouse_wa2_teleop.py`
- ServoL 封装：`catkin_ws/src/naviai_controller/src/naviai_controller/core/arm.py`
- 灵巧手封装：`catkin_ws/src/naviai_controller/src/naviai_controller/core/hand.py`
- 灵巧手示例：`catkin_ws/src/naviai_controller/scripts/test_hand.py`
- 当前使用指南：`docs/SpaceMouse使用指南.md`

## 14. 步骤 5～8 实施结果（2026-08-10）

本阶段已经完成以下代码接入：

- 现有遥操脚本已接入 `SpaceMouseInputProcessor` 与 `PoseIntegrator`；
- 保留已确认的轴配置：`axis_map=[0,1,2,3,4,5]`、
  `axis_sign=[-1,-1,1,1,-1,-1]`；
- 支持 `left/right` 手臂选择、工具/基座旋转坐标系、单轴屏蔽和累计角度限幅；
- deadman 上升沿以最新实测 TCP 作为本次控制原点，释放后执行 stop/clear；
- 增加 TCP、UplimbState 时间戳 watchdog、奇异状态检查和故障锁存；
- 曾增加 rotation-only 合成输入 dry-run 和带双重确认的 ServoL 实机 Gate；逻辑验收完成后，
  这些一次性脚本已于 2026-08-11 删除；
- 当前保留的离线回归结果：`Ran 27 tests ... OK`。

涉及文件：

- `catkin_ws/src/hilserl_wa2/interventions/spacemouse_wa2_teleop.py`
- `catkin_ws/src/hilserl_wa2/interventions/spacemouse_input.py`
- `catkin_ws/src/hilserl_wa2/interventions/pose_integrator.py`
- `catkin_ws/src/hilserl_wa2/interventions/end_effector.py`
- `catkin_ws/src/naviai_controller/src/naviai_controller/core/arm.py`
- `catkin_ws/src/naviai_controller/src/naviai_controller/naviai_controller.py`

### 14.1 Orin 容器内准备

在 `hilserl` 容器的 `hil-actor` 环境中执行：

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate hil-actor
source /ros_noetic/catkin_ws/devel/setup.bash
source /opt/ros/noetic/setup.bash --extend
source /root/catkin_ws/devel/setup.bash
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=/tmp/hilserl-pycache
cd /root/catkin_ws
```

先复验离线测试：

```bash
python -m unittest discover -s src/hilserl_wa2/tests/unit -v
python -m compileall -q src/hilserl_wa2 src/naviai_controller/src/naviai_controller
```

验收：27 项测试全部通过，两个命令退出码均为 0。

### 14.2 步骤 6A：合成输入 rotation-only dry-run

该 Gate 已于 2026-08-10 完成，所用一次性合成输入脚本已于 2026-08-11 删除。结果保留如下：

逐条验收：

- 末行分别为 `ROTATION-ONLY DRY-RUN ROLL/PITCH/YAW: PASS`；
- `relative_rotation_deg` 在 1.95°～2.0001°；
- `position_delta_mm` 不大于 0.000001 mm；
- 不按 deadman 时目标不变，释放后目标冻结；
- 机器人实际不运动。

后续逻辑回归由 `tests/unit` 中保留的姿态积分和输入映射测试承担；需要重新进行物理 dry-run 时，
使用下一节的最终遥操脚本和单轴 `axis_enable`，不要恢复旧的一次性测试文件。

### 14.3 步骤 6B：物理 SpaceMouse rotation-only dry-run

该 Gate 用于确认真实设备的方向和串扰，仍然保持 `execute=false`。运行
`spacenavd` 和 `spacenav_node` 后，一次只开放一个旋转轴。`hilserl_wa2`
当前不是 catkin package，因此直接使用 Python 入口。例如左臂 Roll：

```bash
python src/hilserl_wa2/interventions/spacemouse_wa2_teleop.py \
  _arm_side:=left \
  _axis_map:="[0,1,2,3,4,5]" \
  _axis_sign:="[-1,-1,1,1,-1,-1]" \
  _axis_enable:="[0,0,0,1,0,0]" \
  _linear_scale:=0.0 \
  _angular_scale:=0.08 \
  _rotation_limit_deg:=2.0 \
  _rotation_frame:=tool \
  _execute:=false
```

随后将 `axis_enable` 依次改为 Pitch 的 `[0,0,0,0,1,0]` 和 Yaw 的
`[0,0,0,0,0,1]`。验收方向为：左倾为 +Roll、前倾为 +Pitch、顺时针为
+Yaw；纯旋转时 XYZ 目标不得变化，释放 deadman 后目标立即冻结。

### 14.4 步骤 7：±2° ServoL 实机 Gate

实机前必须满足：急停可触及、工作区无人、无其他节点控制同一机械臂、TCP 和
UplimbState 持续更新、对应手臂 `is_singular=False`。建议另开一个终端准备：

```bash
rosservice call /zj_humanoid/upperlimb/stop "{}"
```

该 Gate 已于 2026-08-10 使用一次性合成输入脚本完成；脚本已于 2026-08-11 删除。验收标准为：

- 末行是 `SERVOL ROTATION ROLL POSITIVE 2DEG: PASS`；
- `commanded_rotation_deg` 与 2° 的误差不超过 0.05°；
- `actual_rotation_deg` 与 2° 的误差不超过 1°，且方向正确；
- `off_axis_rotation_deg` 不超过 1°；
- `position_drift_mm` 不超过约 2.07 mm；
- 测试后机械臂仍为非奇异状态，释放后不继续运动。

实测 +Roll 为 2.0019°、位置漂移 0.0234 mm；-Roll 为 2.0027°、位置漂移
0.0328 mm，两项均 PASS。完整数据见 `调试日志/0810调试日志.md`。

### 14.5 步骤 8：逐级扩大翻转角度

只有三个旋转轴的 ±2° 都通过后，才按 `10° → 30° → 60° → 90°` 逐级测试。
每一级必须先正向、再反向，并复查机器人空间、关节余量和线缆。使用最终遥操脚本调整
`rotation_limit_deg`，当前核心拒绝超过 90° 的累计角度。最新启动命令和参数范围以
`docs/SpaceMouse使用指南.md` 为准。60° 和 90°必须在审阅 30° 数据后再开展；本阶段不开放
180° 翻转。

## 15. 真实 SpaceMouse 与灵巧手按钮组合实现（2026-08-10）

### 15.1 已实现行为

- 左键 `buttons[1]`：机械臂 deadman；
- 右键 `buttons[0]`：同侧灵巧手抓握/释放切换；
- `arm_side=left/right` 自动选择 `HandType.LEFT/RIGHT`；
- 抓握目标固定为 `[0.1,1.5,1.2,1.2,1.2,1.2]`，释放目标为六个零；
- 右键只在上升沿触发，按住不会重复调用；
- 右键只在 deadman 释放、SpaceMouse 回中、ServoL 已停止、状态健康且冷却结束时接受；
- 手部 Service 在后台线程调用，不阻塞 50 Hz ServoL 循环；
- Service 失败时不切换内部抓握状态；手部命令执行期间禁止启动机械臂；
- `execute` 和 `hand_execute` 分别控制机械臂、灵巧手是否真机执行。

新增文件：`catkin_ws/src/hilserl_wa2/interventions/end_effector.py`。离线测试增加到
27 项。

### 15.2 左手独立初始化

先查看左手反馈：

```bash
python src/naviai_controller/scripts/test_hand.py --hand left --action status
```

组合实验前建议明确释放左手：

```bash
python src/naviai_controller/scripts/test_hand.py --hand left --action release
```

检查区域安全后输入确认口令 `RELEASE_LEFT`。必须看到 `release_hand 返回: True`，并确认
六维反馈接近零。

### 15.3 物理 SpaceMouse 全功能预览（机械臂和手都不动作）

```bash
python src/hilserl_wa2/interventions/spacemouse_wa2_teleop.py \
  _arm_side:=left \
  _axis_map:="[0,1,2,3,4,5]" \
  _axis_sign:="[-1,-1,1,1,-1,-1]" \
  _axis_enable:="[1,1,1,1,1,1]" \
  _deadman_button:=1 _hand_button:=0 \
  _linear_scale:=0.005 _angular_scale:=0.04 \
  _max_step_m:=0.0003 _max_angular_step:=0.001 \
  _workspace_m:=0.02 _rotation_limit_deg:=2.0 \
  _rotation_frame:=tool \
  _initial_hand_state:=released \
  _execute:=false _hand_execute:=false
```

验收：左键按住时六个方向分别生成正确目标；机器人不动。松开左键并将手柄完全回中后，
第一次按右键打印 `grasp requested` 和 `dry_run=True`，松开一秒后第二次按右键打印
`release requested`；灵巧手实际不动。左键和右键同时按时，右键必须被拒绝。

### 15.4 灵巧手按钮独立实机

确认左手已经释放后，沿用 15.3 命令，只修改：

```text
_initial_hand_state:=released _execute:=false _hand_execute:=true
```

机械臂不会动作。手柄回中、两个按钮都松开，然后短按一次右键；验收
`hand command=grasp success=True dry_run=False`，反馈向抓握目标变化。再次短按右键前至少等待
一秒；第二次验收 `hand command=release success=True`，反馈回到六个零附近。

### 15.5 六自由度机械臂与灵巧手组合真机

15.3 和 15.4 均通过后，使用与 15.3 相同的完整命令，只修改：

```text
_initial_hand_state:=released _execute:=true _hand_execute:=true
```

第一阶段保留 2°/20 mm 限幅和低速参数。标准操作序列：

```text
按住左键 → 移动或旋转机械臂 → 松开左键 → 等待 stop=True clear=True
→ SpaceMouse 回中 → 短按右键抓握 → 等待 success=True
→ 按住左键继续移动 → 松开左键 → 回中 → 短按右键释放
```

当前串扰抑制策略一次只接受一个主导自由度；六个自由度都可控制，但第一阶段不允许同时平移
和旋转。组合流程重复通过后，再逐级把 `rotation_limit_deg` 调到 10/30°，不得直接跳到
60/90°。
