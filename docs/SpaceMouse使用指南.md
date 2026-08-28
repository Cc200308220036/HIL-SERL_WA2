# SpaceMouse 使用指南（WA2 六自由度 ServoL + 灵巧手）

## 1. 启动指令输入流程

本节是日常启动入口。必须按终端 A → B → C → D 的顺序执行；不要把多条前台进程命令粘贴到
同一个终端。

### 1.1 启动前检查

- 物理急停可触及；
- 左臂、末端、线缆、桌面和另一条机械臂之间有足够空间；
- 没有其他节点控制同一条机械臂；
- Orin 和 `hilserl` 容器中的机器人 ROS/SDK 已正常启动；
- 当前指南默认控制左臂和左灵巧手。

每个容器终端先进入环境：

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate hil-actor
source /ros_noetic/catkin_ws/devel/setup.bash
source /opt/ros/noetic/setup.bash --extend
source /root/catkin_ws/devel/setup.bash
cd /root/catkin_ws
```

检查左臂状态：

```bash
rostopic echo -n 1 /zj_humanoid/upperlimb/uplimb_state
```

必须确认：

```text
left_arm_is_singular: False
```

若为 `True`，不得启动遥操。

### 1.2 终端 A：启动 spacenavd

先检查容器内是否已有实例：

```bash
pgrep -a spacenavd
```

如果已有一个正常实例，不要重复启动。如果没有：

```bash
cd /
spacenavd -d -v
```

必须先 `cd /`，否则前台模式可能把 socket 建到错误位置，导致 `spacenav_node` 连接失败。

若出现 X11 授权错误，在 Orin 宿主机执行一次：

```bash
xhost +SI:localuser:root
```

然后重新启动容器内 `spacenavd`。不要使用范围过大的 `xhost +`。

### 1.3 终端 B：启动 ROS SpaceMouse 节点

```bash
rosrun spacenav_node spacenav_node
```

保持该终端运行。

### 1.4 终端 C：检查输入并准备软件停止

先确认 Joy 持续发布：

```bash
rostopic hz /spacenav/joy
rostopic echo -n 1 /spacenav/joy
```

正常要求：

- `axes` 有六项；
- 推动/倾斜/扭转时数值变化；
- 左键为 `buttons[1]=1`；
- 右键为 `buttons[0]=1`；
- 频率通常接近 50 Hz，不能持续断流。

本终端保留以下软件停止命令，必要时立即执行：

```bash
rosservice call /zj_humanoid/upperlimb/stop "{}"
```



### 1.5 首次组合控制前：明确释放左手

查看左手关节反馈：

```bash
python src/naviai_controller/scripts/test_hand.py --hand left --action status
```

如不能确认左手处于释放状态，执行：

```bash
python src/naviai_controller/scripts/test_hand.py --hand left --action release
```

确认区域安全后输入：

```text
RELEASE_LEFT
```

必须看到 `release_hand 返回: True`，并确认六维反馈随后回到零附近。

### 1.6 终端 D：最终组合控制命令

以下是当前推荐的 10°/40 mm 中低速配置，六个机械臂自由度和灵巧手均为真机执行：

```bash
python src/hilserl_wa2/interventions/spacemouse_wa2_teleop.py \
  _arm_side:=left \
  _joy_topic:=/spacenav/joy \
  _axis_map:="[0,1,2,3,4,5]" \
  _axis_sign:="[-1,-1,1,1,-1,-1]" \
  _axis_enable:="[1,1,1,1,1,1]" \
  _deadman_button:=1 \
  _hand_button:=0 \
  _watchdog:=0.25 \
  _tcp_watchdog:=0.25 \
  _state_watchdog:=0.25 \
  _publish_rate:=50.0 \
  _translation_deadzone:=0.15 \
  _rotation_deadzone:=0.18 \
  _translation_curve_mix:=0.25 \
  _rotation_curve_mix:=0.45 \
  _translation_filter_tau:=0.06 \
  _rotation_filter_tau:=0.10 \
  _translation_enter_threshold:=0.35 \
  _rotation_enter_threshold:=0.65 \
  _intent_exit_threshold:=0.20 \
  _group_switch_hysteresis:=0.15 \
  _axis_switch_hysteresis:=0.10 \
  _linear_scale:=0.010 \
  _angular_scale:=0.08 \
  _max_step_m:=0.0005 \
  _max_angular_step:=0.002 \
  _workspace_m:=0.04 \
  _rotation_limit_deg:=10.0 \
  _rotation_frame:=tool \
  _hand_cooldown:=1.0 \
  _hand_neutral_threshold:=0.08 \
  _initial_hand_state:=auto \
  _grasp_target:="[0.1,0.5,1.0,1.0,1.0,1.0]" \
  _release_target:="[0,0,0,0,0,0]" \
  _execute:=true \
  _hand_execute:=true
```

正常启动日志应包含：

```text
waiting for SpaceMouse topic: /spacenav/joy
left hand LIVE: initial_state=released ...
left arm LIVE: ... enable=[1, 1, 1, 1, 1, 1] rotation_frame=tool limit=10.0deg
idle joy_fresh=True deadman=False fault=None
```

若手部显示 `initial_state=unknown`，右键会被禁止。先退出脚本、单独释放/抓握并核对反馈，
不要直接把 `_initial_hand_state` 改成与真实状态不一致的值。

## 2. 操作方法



### 2.1 机械臂

只有按住左键 deadman 才发送 ServoL 目标：


| SpaceMouse 动作 | 机器人命令方向 | 输出轴   |
| ------------- | ------- | ----- |
| 前平移           | +X      | X     |
| 后平移           | -X      | X     |
| 左平移           | +Y      | Y     |
| 右平移           | -Y      | Y     |
| 上提            | +Z      | Z     |
| 下压            | -Z      | Z     |
| 左倾            | +Roll   | Roll  |
| 右倾            | -Roll   | Roll  |
| 前倾            | +Pitch  | Pitch |
| 后倾            | -Pitch  | Pitch |
| 顺时针扭转         | +Yaw    | Yaw   |
| 逆时针扭转         | -Yaw    | Yaw   |


松开左键后脚本执行：

```text
ServoL stop=True clear=True
```

下一次按住左键时，以最新实测 TCP 作为新原点。因此 `workspace_m` 和
`rotation_limit_deg` 是“每次按住 deadman”的累计范围，不是机器人全局工作空间。

### 2.2 灵巧手

右键在抓握和释放之间切换。正确操作顺序：

1. 松开左键；
2. 等待 `stop=True clear=True`；
3. SpaceMouse 完全回中；
4. 短按并松开右键；
5. 至少等待 `hand_cooldown`；
6. 再次短按右键执行相反动作。

预期日志：

```text
hand button accepted: grasp requested
hand command=grasp success=True dry_run=False

hand button accepted: release requested
hand command=release success=True dry_run=False
```

按住右键不会重复触发。左键未释放、手柄未回中、冷却时间未结束或手部命令仍在执行时，
右键会被拒绝。

当前 Service 通常在约 0.01～0.03 s 内返回，但手指机械运动需要更长时间；日志中的即时
`feedback` 可能仍是动作前关节值。应以随后稳定反馈和实际手指状态判断是否到位。

### 2.3 当前控制策略限制

六个自由度均已开放，但当前版本为了抑制实测串扰，每个周期只输出一个主导轴：

```text
XYZ 平移组 与 Roll/Pitch/Yaw 旋转组互斥
组内也只保留当前最强轴
```

因此目前支持“依次控制六个自由度”，还不支持真正同时斜向平移、同时多轴旋转或边平移边旋转。
这也是方向切换时可能感觉卡顿的主要原因。

## 3. 最终控制命令各部分含义



### 3.1 机器人和执行开关


| 参数                | 含义                | 允许值/建议                    |
| ----------------- | ----------------- | ------------------------- |
| `_arm_side`       | 选择机械臂和同侧手         | `left` / `right`          |
| `_execute`        | 是否真实发送机械臂 ServoL  | `false` dry-run；`true` 真机 |
| `_hand_execute`   | 是否真实调用灵巧手 Service | `false` 预览；`true` 真机      |
| `_rotation_frame` | 增量旋转所用坐标系         | `tool` 推荐；或 `base`        |
| `_publish_rate`   | 遥操循环频率            | 固定推荐 `50 Hz`（20 Hz 试验见 `docs/solution/SpaceMouse-20Hz试验说明.md`） |


`_execute` 和 `_hand_execute` 相互独立。调试按钮时可使用机械臂 dry-run：

```text
_execute:=false _hand_execute:=true
```



### 3.2 轴映射和开关


| 参数                | 当前值                 | 含义                             |
| ----------------- | ------------------- | ------------------------------ |
| `_axis_map`       | `[0,1,2,3,4,5]`     | Joy 源轴到 XYZ/Roll/Pitch/Yaw 的顺序 |
| `_axis_sign`      | `[-1,-1,1,1,-1,-1]` | 六轴方向修正                         |
| `_axis_enable`    | `[1,1,1,1,1,1]`     | 每个输出轴是否启用                      |
| `_deadman_button` | `1`                 | 左键，机械臂使能                       |
| `_hand_button`    | `0`                 | 右键，灵巧手切换                       |


常用单轴 Gate：

```text
X     [1,0,0,0,0,0]
Y     [0,1,0,0,0,0]
Z     [0,0,1,0,0,0]
Roll  [0,0,0,1,0,0]
Pitch [0,0,0,0,1,0]
Yaw   [0,0,0,0,0,1]
```

轴映射和方向已经物理标定，日常调速度时不要修改。

## 4. 速度和运动范围参数



### 4.1 参数含义与代码硬限制


| 参数                    | 单位     | 当前推荐     | 代码允许范围       | 控制特性              |
| --------------------- | ------ | -------- | ------------ | ----------------- |
| `_linear_scale`       | m/s    | `0.010`  | `[0, 0.03]`  | 满输入平移速度           |
| `_angular_scale`      | rad/s  | `0.08`   | `[0, 0.30]`  | 满输入旋转速度           |
| `_max_step_m`         | m/周期   | `0.0005` | `(0, 0.002]` | 单个循环最大平移步长        |
| `_max_angular_step`   | rad/周期 | `0.002`  | `(0, 0.01]`  | 单个循环最大旋转步长        |
| `_workspace_m`        | m      | `0.04`   | `(0, 0.10]`  | 单次 deadman 累计平移半径 |
| `_rotation_limit_deg` | °      | `10`     | `(0, 90]`    | 单次 deadman 累计旋转角  |


速度的实际上限同时受 scale 和单周期步长限制：

```text
有效线速度上限 ≈ min(linear_scale, max_step_m × publish_rate)
有效角速度上限 ≈ min(angular_scale, max_angular_step × publish_rate)
```

当前参数约为：

```text
线速度：10 mm/s
角速度：0.08 rad/s ≈ 4.58°/s
单次范围：40 mm / 10°
```



### 4.2 推荐档位


| 档位      | linear | angular   | max step | max angular step | workspace | rotation |
| ------- | ------ | --------- | -------- | ---------------- | --------- | -------- |
| 首次/靠近工件 | 0.005  | 0.04      | 0.0003   | 0.001            | 0.02      | 2°       |
| 当前推荐    | 0.010  | 0.08      | 0.0005   | 0.002            | 0.04      | 10°      |
| 较快调试    | 0.015  | 0.10～0.12 | 0.0008   | 0.003～0.004      | 0.06      | 30°      |


“较快调试”不是直接启用的默认值。必须先完成当前推荐档六个单轴、停止、奇异和线缆验收。
60°/90°必须逐级开展，不得只因为软件允许就直接使用。

只提高 `linear_scale/angular_scale` 会增加速度，但不能消除主轴切换卡顿；只提高
`workspace_m/rotation_limit_deg` 只会扩大范围，不会提高速度。

## 5. 灵敏度、曲线与丝滑度参数



### 5.1 参数速查


| 参数                             | 默认/当前  | 代码范围                    | 增大后的效果              |
| ------------------------------ | ------ | ----------------------- | ------------------- |
| `_translation_deadzone`        | 0.15   | `[0,1)`                 | 更不易漂移，但轻推更迟钝        |
| `_rotation_deadzone`           | 0.18   | `[0,1)`                 | 更不易误旋转，但倾斜起步更迟      |
| `_translation_curve_mix`       | 0.25   | `[0,1]`                 | 中小输入更弱，接近满推才明显      |
| `_rotation_curve_mix`          | 0.45   | `[0,1]`                 | 旋转中小输入更弱、更非线性       |
| `_translation_filter_tau`      | 0.06 s | `>=0`                   | 更平滑但延迟更大            |
| `_rotation_filter_tau`         | 0.10 s | `>=0`                   | 更平滑但延迟更大            |
| `_translation_enter_threshold` | 0.35   | `(exit,1]`              | 更难进入平移意图            |
| `_rotation_enter_threshold`    | 0.65   | `(exit,1]`              | 更难进入旋转意图            |
| `_intent_exit_threshold`       | 0.20   | `(0,translation_enter)` | 增大后更早退出当前意图         |
| `_group_switch_hysteresis`     | 0.15   | `>=0`                   | 平移/旋转组更稳定，但切换更慢     |
| `_axis_switch_hysteresis`      | 0.10   | `>=0`                   | 主轴更稳定，但 X/Y/Z 等切换更慢 |


这里的“代码范围”只是参数校验范围，不代表整个范围都适合真机。

### 5.2 当前卡顿的判断

今日终端日志中持续为：

```text
joy_fresh=True
fault=None
ServoL stop=True clear=True
```

没有 watchdog 或 ROS 断流证据。当前卡顿主要来自：

- 单一主轴选择；
- 主轴切换时滤波状态清零；
- 平移/旋转进入阈值；
- 轴/组切换迟滞；
- 较强的旋转非线性曲线。

提高速度只会让选中的轴更快，不能从根本上解决上述不连续。

### 5.3 可进行的手感实验参数

若当前推荐档所有安全 Gate 已通过，可先在 `_execute:=false` 或 2°/20 mm 低范围下试验：

```bash
_translation_deadzone:=0.12 \
_rotation_deadzone:=0.15 \
_translation_curve_mix:=0.15 \
_rotation_curve_mix:=0.25 \
_translation_filter_tau:=0.05 \
_rotation_filter_tau:=0.08 \
_translation_enter_threshold:=0.30 \
_rotation_enter_threshold:=0.60 \
_intent_exit_threshold:=0.18 \
_group_switch_hysteresis:=0.12 \
_axis_switch_hysteresis:=0.06
```

预期：更容易起步、轴切换更快、中小幅操作更线性。风险：串扰和错误换轴概率增加。每次只改变
一组参数并记录手感，不要同时提高速度、扩大角度和降低阈值。

这仍不能实现真正的多轴丝滑控制；根本优化需要修改 `spacemouse_input.py`，允许组内多轴连续输出
并对主轴交接做淡入淡出或 6×6 解耦。

## 6. watchdog 和安全参数


| 参数                | 当前值    | 代码要求     | 说明                  |
| ----------------- | ------ | -------- | ------------------- |
| `_watchdog`       | 0.25 s | `>0`     | Joy 超时后禁止控制         |
| `_tcp_watchdog`   | 0.25 s | `>0`     | TCP 反馈过期后禁止控制       |
| `_state_watchdog` | 0.25 s | `>0`     | UplimbState 过期后禁止控制 |
| `_max_dt`         | 0.05 s | `>0`     | 单次积分使用的最大时间间隔       |
| `_servo_time`     | 0.02 s | 只能为 0.02 | 已验证 ServoL 参数；20 Hz 须改为 0.05 s，见 `docs/solution/SpaceMouse-20Hz试验说明.md` |
| `_servo_gain`     | 800    | 只能为 800  | 已验证 ServoL 参数       |


不要通过放宽 watchdog 掩盖输入或状态频率问题。若出现 stale/fault，应先定位发布端。

## 7. 灵巧手参数


| 参数                         | 当前推荐                | 允许值/范围                | 说明           |
| -------------------------- | ------------------- | --------------------- | ------------ |
| `_hand_button`             | 0                   | 非负且不能等于 deadman       | 右键下标         |
| `_hand_cooldown`           | 1.0 s               | `>=0.3`               | 两次命令最小间隔     |
| `_hand_neutral_threshold`  | 0.08                | `[0,0.20]`            | 允许按右键时的最大轴偏移 |
| `_initial_hand_state`      | `auto`              | auto/released/grasped | 启动状态判断       |
| `_hand_feedback_timeout`   | 2.0 s               | `>0`                  | 等待初始手部反馈     |
| `_hand_feedback_tolerance` | 0.35                | `>0`                  | 自动状态判断距离阈值   |
| `_grasp_target`            | `[0.1,0.5,1,1,1,1]` | 六个有限值                 | 今日最新实机抓握目标   |
| `_release_target`          | 六个零                 | 必须六个零                 | 释放目标         |


抓握角度必须根据物体和灵巧手安全范围单独验证。参数合法不代表机械上安全。

如果静止时仍出现 `SpaceMouse must be centered`，记录连续静止 `axes`，再考虑把
`hand_neutral_threshold` 从 0.08 调到 0.10/0.12；不得直接拉到 0.20。

## 8. Dry-run 和分层验收



### 8.1 全部预览，不动作

在最终命令中使用：

```text
_execute:=false _hand_execute:=false _initial_hand_state:=released
```

机械臂和灵巧手都不动作，只检查目标、方向、按钮状态机和拒绝条件。

### 8.2 只测试灵巧手

```text
_execute:=false _hand_execute:=true
```

机械臂不发送 ServoL；仍需释放 deadman、手柄回中后短按右键。

### 8.3 只测试机械臂

```text
_execute:=true _hand_execute:=false
```

右键只做预览，不移动手指。

### 8.4 组合验收序列

```text
左键 + X 小范围平移
→ 松开，确认 stop=True clear=True
→ 左键 + Roll 小角度旋转
→ 松开并回中
→ 右键抓握，等待实际手指到位
→ 左键低速移动
→ 松开并回中
→ 右键释放
```

通过标准：

- 六个方向与标定一致；
- 松开 deadman 后立即停止；
- 无 watchdog、fault、奇异、跳变和异响；
- 右键一次只触发一个命令；
- 手部 busy 时机械臂不能启动；
- 抓握/释放实际到位，不只看 Service 返回；
- `Ctrl+C` 后完成 stop/clear。



## 9. 常见问题



### 9.1 能动但卡顿、不丝滑

先检查：

```bash
rostopic hz /spacenav/joy
rostopic echo /spacenav/joy
```

若频率稳定且日志始终 `joy_fresh=True fault=None`，优先按第 5 节判断输入策略，不要先放宽
watchdog。先降低 curve/threshold/hysteresis 做小范围试验；如果主轴交接仍明显卡顿，需要代码级
多轴输出优化。

### 9.2 有输入但动作很小

- `linear_scale/angular_scale` 太小；
- 推入力度落在 deadzone/enter threshold 以下；
- 已到 `workspace_m/rotation_limit_deg`；
- 松开并重新按 deadman 会以当前 TCP 开始新一段，但必须注意累计物理位移。



### 9.3 方向切换时停一下

这是当前 dominant-axis lock 的已知表现。`axis_switch_hysteresis` 越大，主轴越稳定但切换越慢；
适度降低到 0.06～0.08 可测试，但可能增加串扰。

### 9.4 右键没有动作

查看拒绝日志：

- `deadman must be released`：松开左键；
- `ServoL session must be stopped`：等 stop/clear 后重新短按；
- `SpaceMouse must be centered`：手柄回中；
- `hand button cooldown active`：等待至少一秒；
- `hand state is unknown`：单独验证并初始化手部状态；
- `success=False`：检查 `/zj_humanoid/hand/joint_switch/{side}` Service。



### 9.5 Service 成功但反馈还是旧值

Service 返回早于手指到位，当前回调立即读取会得到旧反馈。等待约一段机械动作时间后重新执行：

```bash
python src/naviai_controller/scripts/test_hand.py --hand left --action status
```



### 9.6 松手不停或出现异常

立即执行物理急停或：

```bash
rosservice call /zj_humanoid/upperlimb/stop "{}"
```

然后退出遥操，检查是否有第二个控制节点、Joy 按钮未回零或状态反馈异常。

### 9.7 `spacenav_node` 连接失败

```bash
pgrep -a spacenavd
ls -l /var/run/spnav.sock /run/spnav.sock
```

确保只有一个 daemon，并且从 `/` 启动 `spacenavd -d -v`。

## 10. 退出顺序

1. 松开两个 SpaceMouse 按钮；
2. 等待 `ServoL stop=True clear=True`；
3. `Ctrl+C` 退出遥操脚本；
4. 必要时执行一次 upperlimb stop；
5. `Ctrl+C` 退出 `spacenav_node`；
6. 最后退出前台 `spacenavd`。

