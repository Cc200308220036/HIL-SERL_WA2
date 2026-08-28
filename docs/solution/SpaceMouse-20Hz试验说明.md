# 独立 SpaceMouse 遥操：20 Hz Servo 试验说明

本文只覆盖 **独立遥操**（`spacemouse_wa2_teleop.py`）。  
**不要**改 Actor / Learner / `WA2ServoSession` / `wa2_env_contract.yaml`。测完改回 50 Hz。

日常 50 Hz 启动仍看《SpaceMouse使用指南.md》。

---

## 1. 先分清两件事

| 名称 | 当前值 | 20 Hz 应对齐成 | 作用 |
|---|---|---|---|
| `_publish_rate` | 50 Hz | **20 Hz** | Python 循环多久调用一次 `servol()` |
| `_servo_time` | **0.02 s** | **0.05 s** | 固件 `set_servo_params` 的指令周期 |

对应关系：

```text
频率 = 1 / 周期
50 Hz  ↔  0.02 s
20 Hz  ↔  0.05 s     ← 要测 20 Hz 必须用这个
```

**不要**把 20 Hz 写成 `_servo_time:=0.02`。0.02 s 仍是 50 Hz。  
**不要**只把 `_publish_rate` 改成 20、固件仍保持 0.02：控制器按每 20 ms 等指令，发慢了会顿挫，严重时进 `PROTECTED`（`cmd_num=15`）。

当前代码把固件周期锁死在 0.02，只改 ROS 参数会启动失败。必须先放开下面两处硬限制。

---

## 2. 最小改动（只动独立遥操链路）

只改这三处。Actor 继续传 `set_servo_params(0.02, 800)`，行为不变。

### 2.1 放开固件周期白名单

文件：`catkin_ws/src/naviai_controller/src/naviai_controller/core/arm.py`  
函数：`ArmController.set_servo_params`

把「只允许 0.02」改成允许 `0.02` 或 `0.05`：

```python
allowed_times = (0.02, 0.05)
if not np.isfinite(time_sec) or not any(
    np.isclose(time_sec, allowed) for allowed in allowed_times
):
    raise ValueError(
        "only time_sec=0.02 (50 Hz) or 0.05 (20 Hz trial) is currently allowed"
    )
```

不要把白名单改成任意值。`gain` 仍只允许 `800`。

### 2.2 放开遥操校验，并强制频率与周期对齐

文件：`catkin_ws/src/hilserl_wa2/interventions/spacemouse_wa2_teleop.py`

**默认值**（`__init__`，约第 46–48 行）改成：

```python
self.publish_rate = float(rospy.get_param("~publish_rate", 20.0))
self.max_dt = float(rospy.get_param("~max_dt", 0.08))
self.servo_time = float(rospy.get_param("~servo_time", 0.05))
```

`max_dt` 从 0.05 提到 0.08：20 Hz 一拍就是 50 ms，默认 `max_dt=0.05` 刚好顶死，循环稍慢积分会被截断。

**校验**（`_validate_config`，约第 248–249 行）把原来的

```python
if not np.isclose(self.servo_time, 0.02) or self.servo_gain != 800:
    raise ValueError("only verified ServoL parameters time=0.02 gain=800 are allowed")
```

换成：

```python
if self.servo_gain != 800 or not any(
    np.isclose(self.servo_time, allowed) for allowed in (0.02, 0.05)
):
    raise ValueError(
        "only ServoL time=0.02|0.05 and gain=800 are allowed"
    )
expected_hz = 1.0 / self.servo_time
if not np.isclose(self.publish_rate, expected_hz, atol=0.51):
    raise ValueError(
        "publish_rate must match servo_time "
        "({:.0f} Hz for {:.3f} s), got {:.1f}".format(
            expected_hz, self.servo_time, self.publish_rate
        )
    )
```

后一段是为了拦住「循环 50 Hz + 固件 0.05 s」或反过来的错配。

启动后日志里应能看到 `servo_time=0.050s publish_rate=20.0Hz`（可在 `run()` 现有 `logwarn` 里把这两项打出来）。

### 2.3 若走 YAML 入口，同步 teleop 段

仅当使用：

```bash
python src/hilserl_wa2/scripts/run_spacemouse_teleop_from_yaml.py --config default
```

才需要改 `catkin_ws/src/hilserl_wa2/configs/spacemouse/default.yaml` 的 **`teleop:`** 段：

```yaml
  publish_rate: 20.0
  servo_time: 0.05
```

**禁止**改同一文件里的 `intervention.control_dt: 0.02`，那是 Actor 用的。

`collect.yaml` 若仍是 `publish_rate: 50.0` 且没有 `servo_time`，走 `--config collect` 会和 2.2 的对齐校验冲突。这次试验不要用 collect；或给 collect 也写上配对的 20 / 0.05。

---

## 3. 启动参数（指南命令上只改这几项）

其余轴映射、死区、速度档与《SpaceMouse使用指南.md》§1.6 相同。在那条命令里替换/增加：

```bash
  _publish_rate:=20.0 \
  _servo_time:=0.05 \
  _max_dt:=0.08 \
```

不要再写 `_publish_rate:=50.0`。  
不要写 `_servo_time:=0.02`。

先 `_execute:=false` 看能否起来；真机再用低速档（例如 `_linear_scale:=0.005`、`_workspace_m:=0.02`），急停在手边。

当前推荐档满杆线速度名义上仍约 10 mm/s：

```text
min(linear_scale, max_step_m × publish_rate)
= min(10 mm/s, 0.5 mm × 20 Hz)
= 10 mm/s
```

手感会更「一格一格」，不是更丝滑。这是降频试验，不是手感优化。

---

## 4. 明确不要动的文件

| 文件 | 原因 |
|---|---|
| `wa2_env_contract.yaml` 的 `control_hz` / `control_dt` / `servo_time` | Actor 50 Hz × 1 mm 契约 |
| `ros_adapters/servo_session.py` | HIL / 采集唯一 ServoL 发布线程 |
| `interventions/wa2_spacemouse_intervention.py` | 训练期干预，不是独立遥操 |
| `scripts/r13_actor_train.py`、`record_r13_demos.py` | Actor / 采集主循环 |
| `HILSERL_Learner/` 下任何副本 | Learner 端 |

---

## 5. 现场看什么

1. 启动日志：`servo_time=0.050s publish_rate=20.0Hz`。
2. `/zj_humanoid/upperlimb/uplimb_state` 不要变成 `PROTECTED` / `cmd_num=15`。
3. 一旦保护：松开 deadman，确认日志有 `ServoL stop=... clear=...`；必要时手动：

```bash
rosservice call /zj_humanoid/upperlimb/stop "{}"
```

4. 测完立刻改回：`publish_rate=50`、`servo_time=0.02`、`max_dt=0.05`，两处白名单若不想留 0.05 就一并还原。

---

## 6. 改回 50 Hz 检查清单

- [ ] `spacemouse_wa2_teleop.py` 默认 `50.0` / `0.02` / `max_dt=0.05`，校验恢复「只允许 0.02」
- [ ] `arm.py` 恢复「只允许 `time_sec=0.02`」（或白名单仍含 0.05 但默认不再走 20 Hz）
- [ ] `default.yaml` 的 `teleop.publish_rate` 回到 `50.0`，去掉试验用的 `servo_time: 0.05`
- [ ] 使用指南命令仍是 `_publish_rate:=50.0`，不要留 20
- [ ] 没有 Actor / 采集进程在控同一条臂时再开独立遥操
