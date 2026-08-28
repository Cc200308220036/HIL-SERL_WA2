# R10 Learner–Actor 正式联调手册

> 更新：2026-08-14，按现场已通过的 GPU、网络、manifest 与防火墙证据固化。  
> Learner：笔记本 `/home/cyw/orin_hilserl/HILSERL_Learner`，Conda `hil-learner`。  
> Actor：Orin `hilserl` 容器 `/root/catkin_ws`，Conda `hil-actor`。  
> 权威方案：分析根 `docs/solution/R10_方案.md` §2.4、§5.9～§5.17、§6。  
> 本手册是执行入口，不改变权威方案的 PASS 口径。

## 1. 当前前置状态与冻结值

以下前置已验收通过，可以开始正式跨机联调：

```text
Learner 依赖 / 新路径                    PASS
RTX 4060 / JAX CudaDevice                PASS
SAC params / JIT action                  PASS
Replay Buffer 4 insert calls             PASS
local.yaml 三端一致                      PASS
双向局域网                               PASS
Actor/Learner manifest                    PASS
Learner UFW 最小权限                      PASS
R10 GATE                                 NOT RUN
```

冻结值：

```text
task                         bottle_pick
exp                          wa2_bottle_pick
Learner IP                   172.16.9.36
Actor/Orin IP                172.16.8.58
request / broadcast          TCP 5588 / 5589
local.yaml SHA-256           88455aa14a54c857cbbb756e61036f569822e561fab67e2af72e368e5f08edf0
network_config_hash          5a53982b1d8342def2af5632156cc99c372358f6f36de920860436d59c6d7c08
config_bundle_hash           bff87be7f04e6752ea2c19cec058f5687e51bc9faba15d21aab0535cc2080923
space_hash                   b7a0b860d94b648a56cc453d772c478886dc4d0acecb89d6e8eb527b6831b367
params_tree_signature        08f79859e3ba2f5da5a9b0cb16e63bd10aa355878e3a0f724465b8850aae6920
Agentlace wheel SHA-256      1a800cc341f03eb6844273571ba26a265920fa1b5a698acc3d954438cbb72d32
source_tree_sha256           b3a7537e0b8fb4d6b5cbbc36754478a7462de2b3a9637d098de3f65e657359cb
protocol / transition        wa2-r10-v1 / r9-v1
```

UFW 保持现有最小权限规则：仅允许 `172.16.8.58` 访问笔记本 `wlo1` 的 TCP 5588/5589。不要改成全端口互信。

## 2. 终端划分、强制顺序与安全边界

| 标识 | 位置 | 用途 |
|---|---|---|
| L | Learner 笔记本 | 当前阶段 Learner Server 前台进程；用 `Ctrl+C` 做规定故障注入 |
| L2 | Learner 笔记本第二终端 | 端口、状态 JSON 与资源监控；不启动第二个 Server |
| O | Orin 宿主机 `naviai@172.16.8.58` | ping/nc、Docker 与宿主机归档 |
| A | Orin `hilserl` 容器 | 唯一 Actor Gate 进程 |
| J | Orin `hilserl` 容器第二终端 | readonly/live-zero 前启动并检查 SpaceMouse Joy；不启动 teleop |

强制执行顺序：

```text
F0 fake 1020/20
  → F1 fake 断线 fail-closed
  → F2 新 Server + 人工新 Actor session 恢复 200/10
  → H0 readonly 真观测无运动
  → H1 live-zero 断线 stop+clear
  → 双端归档与最终验收
```

任何阶段失败立即停止，不越级。R10 禁止：

- 未训练 SAC 真机 action；
- 正式 RLPD 梯度更新；
- 断线自动重连/自动续跑；
- 同时启动 teleop 与 Actor；
- 用 R9 localhost 脚本冒充 R10；
- 在 Orin 启 Learner Server，或在笔记本启动 Actor。

## 3. 每个新终端的环境初始化

### 3.1 Learner 终端 L / L2

```bash
export R10_LEARNER_ROOT=/home/cyw/orin_hilserl/HILSERL_Learner
export R10_LEARNER_IP=172.16.9.36
export R10_ACTOR_IP=172.16.8.58

source "$R10_LEARNER_ROOT/scripts/activate_hil_learner.sh"
unset JAX_PLATFORMS

test "$CONDA_DEFAULT_ENV" = hil-learner
test "$HILSERL_LEARNER_ROOT" = "$R10_LEARNER_ROOT"
```

### 3.2 Orin 宿主机终端 O

```bash
export R10_LEARNER_IP=172.16.9.36
export R10_ACTOR_IP=172.16.8.58

test "$(hostname -I | tr ' ' '\n' | grep -Fx 172.16.8.58)" = 172.16.8.58
docker ps --filter name=hilserl --format '{{.Names}} {{.Status}}'
```

### 3.3 Orin 容器终端 A / J

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate hil-actor
source /ros_noetic/catkin_ws/devel/setup.bash
source /opt/ros/noetic/setup.bash --extend
source /root/catkin_ws/devel/setup.bash

export R10_ACTOR_ROOT=/root/catkin_ws
export R10_LEARNER_IP=172.16.9.36
export R10_ACTOR_IP=172.16.8.58
export PYTHONPATH=/root/catkin_ws/src:/root/catkin_ws/src/hil-serl-main/examples:/root/catkin_ws/src/hil-serl-main/serl_launcher:${PYTHONPATH:-}
export XLA_FLAGS=--xla_gpu_autotune_level=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.1

cd /root/catkin_ws
test "$CONDA_DEFAULT_ENV" = hil-actor
```

## 4. 建立本次统一运行 ID

只在 Learner L2 生成一次：

```bash
export R10_RUN_ID="$(date +%Y%m%d_%H%M%S)_r10"
echo "R10_RUN_ID=$R10_RUN_ID"
```

把打印值原样复制到 L、O、A、J。例如：

```bash
export R10_RUN_ID=20260814_150000_r10
```

然后分别建立目录。

Learner L / L2：

```bash
export R10_RUN="$R10_LEARNER_ROOT/runs/wa2_bottle_pick/$R10_RUN_ID"
mkdir -p "$R10_RUN"
echo "$R10_RUN"
```

Orin 容器 A / J：

```bash
export R10_ACTOR_RUN="/root/catkin_ws/runs/wa2_bottle_pick/$R10_RUN_ID"
mkdir -p "$R10_ACTOR_RUN"
echo "$R10_ACTOR_RUN"
```

Orin 宿主机 O：

```bash
export R10_ORIN_RUN="/home/naviai/hilserl_orin/catkin_ws/runs/wa2_bottle_pick/$R10_RUN_ID"
mkdir -p "$R10_ORIN_RUN"
echo "$R10_ORIN_RUN"
```

各终端打印的最后一级目录必须是同一个 `R10_RUN_ID`。

## 5. F0：fake 1020/20 主通信 Gate（无机器人）

### 5.1 L：先启动 Learner Server

```bash
python "$R10_LEARNER_ROOT/src/hilserl_wa2/scripts/r10_learner_server.py" \
  --task bottle_pick \
  --network-config "$R10_LEARNER_ROOT/src/hilserl_wa2/configs/network/local.yaml" \
  --manifest "$R10_LEARNER_ROOT/runs/r10_preflight/learner_manifest.json" \
  --expect-env 1020 \
  --expect-intvn 20 \
  --capacity 5000 \
  --republish-s 1.0 \
  --output "$R10_RUN/learner_summary.json" \
  --status-file "$R10_RUN/learner_status.json" \
  2>&1 | tee "$R10_RUN/learner.log"
```

首次 SAC 初始化约需 3 分钟。不要中断，必须先看到：

```text
JAX_DEVICES=[CudaDevice(...)]
PARAMS_TREE_SIGNATURE=08f79859e3ba2f5da5a9b0cb16e63bd10aa355878e3a0f724465b8850aae6920
PARAMS_TREE_READY
R10_LEARNER_READY request=5588 broadcast=5589
SERVER_INSTANCE_ID=<非空>
```

将 `SERVER_INSTANCE_ID` 记入验收记录。此时 L 保持运行。

### 5.2 L2：验证本机监听和 UFW

```bash
ss -ltnp | grep -E ':(5588|5589)\b' \
  | tee "$R10_RUN/port_listen.txt"

sudo ufw status numbered \
  | tee "$R10_RUN/firewall.txt"
```

必须同时监听 5588/5589，且 UFW 仍只允许 `172.16.8.58`。

### 5.3 O：从 Orin 宿主机探测 Learner

```bash
ping -c 10 "$R10_LEARNER_IP" \
  | tee "$R10_ORIN_RUN/ping.txt"

{
  nc -vz -w 3 "$R10_LEARNER_IP" 5588
  nc -vz -w 3 "$R10_LEARNER_IP" 5589
} 2>&1 | tee "$R10_ORIN_RUN/port_probe.txt"
```

两个端口都必须 `succeeded`。任一失败，不启动 Actor。

### 5.4 A：后启动 fake Actor

```bash
python /root/catkin_ws/src/hilserl_wa2/scripts/verify_r10_actor_remote.py \
  --task bottle_pick \
  --network-config /root/catkin_ws/src/hilserl_wa2/configs/network/local.yaml \
  --manifest /root/catkin_ws/runs/wa2_bottle_pick/r10_preflight/actor_manifest.json \
  --server-ip "$R10_LEARNER_IP" \
  --mode fake \
  --policy zero \
  --steps 1020 \
  --synthetic-intervention-start 1000 \
  --synthetic-intervention-steps 20 \
  --require-intervention-segments 1 \
  --require-network-update \
  --max-seconds 300 \
  --output "$R10_ACTOR_RUN/actor_main.json" \
  --dump "$R10_ACTOR_RUN/actor_main.pkl" \
  2>&1 | tee "$R10_ACTOR_RUN/actor_main.log"
```

Actor 必须出现：

```text
R10_HANDSHAKE: PASS
LOCAL_ENV_COUNT=1020
LOCAL_INTVN_COUNT=20
NON_INTERVENTION_STEPS=1000
INTERVENTION_SEGMENTS=1
SERVER_ENV_COUNT=1020
SERVER_INTVN_COUNT=20
LAST_UPDATE_ID_MATCH=PASS
ORDERED_DIGEST_MATCH=PASS
NETWORK_UPDATE_COUNT>=1
R10_ACTOR_FAKE: PASS
```

Learner L 必须出现：

```text
R10_HANDSHAKE: PASS
EXPECT_MET env=1020 intvn=20
R10_AGENTLACE_REMOTE: PASS
```

### 5.5 L2：核对 Learner status

```bash
python - "$R10_RUN/learner_status.json" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
s = json.loads(p.read_text())
assert s["handshake_accepted"] is True, s
assert s["schema_ok"] is True, s
assert s["actor_env_count"] == 1020, s
assert s["actor_env_intvn_count"] == 20, s
ids = s["last_update_id"]
assert "actor_env" in ids and "actor_env_intvn" in ids, ids
assert s["ordered_digest"], s
assert s["ordered_intvn_digest"], s
print(f'SERVER_INSTANCE_ID={s["server_instance_id"]}')
print(f'LAST_UPDATE_ID={ids}')
print('R10_LEARNER_MAIN_STATUS: PASS')
PY
```

### 5.6 A：核对性能 summary

```bash
python - "$R10_ACTOR_RUN/actor_main.json" <<'PY'
import json, sys
from pathlib import Path
s = json.loads(Path(sys.argv[1]).read_text())
assert s["local_env_count"] == 1020, s
assert s["local_intvn_count"] == 20, s
assert s["non_intervention_steps"] == 1000, s
assert s["intervention_segments"] == 1, s
assert s["server_env_count"] == 1020, s
assert s["server_intvn_count"] == 20, s
assert s["last_update_id_match"] is True, s
assert s["ordered_digest_match"] is True, s
assert s["network_update_count"] >= 1, s
assert s["upload_rtt_ms"]["p95"] < 3000.0, s
assert s["network_age_s"] is not None and s["network_age_s"] < 5.0, s
assert not s["fault_reason"], s
print(f'HANDSHAKE_RTT_MS={s["handshake_rtt_ms"]}')
print(f'UPLOAD_RTT_MS={s["upload_rtt_ms"]}')
print(f'STATUS_RTT_MS={s["status_rtt_ms"]}')
print(f'TRANSITIONS_PER_SECOND={s["transitions_per_second"]}')
print(f'EFFECTIVE_MIB_PER_SECOND={s["effective_mib_per_second"]}')
print('R10_PERFORMANCE: PASS')
PY
```

F0 全部 PASS 后，在 L 对 Server 按 `Ctrl+C`。必须看到 `R10_LEARNER_STOPPED`。L2 确认端口释放：

```bash
ss -ltnp | grep -E ':(5588|5589)\b' || echo 'R10_MAIN_SERVER_PORTS_RELEASED: PASS'
```

## 6. F1：主动停止 Learner，验证 Actor fail-closed（无机器人）

### 6.1 L：启动新的故障 Server 进程

```bash
python "$R10_LEARNER_ROOT/src/hilserl_wa2/scripts/r10_learner_server.py" \
  --task bottle_pick \
  --network-config "$R10_LEARNER_ROOT/src/hilserl_wa2/configs/network/local.yaml" \
  --manifest "$R10_LEARNER_ROOT/runs/r10_preflight/learner_manifest.json" \
  --capacity 5000 \
  --republish-s 1.0 \
  --output "$R10_RUN/disconnect_server.json" \
  --status-file "$R10_RUN/disconnect_status.json" \
  2>&1 | tee "$R10_RUN/disconnect_server.log"
```

等待新的 `R10_LEARNER_READY` 和 `SERVER_INSTANCE_ID`。

### 6.2 O：重新检查两个端口

```bash
nc -vz -w 3 "$R10_LEARNER_IP" 5588
nc -vz -w 3 "$R10_LEARNER_IP" 5589
```

### 6.3 A：启动长运行 fake Actor

```bash
python /root/catkin_ws/src/hilserl_wa2/scripts/verify_r10_actor_remote.py \
  --task bottle_pick \
  --network-config /root/catkin_ws/src/hilserl_wa2/configs/network/local.yaml \
  --manifest /root/catkin_ws/runs/wa2_bottle_pick/r10_preflight/actor_manifest.json \
  --server-ip "$R10_LEARNER_IP" \
  --mode fake \
  --policy zero \
  --steps 100000 \
  --control-hz 20 \
  --require-network-update \
  --expect-fault network_loss \
  --output "$R10_ACTOR_RUN/actor_disconnect.json" \
  --dump-on-fault "$R10_ACTOR_RUN/fault_dump" \
  2>&1 | tee "$R10_ACTOR_RUN/actor_disconnect.log"
```

### 6.4 L2：等至少 30 条后停止 Server

监控：

```bash
watch -n 1 "python -c 'import json; print(json.load(open(\"$R10_RUN/disconnect_status.json\"))[\"actor_env_count\"])'"
```

计数达到 30 后退出 `watch`，在 L 对 Python Server 按 `Ctrl+C`。不要关闭 Docker、ROS master、Wi-Fi或整台笔记本。

Actor 必须自动退出并出现：

```text
FAULT_REASON=network_loss
FAULT_DETAIL=server_disconnect 或 network_stale
NO_FURTHER_STEPS=true
ENV_CLOSED=true
CLIENT_STOPPED=true
FAULT_DUMP_WRITTEN=true
AUTO_RECONNECT=false
R10_SERVER_DISCONNECT: PASS
```

确认 dump：

```bash
test -f "$R10_ACTOR_RUN/fault_dump/fault_dump.pkl"
test -f "$R10_ACTOR_RUN/fault_dump/fault_meta.json"
echo 'R10_FAULT_DUMP: PASS'
```

## 7. F2：Learner 重启与人工新 session 恢复 200/10（无机器人）

### 7.1 前置确认

旧 Actor 必须已经退出：

```bash
pgrep -af verify_r10_actor_remote.py || echo 'OLD_ACTOR_EXITED: PASS'
```

L2 确认旧 Server 端口已释放：

```bash
ss -ltnp | grep -E ':(5588|5589)\b' || echo 'OLD_SERVER_EXITED: PASS'
```

### 7.2 L：启动新的恢复 Server

```bash
python "$R10_LEARNER_ROOT/src/hilserl_wa2/scripts/r10_learner_server.py" \
  --task bottle_pick \
  --network-config "$R10_LEARNER_ROOT/src/hilserl_wa2/configs/network/local.yaml" \
  --manifest "$R10_LEARNER_ROOT/runs/r10_preflight/learner_manifest.json" \
  --expect-env 200 \
  --expect-intvn 10 \
  --capacity 5000 \
  --republish-s 1.0 \
  --output "$R10_RUN/recovery_server.json" \
  --status-file "$R10_RUN/recovery_status.json" \
  2>&1 | tee "$R10_RUN/recovery_server.log"
```

记录新的 `SERVER_INSTANCE_ID`，必须不同于 F1。

### 7.3 O：重新检查端口后，A 人工启动新 Actor

```bash
nc -vz -w 3 "$R10_LEARNER_IP" 5588
nc -vz -w 3 "$R10_LEARNER_IP" 5589
```

A：

```bash
python /root/catkin_ws/src/hilserl_wa2/scripts/verify_r10_actor_remote.py \
  --task bottle_pick \
  --network-config /root/catkin_ws/src/hilserl_wa2/configs/network/local.yaml \
  --manifest /root/catkin_ws/runs/wa2_bottle_pick/r10_preflight/actor_manifest.json \
  --server-ip "$R10_LEARNER_IP" \
  --mode fake \
  --policy zero \
  --steps 200 \
  --synthetic-intervention-start 50 \
  --synthetic-intervention-steps 10 \
  --require-intervention-segments 1 \
  --require-network-update \
  --output "$R10_ACTOR_RUN/actor_recovery.json" \
  2>&1 | tee "$R10_ACTOR_RUN/actor_recovery.log"
```

必须满足：

```text
R10_HANDSHAKE: PASS
LOCAL_ENV_COUNT=200
LOCAL_INTVN_COUNT=10
SERVER_ENV_COUNT=200
SERVER_INTVN_COUNT=10
LAST_UPDATE_ID_MATCH=PASS
ORDERED_DIGEST_MATCH=PASS
NETWORK_UPDATE_COUNT>=1
R10_ACTOR_FAKE: PASS
```

L2 对账 Server instance 与空 store 恢复：

```bash
python - "$R10_RUN/disconnect_status.json" "$R10_RUN/recovery_status.json" <<'PY'
import json, sys
from pathlib import Path
d_srv, r_srv = [json.loads(Path(p).read_text()) for p in sys.argv[1:]]
assert d_srv["server_instance_id"] != r_srv["server_instance_id"]
assert r_srv["actor_env_count"] == 200
assert r_srv["actor_env_intvn_count"] == 10
assert r_srv["schema_ok"] is True
print('NEW_SERVER_INSTANCE_ID: PASS')
print('NO_OLD_STORE_REPLAY: PASS')
print('R10_RECOVERY_SERVER: PASS')
PY
```

A 对账 Actor session（两个 JSON 都在容器运行目录）：

```bash
python - "$R10_ACTOR_RUN/actor_disconnect.json" "$R10_ACTOR_RUN/actor_recovery.json" <<'PY'
import json, sys
from pathlib import Path
d_act, r_act = [json.loads(Path(p).read_text()) for p in sys.argv[1:]]
assert d_act["session_id"] != r_act["session_id"]
assert r_act["local_env_count"] == 200
assert r_act["local_intvn_count"] == 10
assert r_act["server_env_count"] == 200
assert r_act["server_intvn_count"] == 10
assert r_act["last_update_id_match"] is True
assert r_act["ordered_digest_match"] is True
print('NEW_ACTOR_SESSION_ID: PASS')
print('R10_RECOVERY_ACTOR: PASS')
PY
```

完成后在 L 按 `Ctrl+C` 停恢复 Server。

## 8. H0：readonly 跨机真观测 Gate（无运动）

只有 R6/R7 设备 Gate 仍有效、工作区安全时才执行。

### 8.1 J：启动并检查 SpaceMouse Joy，但禁止 teleop

```bash
if pgrep -af 'spacemouse_wa2_teleop|run_spacemouse_teleop_from_yaml'; then
  echo 'R10_READONLY: FAIL — teleop publisher exists'
  false
else
  echo 'ACTION_PUBLISHER_EXCLUSIVE: PASS'
fi

bash /root/catkin_ws/src/hilserl_wa2/scripts/start_spacemouse_joy.sh
```

必须输出 `SPACEMOUSE_JOY: OK`。Joy 节点不是 Action 发布者；不要按 deadman。

### 8.2 L：启动新的 readonly Server

```bash
python "$R10_LEARNER_ROOT/src/hilserl_wa2/scripts/r10_learner_server.py" \
  --task bottle_pick \
  --network-config "$R10_LEARNER_ROOT/src/hilserl_wa2/configs/network/local.yaml" \
  --manifest "$R10_LEARNER_ROOT/runs/r10_preflight/learner_manifest.json" \
  --capacity 5000 \
  --republish-s 1.0 \
  --output "$R10_RUN/readonly_server.json" \
  --status-file "$R10_RUN/readonly_status.json" \
  2>&1 | tee "$R10_RUN/readonly_server.log"
```

等待 READY，O 重新 `nc` 两端口，然后 A：

```bash
python /root/catkin_ws/src/hilserl_wa2/scripts/verify_r10_actor_remote.py \
  --task bottle_pick \
  --network-config /root/catkin_ws/src/hilserl_wa2/configs/network/local.yaml \
  --manifest /root/catkin_ws/runs/wa2_bottle_pick/r10_preflight/actor_manifest.json \
  --server-ip "$R10_LEARNER_IP" \
  --mode readonly \
  --policy zero \
  --min-seconds 20 \
  --require-network-update \
  --output "$R10_ACTOR_RUN/actor_readonly.json" \
  2>&1 | tee "$R10_ACTOR_RUN/actor_readonly.log"
```

必须出现：

```text
ROBOT_MOTION=false
REAL_IMAGES=true
STATE_FINITE=true
action_ignored_for_motion=true
ENV_CLOSED=true
CLIENT_STOPPED=true
R10_ACTOR_READONLY: PASS
```

若机器人发生任何运动，立即急停并判 R10 FAIL。完成后在 L 停 Server。

## 9. H1：live-zero 跨机断线 stop+clear Gate

这是 R10 唯一真机控制链 Gate。必须：现场授权、急停可达、工作区清空、Joy 正常、teleop 关闭、全程不按 SpaceMouse deadman。

### 9.1 J：安全门禁

```bash
if pgrep -af 'spacemouse_wa2_teleop|run_spacemouse_teleop_from_yaml'; then
  echo 'R10_LIVE: FAIL — teleop publisher exists'
  false
else
  echo 'ACTION_PUBLISHER_EXCLUSIVE: PASS'
fi

bash /root/catkin_ws/src/hilserl_wa2/scripts/start_spacemouse_joy.sh
```

### 9.2 L：启动新的 live 故障 Server

```bash
python "$R10_LEARNER_ROOT/src/hilserl_wa2/scripts/r10_learner_server.py" \
  --task bottle_pick \
  --network-config "$R10_LEARNER_ROOT/src/hilserl_wa2/configs/network/local.yaml" \
  --manifest "$R10_LEARNER_ROOT/runs/r10_preflight/learner_manifest.json" \
  --capacity 5000 \
  --republish-s 1.0 \
  --output "$R10_RUN/live_disconnect_server.json" \
  --status-file "$R10_RUN/live_disconnect_status.json" \
  2>&1 | tee "$R10_RUN/live_disconnect_server.log"
```

等待 READY，O 重新 `nc` 两端口。

### 9.3 A：现场授权后启动 live-zero Actor

```bash
export R4_CONFIRM=YES

python /root/catkin_ws/src/hilserl_wa2/scripts/verify_r10_actor_remote.py \
  --task bottle_pick \
  --network-config /root/catkin_ws/src/hilserl_wa2/configs/network/local.yaml \
  --manifest /root/catkin_ws/runs/wa2_bottle_pick/r10_preflight/actor_manifest.json \
  --server-ip "$R10_LEARNER_IP" \
  --mode live-zero \
  --policy zero \
  --steps 100000 \
  --control-hz 50 \
  --require-network-update \
  --require-intervention-steps 0 \
  --max-total-translation-m 0.002 \
  --max-total-rotation-deg 0.2 \
  --expect-fault network_loss \
  --output "$R10_ACTOR_RUN/actor_live_disconnect.json" \
  2>&1 | tee "$R10_ACTOR_RUN/actor_live_disconnect.log"
```

Actor 稳定、收到 params 且执行至少 30 个 zero 步后，在 L 对 Python Server 按 `Ctrl+C`。不要停止 Docker 或 ROS master。

必须出现：

```text
POLICY=zero
INTERVENTION_STEPS=0
MOTION_WITHOUT_INTERVENTION=false
translation_m <= 0.002
rotation_deg <= 0.2
FAULT_REASON=network_loss
FAULT_DETAIL=server_disconnect 或 network_stale
NO_FURTHER_STEPS=true
STOP_OK=true
CLEAR_OK=true
ENV_CLOSED=true
CLIENT_STOPPED=true
AUTO_RECONNECT=false
R10_LIVE_DISCONNECT: PASS
```

出现任何位移超限、干预、旧策略继续执行、`STOP_OK/CLEAR_OK` 非 true：立即急停并判 `R10 GATE: FAIL`。

## 10. 双端归档

### 10.1 Learner L2

```bash
cp "$R10_LEARNER_ROOT/runs/r10_preflight/learner_manifest.json" "$R10_RUN/"
cp "$R10_LEARNER_ROOT/src/hilserl_wa2/configs/network/local.yaml" "$R10_RUN/network_config.yaml"
cp "$R10_LEARNER_ROOT/runs/r10_preflight/ufw_status_numbered.txt" "$R10_RUN/" 2>/dev/null || true

find "$R10_RUN" -type f ! -name SHA256SUMS -print0 \
  | sort -z \
  | xargs -0 sha256sum > "$R10_RUN/SHA256SUMS"

sha256sum -c "$R10_RUN/SHA256SUMS"
```

### 10.2 Actor A

```bash
cp /root/catkin_ws/runs/wa2_bottle_pick/r10_preflight/actor_manifest.json "$R10_ACTOR_RUN/"
cp /root/catkin_ws/src/hilserl_wa2/configs/network/local.yaml "$R10_ACTOR_RUN/network_config.yaml"

find "$R10_ACTOR_RUN" -type f ! -name SHA256SUMS -print0 \
  | sort -z \
  | xargs -0 sha256sum > "$R10_ACTOR_RUN/SHA256SUMS"

sha256sum -c "$R10_ACTOR_RUN/SHA256SUMS"
```

Orin 宿主机应能在 `$R10_ORIN_RUN` 看到 Actor 全部文件。将 Learner `$R10_RUN` 同步到 Orin 的阶段验收目录，最终编写实际联机日期对应的：

```text
/home/naviai/hilserl_orin/调试日志/阶段验收日志/<日期>_R10验收.md
```

## 11. 最终验收表

### 11.1 Learner

- [ ] GPU、SAC params signature、JIT action、Buffer 4 插入证据完整；
- [ ] UFW 仅允许 Actor `/32` 的 TCP 5588/5589；
- [ ] 每次 Server 都打印 CudaDevice、params signature、READY 与唯一 instance id；
- [ ] F0 status 为 1020/20、schema true、last ids 与 ordered digests 非空；
- [ ] F1/F2 Server instance id 不同，新进程不续旧 store；
- [ ] Learner 日志、JSON、manifest、network config 与 SHA256SUMS 完整。

### 11.2 Actor

- [ ] F0 handshake 在 env 第一步前 PASS；1020/20/1000、last id、digest 全部匹配；
- [ ] params callback 至少一次、签名一致、新鲜度 `<5 s`；
- [ ] upload RTT p95 `<3000 ms`，无 timeout；
- [ ] F1 断线 `network_loss`、无继续 step、Env/Client 关闭、fault dump 存在、无自动重连；
- [ ] F2 新 session 200/10，无旧 dump 混入；
- [ ] readonly 真图像、有限状态、动作忽略、机器人无运动；
- [ ] live-zero 干预 0、平移 `<=2 mm`、旋转 `<=0.2°`、stop/clear 成功；
- [ ] Actor 日志、JSON、dump、manifest、network config 与 SHA256SUMS 完整。

### 11.3 联合判定

全部完成才允许在验收日志写：

```text
R10 GATE: PASS
```

任一必需项失败则写：

```text
R10 GATE: FAIL — <一条可复现根因>
```

只完成 F0 fake 通信仍不是完整 R10 PASS；只有 F0、F1、F2、H0、H1 和双端归档全部通过才是阶段完成。

## 12. 常见现场错误

| 现象 | 判定与处理 |
|---|---|
| Server 启动约 3 分钟暂无 READY | SAC/JIT 初始化中；等待 `PARAMS_TREE_READY`，不要中断 |
| 5588 通、5589 不通 | 不启动 Actor；检查 Server 监听和 UFW |
| handshake mismatch | Actor 必须 `ENV_STEPS=0`；重新构建/比对 manifest，禁止忽略字段 |
| Actor 显示 update 但 Server 数量不对 | 以 Server count/last id/digest 为准，不以 `client.update()` 布尔值为证 |
| 第一次 params 未收到 | PUB/SUB slow join；Server 会每秒重发，不取消 callback Gate |
| 停 Server 后 Actor 继续 step | 立即判 FAIL；不得进行真机 Gate |
| 恢复 Server 收到旧 0..N | 旧队列被自动重放，判 FAIL |
| readonly 中机器人运动 | 立即急停，判 FAIL |
| live-zero 出现干预或位移超限 | 立即急停，判 FAIL |
| 两个 action publisher | 停止 teleop，只允许 Actor；重新做独占检查 |
