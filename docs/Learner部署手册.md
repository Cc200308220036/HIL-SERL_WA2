# HIL-SERL Learner 快速部署手册

更新时间：2026-09-01  
适用工程：WA2 `bottle_pick` / R13 HIL-SERL  
目标平台：Ubuntu 22.04、Linux x86_64、NVIDIA GPU、Conda `hil-learner`

## 1. 文档目标与运行边界

本手册用于在新的 Learner 笔记本上，从空白环境开始完成：

1. Conda/Python 工具链创建；
2. `hil-learner.lock.txt` 固定依赖安装；
3. Agentlace、`serl_launcher` 本地组件安装；
4. ResNet10 模型部署；
5. 环境隔离、依赖和 GPU 验收；
6. Demo、双端 Manifest 和网络预检；
7. R13 Fake 预检及正式 Learner 训练启动。

Actor 与 Learner 的边界必须保持清晰：

| 角色 | 工作目录 | 环境 | 责任 |
|---|---|---|---|
| Learner | `HILSERL_Learner/` | Conda `hil-learner` | Demo/Replay Buffer、SAC/RLPD 更新、checkpoint、参数广播 |
| Actor | Orin 容器 `/root/catkin_ws` | Conda `hil-actor` | ROS、相机、ServoL、SpaceMouse、分类器推理、动作执行和 transition 上传 |

Learner 不安装或运行 `rospy`、WA2 控制器、RealSense、SpaceMouse、ServoL。Actor 不运行 `r13_learner_train.py`。

## 2. 必须随工程复制的内容

新笔记本至少需要完整复制：

```text
HILSERL_Learner/
├── artifacts/
│   ├── models/resnet10_params.pkl
│   └── wheels/agentlace-0.1.3-py3-none-any.whl
├── requirements/hil-learner.lock.txt
├── scripts/
│   ├── activate_hil_learner.sh
│   ├── build_source_sha256s.py
│   ├── verify_hil_learner_dependencies.py
│   └── verify_hil_learner_gpu.py
├── src/hilserl_wa2/
├── src/hil-serl-main/serl_launcher/
└── SOURCE_SHA256SUMS
```

Demo、checkpoint 和 Buffer 缓存不属于依赖安装包，但恢复既有训练时必须另行完整复制对应的 `runs/<task>/<run_id>/`。

推荐沿用已验收路径：

```text
/home/cyw/orin_hilserl/HILSERL_Learner
```

当前 `activate_hil_learner.sh` 和依赖验收脚本以该路径为正式根目录。如果新笔记本用户名或工程路径不同，必须先同步修改这两个脚本中的固定路径，再执行验收，不能通过软链接或错误 `PYTHONPATH` 混用另一份源码。

## 3. 主机前置条件

安装 Python 依赖前先确认：

```bash
uname -m
lsb_release -ds
nvidia-smi
```

要求：

- `uname -m` 为 `x86_64`；
- Ubuntu 22.04；
- NVIDIA 驱动可以正常识别独立显卡；
- 驱动支持本项目锁定的 CUDA 12 runtime；
- BIOS/系统未禁用独显；
- 训练目录有足够磁盘空间。

已验收机器为 RTX 4060，历史驱动版本为 `580.173.02`。驱动是主机组件，不由 pip lock 安装。受控终端或沙箱可能看不到 `/dev/nvidia*`，最终 GPU Gate 必须在新笔记本的真实交互终端执行。

## 4. 校验迁移文件

进入 Learner 根目录：

```bash
cd /home/cyw/orin_hilserl/HILSERL_Learner
sha256sum artifacts/wheels/agentlace-0.1.3-py3-none-any.whl
sha256sum artifacts/models/resnet10_params.pkl
```

期望：

```text
Agentlace wheel:
1a800cc341f03eb6844273571ba26a265920fa1b5a698acc3d954438cbb72d32

ResNet10:
175745d43d30233eb01b5369465d1c24c11b8ee71ccb734cc1c1bca13e07f57b
```

校验源码迁移包：

```bash
python scripts/build_source_sha256s.py --check
```

若失败，先确认文件是否传输不完整。不要在来源不明的源码上直接重新生成校验清单来掩盖差异。

## 5. 创建 Conda 环境

不要克隆其他仿真、ROS 或开发环境。创建独立环境：

```bash
source /home/cyw/anaconda3/etc/profile.d/conda.sh

conda create -y -n hil-learner \
  python=3.10.20 \
  pip=26.1.2 \
  setuptools=83.0.0 \
  wheel=0.47.0 \
  packaging=26.2

conda activate hil-learner
python --version
python -m pip --version
```

如果使用 Miniconda 或其他安装目录，应把 `conda.sh` 路径替换为实际路径。若 `hil-learner` 已存在，不要直接覆盖或删除；先改用新的环境名做恢复演练，或者确认旧环境可以废弃。

## 6. 安装 Learner 专用 lock

[hil-learner.lock.txt](../HILSERL_Learner/requirements/hil-learner.lock.txt) 是从启用 `PYTHONNOUSERSITE=1` 后的已验收环境生成的 Linux x86_64 版本锁，包含 95 个 pip 包：

- JAX/JAXLIB/CUDA plugin/PJRT `0.4.35`；
- 11 个固定版本的 NVIDIA CUDA 12 runtime wheel；
- TensorFlow `2.21.0`、TFP `0.25.0`、`tf_keras 2.21.0`；
- Flax、Optax、Orbax、Gym/Gymnasium；
- OpenCV、NumPy、SciPy 等完整传递依赖。

安装前隔离用户级包，然后按 lock 精确安装：

```bash
cd /home/cyw/orin_hilserl/HILSERL_Learner
conda activate hil-learner
export PYTHONNOUSERSITE=1

python -m pip install --no-deps \
  -r requirements/hil-learner.lock.txt
```

必须使用 `--no-deps`。Lock 已列出完整依赖闭包；允许 pip resolver 自行选择依赖可能造成 `jax==0.4.35` 与 `jaxlib==0.4.34`、Gymnasium 或 SciPy 版本漂移。

`actor-requirements.txt` 和 `orin-hil-actor.lock.txt` 仅作为 Orin Actor 基线保留，不用于创建新 Learner 环境。上游 `serl_launcher/requirements.txt` 也不能直接安装，因为其中的 Gymnasium、SciPy 版本与当前双端基线冲突。

## 7. 安装本地组件

### 7.1 Agentlace

必须安装工程随附的修正版 wheel：

```bash
python -m pip install --force-reinstall --no-deps \
  artifacts/wheels/agentlace-0.1.3-py3-none-any.whl
```

不要用 PyPI 上同名包替代。

### 7.2 SERL Launcher

从工程源码做 editable 安装：

```bash
python -m pip install --no-deps --editable \
  src/hil-serl-main/serl_launcher
```

### 7.3 检查安装来源

```bash
python - <<'PY'
import importlib.metadata as m
for name in (
    "agentlace", "serl_launcher", "jax", "jaxlib",
    "jax-cuda12-plugin", "jax-cuda12-pjrt",
):
    print(name, m.version(name))
PY
```

期望：

```text
agentlace 0.1.3
serl_launcher 0.1.2
jax 0.4.35
jaxlib 0.4.35
jax-cuda12-plugin 0.4.35
jax-cuda12-pjrt 0.4.35
```

## 8. 部署 ResNet10 权重

SERL 默认读取 `~/.serl/resnet10_params.pkl`。先检查目标，再创建指向工程资产的链接：

```bash
mkdir -p /home/cyw/.serl
ls -l /home/cyw/.serl/resnet10_params.pkl 2>/dev/null || true
sha256sum -L /home/cyw/.serl/resnet10_params.pkl 2>/dev/null || true
```

若目标不存在：

```bash
ln -s \
  /home/cyw/orin_hilserl/HILSERL_Learner/artifacts/models/resnet10_params.pkl \
  /home/cyw/.serl/resnet10_params.pkl
```

若目标已存在但 SHA256 不同，先人工备份并确认来源，禁止直接覆盖。最终验证：

```bash
sha256sum -L /home/cyw/.serl/resnet10_params.pkl
```

## 9. 激活正式 Learner 环境

以后每个 Learner 终端都从以下命令开始：

```bash
cd /home/cyw/orin_hilserl/HILSERL_Learner
source scripts/activate_hil_learner.sh
```

该脚本会：

- 激活 Conda `hil-learner`；
- 设置 `HILSERL_LEARNER_ROOT` 和 `PYTHONPATH`；
- 设置 `PYTHONNOUSERSITE=1`；
- 清除 ROS2 overlay 环境变量；
- 设置 JAX 显存和 CUDA 运行参数；
- 清除可能残留的 `JAX_PLATFORMS=cpu`。

检查隔离是否生效：

```bash
test "$CONDA_DEFAULT_ENV" = hil-learner
test "$PYTHONNOUSERSITE" = 1
python -c 'import site; print(site.ENABLE_USER_SITE); assert not site.ENABLE_USER_SITE'
```

## 10. 依赖与 GPU 验收

### 10.1 依赖 Gate

```bash
python -m pip check
python scripts/verify_hil_learner_dependencies.py
```

必须看到：

```text
No broken requirements found.
HIL_LEARNER_DEPENDENCIES: PASS
```

该 Gate 同时检查 Python、Conda 环境、核心包版本、源码导入位置、Agentlace wheel 和 ResNet10 SHA256。

### 10.2 GPU Gate

在可以访问 NVIDIA 设备的真实终端执行：

```bash
nvidia-smi
python scripts/verify_hil_learner_gpu.py
```

必须看到 JAX GPU/CUDA device、GPU 矩阵乘法结果，以及：

```text
HIL_LEARNER_GPU: PASS
```

出现 `CpuDevice`、驱动不可见、CUDA 初始化失败或 TensorFlow/TFP import 失败时，不允许启动正式训练。

## 11. 训练前数据和双端一致性 Gate

### 11.1 设置本次运行目录

```bash
export R13_LEARNER_ROOT=/home/cyw/orin_hilserl/HILSERL_Learner
export R13_RUN_ID="$(date +%Y%m%d_%H%M%S)_r13"
export R13_RUN="$R13_LEARNER_ROOT/runs/wa2_bottle_pick/$R13_RUN_ID"
mkdir -p "$R13_RUN"/{checkpoints,logs,demos}
echo "$R13_RUN_ID"
```

Actor 必须使用 Learner 打印的同一个 `R13_RUN_ID`，不能在 Actor 上再次执行 `date`。

### 11.2 准备原生 7D Demo

正式 R13 使用 Actor 当场采集的 7D Demo bundle。Bundle 必须同时包含：

```text
demo.pkl
bundle.json
episodes/ep000.pkl ... ep019.pkl
```

分局文件必须在内存较充足的 Actor 端用 `split_r13_demo_pkl.py` 生成。16 GB Learner 禁止直接对大型总 `demo.pkl` 做 `pickle.load`。

把完整 bundle 复制到 `$R13_RUN/demos/` 后：

```bash
export R13_DEMO="$R13_RUN/demos/<实际的7D_Demo目录>"

python "$R13_LEARNER_ROOT/src/hilserl_wa2/scripts/verify_r13_demo_load.py" \
  --task bottle_pick \
  --bundle "$R13_DEMO" \
  --expect-episodes 20 \
  --require-real-images
```

必须看到：

```text
ACTION_DIM=7
R13_DEMO_LOAD: PASS
```

### 11.3 生成 Learner Manifest

```bash
python "$R13_LEARNER_ROOT/src/hilserl_wa2/scripts/build_r13_manifest.py" \
  --repo "$R13_LEARNER_ROOT" \
  --task bottle_pick \
  --network-config "$R13_LEARNER_ROOT/src/hilserl_wa2/configs/network/local.yaml" \
  --role learner \
  --demo-pkl "$R13_DEMO/demo.pkl" \
  --output "$R13_RUN/manifest_learner.json"
```

Actor 端使用同一 Demo、任务和网络配置生成 `manifest_actor.json`。把 Actor Manifest 复制到 Learner 后比对：

```bash
python "$R13_LEARNER_ROOT/src/hilserl_wa2/scripts/build_r13_manifest.py" \
  --compare \
  "$R13_RUN/manifest_learner.json" \
  "$R13_RUN/manifest_actor.json"
```

必须看到 `R13_MANIFEST_MATCH: PASS`。任何任务、空间、源码、协议、Agentlace wheel、Demo 或时间尺度不一致都禁止进入真机训练。

## 12. R13 Learner 启动

Learner 必须先于 Actor 启动。默认端口：

- TCP 5588：transition/request；
- 5589：参数广播。

只允许 Actor 的局域网 IPv4 访问，不做公网端口转发。

### 12.1 Fake 数据流预检

首次部署新笔记本必须先做 Fake 预检：

```bash
python "$R13_LEARNER_ROOT/src/hilserl_wa2/scripts/r13_learner_train.py" \
  --task bottle_pick \
  --network-config "$R13_LEARNER_ROOT/src/hilserl_wa2/configs/network/local.yaml" \
  --manifest "$R13_RUN/manifest_learner.json" \
  --demo-path "$R13_DEMO/demo.pkl" \
  --checkpoint-path "$R13_RUN/checkpoints" \
  --output "$R13_RUN" \
  --status-file "$R13_RUN/status.json" \
  --mode fake --debug \
  --training-starts 50 \
  --steps-per-update 5 \
  --checkpoint-period 20 \
  --max-learner-steps 40 \
  --action-scale 1.0 \
  --end-episode
```

首次启动不要加 `--resume`，并确保 checkpoint 目录为空。等待打印 `R13_LEARNER_READY` 后才能启动 Fake Actor。Fake 双端需验证：握手、transition、人工干预数据、参数广播、checkpoint、Buffer snapshot 和断线 fail-closed。

### 12.2 正式 Live 训练

如果 Fake 预检已经在同一运行目录产生合法 checkpoint，正式训练使用 `--resume`：

```bash
python "$R13_LEARNER_ROOT/src/hilserl_wa2/scripts/r13_learner_train.py" \
  --task bottle_pick \
  --network-config "$R13_LEARNER_ROOT/src/hilserl_wa2/configs/network/local.yaml" \
  --manifest "$R13_RUN/manifest_learner.json" \
  --demo-path "$R13_DEMO/demo.pkl" \
  --checkpoint-path "$R13_RUN/checkpoints" \
  --output "$R13_RUN" \
  --status-file "$R13_RUN/status.json" \
  --mode live --debug --resume \
  --training-starts 100 \
  --steps-per-update 50 \
  --checkpoint-period 1000 \
  --action-scale 1.0 \
  --end-episode
```

若直接创建全新的 Live run 且 checkpoint 目录为空，则不能传 `--resume`。只有 `checkpoints/timescale.json` 与当前配置完全一致、且存在合法 checkpoint 时才允许续训。

Learner 输出 `R13_LEARNER_READY` 后，才能启动 Actor。Actor 的分类器 checkpoint 仅在 Actor 端加载，Learner 不加载奖励分类器做在线推理。

### 12.3 安全停机与恢复

正常停止 Learner 时使用一次 `Ctrl+C`，等待：

```text
BUFFER_SNAPSHOT reason=shutdown
R13_LEARNER_STOPPED
```

不要在 Buffer snapshot 写入过程中强制断电。恢复训练前保留并核对：

```text
checkpoints/
demo_buffer_cache/
online_buffer_cache/
manifest_learner.json
status.json
metrics.jsonl
demos/
```

`demo_buffer_cache` 中可能包含大量人工干预 transition，不能当作可随时重建的普通缓存删除。

## 13. 新笔记本最终验收清单

只有以下项目全部通过，才算 Learner 已具备开始训练的条件：

- [ ] Ubuntu/x86_64 与 NVIDIA GPU 正常；
- [ ] Conda `hil-learner`、Python 3.10.20；
- [ ] `hil-learner.lock.txt` 的 95 个包安装完成；
- [ ] Agentlace 0.1.3 本地修正版 wheel 已安装且 SHA256 正确；
- [ ] `serl_launcher` 0.1.2 从当前工程 editable 安装；
- [ ] ResNet10 模型 SHA256 正确且默认路径可读；
- [ ] `PYTHONNOUSERSITE=1`，没有 ROS2/user-site 污染；
- [ ] `pip check` 通过；
- [ ] `HIL_LEARNER_DEPENDENCIES: PASS`；
- [ ] `HIL_LEARNER_GPU: PASS`，JAX 实际在 GPU 运算；
- [ ] 原生 7D Demo 和 20 个分局文件通过加载验证；
- [ ] Actor/Learner Manifest 比对通过；
- [ ] Fake 数据流、参数广播、checkpoint 和断线保护通过；
- [ ] Learner 先启动并进入 `R13_LEARNER_READY`，再启动 Actor。

## 14. 常见问题

### JAX 只看到 CPU

检查 `JAX_PLATFORMS`、`nvidia-smi`、驱动和 CUDA wheel；重新进入真实终端执行 `source scripts/activate_hil_learner.sh`。不要用 CPU Gate 代替正式 GPU 验收。

### `pip check` 出现 ROS2、Diffusers 等无关冲突

通常是用户 site-packages 泄漏。确认：

```bash
echo "$PYTHONNOUSERSITE"
python -c 'import site; print(site.ENABLE_USER_SITE)'
```

期望分别为 `1` 和 `False`。

### Demo 验证长时间没有逐局输出

检查是否缺少 `episodes/epXXX.pkl`。不要让 16 GB Learner 加载大型总 `demo.pkl`；回到 Actor 端拆分后重新复制完整 bundle。

### Manifest 不一致

停止双端启动。确认两端任务配置、共享源码、Agentlace wheel、Demo 和时间尺度一致，然后分别重建 Manifest；不能手工修改 JSON 绕过比较。

### `--resume` 被拒绝

确认 checkpoint 目录存在合法 checkpoint 和 `timescale.json`。新 run 第一次启动不能使用 `--resume`；旧时间尺度 checkpoint 不能混入当前 run。
