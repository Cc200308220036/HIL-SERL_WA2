# HIL-SERL Actor在Jetson Orin上的安装流程

## 1. 适用范围

本文记录已在当前Jetson Orin上实际验收通过的HIL-SERL Actor环境。

目标架构：

- Jetson Orin：运行Actor、JAX推理、ROS机器人接口、相机与人工干预。
- 独立笔记本：运行Learner。
- Actor与Learner：通过Agentlace/ZeroMQ交换经验数据和网络参数。

本文只覆盖Orin端Actor基础环境。以下内容不属于本次安装完成范围：

- WA2 Gymnasium Env实现。
- ROS控制接口映射。
- SpaceMouse实际接入。
- 笔记本Learner环境。
- Actor–Learner跨机通信和真机训练闭环。

## 2. 已验收基线

```text
硬件架构：aarch64
容器镜像：ros1_docker:latest
容器名称：hilserl
Conda环境：hil-actor
环境路径：/opt/conda/envs/hil-actor
Python：3.10.20
HIL-SERL源码：/root/catkin_ws/src/hil-serl-main
```

宿主机与容器目录对应：

```text
/home/naviai/hilserl_orin/catkin_ws
    ↕ 双向读写
/root/catkin_ws
```

依赖清单：

```text
/home/naviai/hilserl_orin/docker/actor-requirements.txt
```

Agentlace wheel：

```text
/home/naviai/hilserl_orin/artifacts/wheels/
  agentlace-0.1.3-py3-none-any.whl
  SHA256SUMS
```

## 3. 创建并启动容器

如果容器尚未创建，在宿主机执行：

```bash
cd /home/naviai/hilserl_orin/docker

docker compose -f docker-compose.hilserl.yml \
  up -d --no-build hilserl
```

检查：

```bash
docker ps --filter name=hilserl

docker inspect hilserl --format \
  '{{range .Mounts}}{{println .Source "->" .Destination "rw=" .RW}}{{end}}'
```

应确认：

```text
镜像：ros1_docker:latest
/home/naviai/hilserl_orin/catkin_ws -> /root/catkin_ws
```

注意：

- `hilserl` 与 `assembly` 是两个独立容器。
- 两个容器不要同时控制机器人、相机或SpaceMouse。
- Compose使用host网络，Actor和Learner端口不得被其他进程占用。

## 4. 创建Conda环境

进入容器：

```bash
docker exec -it hilserl bash
```

加载Conda：

```bash
source /opt/conda/etc/profile.d/conda.sh
```

创建环境：

```bash
conda create -y -n hil-actor \
  --override-channels \
  --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/ \
  --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ \
  python=3.10.20 \
  pip=26.2 \
  setuptools=83.0.0 \
  wheel=0.47.0 \
  packaging=26.2
```

激活：

```bash
conda activate hil-actor
```

检查：

```bash
which python
python --version
uname -m
```

预期：

```text
/opt/conda/envs/hil-actor/bin/python
Python 3.10.20
aarch64
```

## 5. 安装OpenCV

OpenCV必须使用已验证的Conda ARM64构建，不要通过pip安装，也不要手工复制 `.so` 文件。

```bash
conda install -y -n hil-actor \
  --freeze-installed \
  --override-channels \
  --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ \
  --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/ \
  "opencv=4.10.0=py310h13dd31d_7"
```

验证：

```bash
conda activate hil-actor

python -c \
  "import cv2; print(cv2.__version__); print(cv2.__file__)"
```

预期版本：

```text
4.10.0
```

不得出现：

```text
libopencv_*.so: cannot open shared object file
```

## 6. 将安装文件传入容器

当前Compose只挂载 `catkin_ws`，调试日志和wheel不直接出现在容器中。

### 6.1 创建目标目录

在宿主机执行：

```bash
docker exec hilserl sh -c \
  'mkdir -p /opt/hilserl/wheels /opt/hilserl/verify'
```

### 6.2 传输requirements

当前容器挂载了多个只读cuDNN文件，`docker cp` 可能因只读挂载失败。建议通过标准输入传输：

```bash
docker exec -i hilserl sh -c \
  'cat > /opt/hilserl/actor-requirements.txt' \
  < /home/naviai/hilserl_orin/docker/actor-requirements.txt
```

### 6.3 传输Agentlace wheel和校验文件

```bash
docker exec -i hilserl sh -c \
  'cat > /opt/hilserl/wheels/agentlace-0.1.3-py3-none-any.whl' \
  < /home/naviai/hilserl_orin/artifacts/wheels/agentlace-0.1.3-py3-none-any.whl

docker exec -i hilserl sh -c \
  'cat > /opt/hilserl/wheels/SHA256SUMS' \
  < /home/naviai/hilserl_orin/artifacts/wheels/SHA256SUMS
```

### 6.4 传输验收脚本

```bash
docker exec -i hilserl sh -c \
  'cat > /opt/hilserl/verify/verify_hil_actor.py' \
  < /home/naviai/hilserl_orin/docker/verify_hil_actor.py

docker exec -i hilserl sh -c \
  'cat > /opt/hilserl/verify/verify_agentlace.py' \
  < /home/naviai/hilserl_orin/docker/verify_agentlace.py
```

检查：

```bash
docker exec hilserl ls -lh \
  /opt/hilserl/actor-requirements.txt \
  /opt/hilserl/wheels/agentlace-0.1.3-py3-none-any.whl \
  /opt/hilserl/wheels/SHA256SUMS \
  /opt/hilserl/verify/verify_hil_actor.py \
  /opt/hilserl/verify/verify_agentlace.py
```

## 7. 安装Python依赖

进入容器并激活环境：

```bash
docker exec -it hilserl bash
source /opt/conda/etc/profile.d/conda.sh
conda activate hil-actor
```

安装完整锁定依赖：

```bash
python -m pip install \
  --no-deps \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  --extra-index-url https://pypi.org/simple \
  -r /opt/hilserl/actor-requirements.txt
```

使用 `--no-deps` 的原因：

- 清单已列出全部pip依赖及其准确版本。
- 防止pip解析器升级JAX、NumPy、protobuf或ml-dtypes。
- 防止TensorFlow依赖解析改变已验收版本组合。
- 防止pip尝试安装ARM64不适用的OpenCV wheel。

## 8. 安装Agentlace

先校验wheel：

```bash
cd /opt/hilserl/wheels
sha256sum --check SHA256SUMS
```

预期：

```text
agentlace-0.1.3-py3-none-any.whl: OK
```

安装：

```bash
python -m pip install \
  --no-deps \
  /opt/hilserl/wheels/agentlace-0.1.3-py3-none-any.whl
```

正确依赖元数据应为：

```text
gym>=0.26.0
lz4
numpy
opencv-python
pyzmq
typing_extensions
```

不要安装：

```text
typing
zmq
```

Python 3.10已经内置 `typing`；ZeroMQ的正确distribution名称是 `pyzmq`。

## 9. 安装HIL-SERL源码

### 9.1 检查源码

```bash
ls /root/catkin_ws/src/hil-serl-main/serl_launcher
```

### 9.2 检查serl_launcher元数据

`serl_launcher/setup.py` 的依赖声明应包含：

```python
install_requires=[
    "pyzmq",
    "typing_extensions",
    "opencv-python",
    "lz4",
    "agentlace==0.1.3",
]
```

不得保留：

```text
"zmq"
"typing"
Git URL形式的Agentlace
```

### 9.3 Editable安装

```bash
python -m pip install \
  --no-deps \
  -e /root/catkin_ws/src/hil-serl-main/serl_launcher
```

使用editable模式后，宿主机对 `serl_launcher` 源码的修改会立即在容器环境中生效。

当前Actor阶段不建议安装完整 `serl_robot_infra`：

- 该包主要面向Franka和RealSense示例。
- 会引入 `pyrealsense2`、Modbus、Franka、HID等硬件依赖。
- WA2应使用后续实现的自定义Gymnasium Env和现有ROS接口。

## 10. 配置Actor运行环境变量

创建Conda激活脚本：

```bash
mkdir -p "${CONDA_PREFIX}/etc/conda/activate.d"

cat > "${CONDA_PREFIX}/etc/conda/activate.d/hil-actor.sh" <<'EOF'
export PYTHONNOUSERSITE=1
export XLA_FLAGS=--xla_gpu_autotune_level=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.1
export TF_CPP_MIN_LOG_LEVEL=2
EOF
```

重新激活：

```bash
conda deactivate
conda activate hil-actor
```

确认：

```bash
printf '%s\n' \
  "$PYTHONNOUSERSITE" \
  "$XLA_FLAGS" \
  "$XLA_PYTHON_CLIENT_PREALLOCATE" \
  "$XLA_PYTHON_CLIENT_MEM_FRACTION"
```

## 11. 完整验收

### 11.1 依赖一致性

```bash
python -m pip check
```

预期：

```text
No broken requirements found.
```

### 11.2 基线和JAX GPU

```bash
python /opt/hilserl/verify/verify_hil_actor.py
```

预期关键输出：

```text
OpenCV import: 4.10.0
JAX devices: [CudaDevice(id=0)]
HIL-ACTOR BASELINE: PASS
```

### 11.3 Agentlace本机通信

```bash
python /opt/hilserl/verify/verify_agentlace.py
```

预期：

```text
AGENTLACE LOCAL COMMUNICATION: PASS
```

### 11.4 Actor核心导入

JAX必须先于TensorFlow导入：

```bash
python - <<'PY'
import jax

print("JAX:", jax.__version__)
print("JAX devices:", jax.devices())

import tensorflow
import tf_keras
import tensorflow_probability
import distrax
import agentlace
import wandb
import matplotlib
import serl_launcher

from agentlace.trainer import TrainerClient
from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.utils.launcher import make_trainer_config

print("TensorFlow:", tensorflow.__version__)
print("TF-Keras:", tf_keras.__version__)
print("TensorFlow Probability:", tensorflow_probability.__version__)
print("Distrax:", distrax.__version__)
print("WandB:", wandb.__version__)
print("serl_launcher:", serl_launcher.__file__)
print("HIL-SERL ACTOR CORE IMPORT: PASS")
PY
```

验收条件：

- JAX版本为 `0.4.35`。
- 至少出现一个 `CudaDevice`。
- TensorFlow与TF-Keras均为 `2.21.0`。
- `serl_launcher` 指向 `/root/catkin_ws/src/hil-serl-main`。
- 最终输出 `HIL-SERL ACTOR CORE IMPORT: PASS`。

## 12. 关键安装注意事项

### 12.1 不要直接安装上游requirements

不要直接执行：

```bash
pip install -r \
  /root/catkin_ws/src/hil-serl-main/serl_launcher/requirements.txt
```

上游文件中包含与当前基线冲突的SciPy、Gymnasium、TensorFlow等版本，还包含Learner或开发工具依赖。

### 12.2 JAX必须先于TensorFlow导入

错误顺序：

```python
import tf_keras
import tensorflow_probability
import distrax
```

可能导致：

```text
File already exists in database: xla/xla_data.proto
Aborted (core dumped)
```

正确顺序：

```python
import jax
import tensorflow
import tf_keras
import tensorflow_probability
import distrax
```

HIL-SERL当前 `train_rlpd.py` 本身先导入JAX，符合要求。

### 12.3 不要混用pip OpenCV

OpenCV必须保留Conda版本，不要执行：

```bash
pip install --force-reinstall opencv-python
```

否则可能导致ARM64 wheel不可用、动态库缺失或覆盖Conda OpenCV。

### 12.4 不要安装错误的zmq和typing

```text
错误：zmq、typing
正确：pyzmq、Python内置typing
```

### 12.5 Gym警告

导入 `gym==0.26.2` 时会提示项目已停止维护。该警告不影响当前基线验收。新增WA2 Env应优先使用：

```text
gymnasium==1.2.2
```

### 12.6 Agentlace测试端口

本机验收脚本使用：

```text
5598
5599
```

如果端口被占用，通信测试会失败。运行测试前不要启动另一个占用相同端口的Trainer。

### 12.7 当前环境位于容器可写层

- `docker stop` 后再 `docker start`：环境保留。
- 删除容器：环境丢失。
- `docker compose down` 并重新创建：环境需要重装。

完成WA2真机全流程后，应将本文件中的最终步骤固化到Dockerfile并构建新镜像。

## 13. 当前验收结论

截至2026-08-06，当前Orin环境已达到：

```text
Python 3.10.20
aarch64
No broken requirements found.
agentlace wheel SHA-256: OK
OpenCV 4.10.0: PASS
JAX 0.4.35 + CudaDevice(id=0): PASS
Agentlace local communication: PASS
HIL-SERL Actor core import: PASS
```

该结论表示Orin端Actor基础软件环境配置成功，不代表WA2机器人真机控制和跨机训练闭环已经完成。
