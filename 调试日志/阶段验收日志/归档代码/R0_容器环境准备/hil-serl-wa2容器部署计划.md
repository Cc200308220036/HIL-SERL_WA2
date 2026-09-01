# HIL-SERL WA2 容器部署前置分析与后续计划

## 1. 文档目的与本次操作边界

本文记录当前设备上 `assembly` Docker 容器及其 `hil-jax` Conda 环境的只读分析结果，并规划下一阶段的 HIL-SERL 部署工作。

本次已遵守以下边界：

- 未修改宿主机、Docker 镜像、Docker 容器、Conda 环境或项目源码。
- 未安装、升级、卸载任何软件包。
- 未创建、停止、重启或提交任何容器。
- 唯一新增内容为本文档。

下一阶段目标（本次不执行）：

- 以 `assembly` 对应的 Dockerfile/Compose 配置为基础制作新镜像。
- 创建名为 `hilserl` 的新 Docker 容器。
- 在新容器中创建名为 `hil-serl` 的 Conda 环境。
- 将当前 `hil-jax` 中经确认可复现、确有需要的依赖写入镜像构建过程。
- 对 JAX GPU、HIL-SERL、ROS 和硬件设备访问进行分层验证。

## 2. 当前 Docker 环境

### 2.1 `assembly` 容器概况

- 状态：运行中。
- Compose 服务名及容器名：`assembly`。
- 镜像：`ros1_docker:latest`。
- 当前镜像 ID：`sha256:5d7f3edb272d2d8b1b7d9ed47a99d79ac9065cb85ed6bc5bb01d7a9cd36902a1`。
- 架构：Linux ARM64/aarch64。
- 容器系统：Ubuntu 22.04。
- 镜像大小：约 41.75 GB。
- 工作目录：`/root/catkin_ws`。
- 默认命令：`/bin/bash`。
- NVIDIA 容器运行时：已启用。
- 网络模式：`host`。
- 特权模式：已启用。
- IPC 模式：`private`。
- 共享内存：64 MiB。
- 重启策略：`no`。

GPU/加速栈的只读观测：

- GPU：Jetson Orin（`nvgpu`）。
- NVIDIA 驱动接口报告版本：`540.4.0`。
- CUDA Toolkit：12.2（`nvcc` 报告 `V12.2.140`）。
- 运行时使用 cuDNN 9.3.0；Compose 将宿主机 cuDNN 9.3.0 动态库只读挂载到容器。

### 2.2 Compose 中与后续容器有关的配置

现有源文件：

- `/home/naviai/ros_docker_test/docker-compose.yml`
- `/home/naviai/ros_docker_test/Dockerfile`

主要配置：

- 通过 `runtime: nvidia` 使用 GPU。
- 使用 `network_mode: host`，服务于 ROS 通信与设备发现。
- 使用 `privileged: true`。
- 将宿主机 `/dev` 和 `/dev/bus/usb` 读写挂载到容器，用于相机、USB 和其他设备。
- 将 `/tmp/.X11-unix` 挂载到容器，用于 GUI。
- 将宿主机 `./catkin_ws` 挂载到 `/root/catkin_ws`。
- 设置 ROS Master、ROS 主机地址、显示、NVIDIA 和动态库搜索路径相关环境变量。
- 将宿主机 cuDNN 9.3.0 的多个库文件以只读方式逐项挂载。

这组配置能够覆盖 ROS、GPU、GUI 和 USB 设备需求，但权限很高。后续应先按现有配置建立功能基线，再根据 HIL-SERL 实际需要评估是否能缩小设备映射和特权范围。

### 2.3 `assembly` Dockerfile 基础

当前 Dockerfile 使用：

```dockerfile
FROM 10.51.33.201:30002/navi_project/sensor:22CUDA
```

其主要内容包括：

- Ubuntu 22.04、CUDA 和源码构建的 ROS Noetic 基础。
- Miniconda 安装于 `/opt/conda`。
- 清华 Conda/PyPI 镜像配置，Conda channel 优先级为 `flexible`。
- 多个 Python 3.10 Conda 环境：`camera`、`rl_screw`、`rlpd`、`sam`、`seg`、`fdpose`。
- Jetson ARM64 PyTorch、TorchVision、cuSPARSELt、cuDNN 9、nvdiffrast 和 pytorch3d 等内容。
- 默认 PATH 指向 `camera` 环境。

注意事项：

- `hil-jax` 不在该 Dockerfile 中。
- 容器创建于 2026-07-14，而 `hil-jax` 创建于 2026-07-31，因此 `hil-jax` 是容器启动后加入的可变状态，无法仅靠当前 Dockerfile 重建。
- Dockerfile 第 280～289 行附近的 shell 续行需要在后续复制时复核；镜像历史中未看到预期的 `ck` 别名，不能假设源文件末段全部按注释意图执行。

### 2.4 当前项目中的 JAX 探测 Dockerfile

`/home/naviai/hilserl_orin/docker/Dockerfile.jax-probe` 是独立探测方案，不是 `assembly` 的源 Dockerfile。它与目标基线存在明显差异：

- 基础镜像为 `nvcr.io/nvidia/l4t-cuda:12.6.11-runtime`，而 `assembly` 当前实际使用 CUDA 12.2 基线。
- 使用 Miniforge，而 `assembly` 使用 Miniconda。
- 创建的环境名是 `jax_probe` 和 `hilserl`，与目标环境名 `hil-serl` 不一致。
- 使用未固定版本的 `"jax[cuda12]"`、Flax、Optax 和 Chex，未来重建时可能得到不同版本。

因此后续不应直接把该文件当作目标 Dockerfile；可以借鉴其分层、`pip check` 和探测思路，但应以 `assembly` 的 Dockerfile 为主线，并固定经过验证的版本。

## 3. `hil-jax` Conda 环境分析

### 3.1 基本信息

- 环境路径：`/opt/conda/envs/hil-jax`。
- Python：3.10.20。
- pip：26.2。
- Conda：26.3.2。
- 平台：`linux-aarch64`。
- 环境创建时间：2026-07-31 13:19。
- Conda channel 优先级：`flexible`。
- Conda channels：
  - `https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/`
  - `https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/`
  - `defaults`

Conda 修订历史只有 `rev 0`。这说明 Conda 只负责创建 Python 基础环境，后续业务依赖主要通过 pip 安装，Conda 修订历史无法追踪这些 pip 操作。

### 3.2 Conda 基础包

环境创建时安装的 Conda 包如下：

```text
_openmp_mutex=4.5
bzip2=1.0.8
ca-certificates=2026.7.22
ld_impl_linux-aarch64=2.46.1
libexpat=2.8.1
libffi=3.7.0
libgcc=16.1.0
libgcc-ng=16.1.0
libgomp=16.1.0
liblzma=5.8.3
libnsl=2.0.1
libsqlite=3.53.4
libuuid=2.42.2
libxcrypt=4.4.36
libzlib=1.3.2
ncurses=6.6
openssl=3.6.3
packaging=26.2
pip=26.2
python=3.10.20
readline=8.3
setuptools=83.0.0
tk=8.6.13
tzdata=2026c
wheel=0.47.0
zstd=1.5.7
```

### 3.3 当前 pip 包完整清单

```text
absl-py==2.5.0
agentlace==0.1.3
attrs==26.1.0
certifi==2026.7.22
charset-normalizer==3.4.9
chex==0.1.88
cloudpickle==3.1.2
decorator==5.3.1
distrax==0.1.5
dm-tree==0.1.10
einops==0.8.1
etils==1.13.0
Farama-Notifications==0.0.6
flax==0.10.2
fsspec==2026.7.0
gast==0.7.0
gymnasium==1.2.2
humanize==4.16.0
idna==3.18
ImageIO==2.37.2
importlib_resources==7.1.0
jax==0.4.35
jax-cuda12-pjrt==0.4.35
jax-cuda12-plugin==0.4.35
jaxlib==0.4.35
lz4==4.4.5
markdown-it-py==4.2.0
mdurl==0.1.2
ml_collections==1.1.0
ml_dtypes==0.5.4
msgpack==1.2.1
natsort==8.4.0
nest-asyncio==1.6.0
numpy==1.26.4
opt-einsum==3.3.0
optax==0.2.4
orbax-checkpoint==0.10.3
packaging==26.2
pillow==12.3.0
pip==26.2
protobuf==7.35.1
Pygments==2.20.0
PyYAML==6.0.3
pyzmq==27.1.0
requests==2.34.2
rich==15.0.0
scipy==1.15.3
setuptools==83.0.0
simplejson==4.1.1
six==1.17.0
tensorflow-probability==0.25.0
tensorstore==0.1.71
toolz==1.1.0
tqdm==4.70.0
typing_extensions==4.16.0
urllib3==2.7.0
wheel==0.47.0
wrapt==2.3.0
zipp==4.1.0
```

### 3.4 可推断的安装批次

根据包元数据时间可以重建大致顺序，但不能还原每条原始 pip 命令：

1. 13:19：创建 Python 3.10 环境并准备 pip、setuptools、wheel。
2. 13:44～14:37：分步安装 JAX GPU 栈：
   - `jax-cuda12-pjrt==0.4.35`
   - `scipy==1.15.3`
   - `ml_dtypes==0.5.4`
   - `jaxlib==0.4.35`
   - `jax-cuda12-plugin==0.4.35`
   - `jax==0.4.35`
   - `opt-einsum==3.3.0`
3. 16:59：安装 Flax/Optax/Chex 及其依赖。
4. 17:20：安装 `tensorflow-probability==0.25.0` 和 `distrax==0.1.5`。
5. 17:57：从本地 wheel 安装 `agentlace==0.1.3`。

`agentlace` 的已记录来源：

```text
file:///root/hilserl_orin/wheels/agentlace-0.1.3-py3-none-any.whl
sha256=8a0d4278cf86f3664384f374ab0b28b0166a5726e07174a001f3107096efa356
```

该 wheel 当前仍存在于容器的上述路径。当前宿主机项目目录没有发现 wheel 文件，所以后续构建前必须先确认其合法来源并将构建输入显式保存到构建上下文；不能依赖旧容器内部文件。

JAX、Flax 等包没有 `direct_url.json`，只能确认其版本和当前安装状态，不能从包元数据严格证明当时使用了哪个索引 URL或本地 wheel。

## 4. JAX GPU 验证结果与约束

项目中已有 2026-07-31 的验证日志，记录了以下事实：

- `jax==0.4.35`
- `jaxlib==0.4.35`
- `jax-cuda12-plugin==0.4.35`
- `jax-cuda12-pjrt==0.4.35`
- JAX 默认后端为 GPU。
- 发现 `CudaDevice(id=0)`。
- JIT 和 1024×1024 矩阵乘法通过。
- 运行时加载 `libcudnn.so.9.3.0`，cuDNN 数值版本为 `90300`。

卷积测试存在重要约束：

- 默认 XLA autotune 条件下，卷积测试曾失败。
- 错误为 `cudaGetFuncBySymbol: no kernel image is available for execution on the device`，发生于 `RepeatBufferKernel`/卷积 autotune 路径。
- 设置 `XLA_FLAGS=--xla_gpu_autotune_level=0` 后，GPU 卷积、JIT、自动微分、参数更新、有限值检查及 GPU 驻留检查全部通过。

结论：

- 当前 JAX 版本组合能在该 Orin 上执行 GPU 计算。
- “能够识别 GPU”不等于默认配置下所有卷积路径都稳定。
- `--xla_gpu_autotune_level=0` 是当前已验证的兼容性规避措施，不应在没有回归测试的情况下删除。
- 后续可先将该变量作为 `hil-serl` 环境或容器级兼容配置，并记录性能影响；若升级 JAX/CUDA 后问题消失，再评估取消。

## 5. 当前依赖一致性问题

只读执行 `pip check` 未通过，报告 `agentlace==0.1.3` 的以下声明依赖缺失：

```text
gym
opencv-python
typing
zmq
```

当前环境虽然安装了 `gymnasium`、`typing_extensions` 和 `pyzmq`，但 pip 按“发行包名称”判断时，它们不分别等同于 `gym`、`typing` 和 `zmq`。

后续不能机械地把这 4 个名称全部加入 requirements：

- `pyzmq` 实际提供 Python 模块 `zmq`；额外安装名为 `zmq` 的发行包通常不是正确修复方式。
- Python 3.10 已内置 `typing`；`typing` backport 一般不应为现代 Python 强行安装。
- `gymnasium` 与旧 `gym` API 有兼容关系但不是相同发行包，需要结合 HIL-SERL/Agentlace 的实际 import 和 API 使用确认。
- 容器可能具备系统级 OpenCV，但 Conda Python 是否能导入、版本是否匹配仍需单独验证；pip 元数据也不会把系统 apt 包视为满足 `opencv-python`。

建议下一阶段先检查 `agentlace` 代码的真实 import 路径和 HIL-SERL 调用面，再决定：

1. 修正本地 `agentlace` wheel 的过时/不准确依赖元数据；
2. 使用经过验证的 Agentlace 上游版本或固定提交；
3. 添加确实缺失的运行依赖；
4. 对仅属元数据别名的问题保留说明，而不是安装错误的占位发行包。

在新镜像验收时，`pip check` 应作为门禁；如果保留已知例外，必须逐项记录原因和运行测试证据。

## 6. 下一阶段建议实施方案（本次未执行）

### 阶段 A：冻结可复现输入

- 复制并版本化 `assembly` 的 Dockerfile 与 Compose 配置，不直接修改 `/home/naviai/ros_docker_test` 中的在用文件。
- 明确新镜像名称和标签，例如使用带日期或版本的不可变标签，避免只使用 `latest`。
- 获取并校验 `agentlace-0.1.3` wheel；保存 SHA-256。
- 将直接依赖与传递依赖分离：
  - 直接依赖文件只写 HIL-SERL 明确需要的包和固定版本。
  - 另行生成完整 lock/constraints 文件用于精确重建。
- 记录 Python、ARM64、CUDA 12.2、cuDNN 9.3.0 和 JetPack/L4T 的兼容矩阵。

### 阶段 B：设计新 Dockerfile

以 `assembly` Dockerfile 为基础增加独立构建层，计划顺序如下：

1. 保留当前 ROS、CUDA、Conda 和硬件运行基础。
2. 创建 Conda 环境 `hil-serl`，固定 `python=3.10.20` 和必要的基础工具版本。
3. 先安装固定版本的 JAX GPU 四件套及数值依赖。
4. 再安装 Flax、Optax、Chex、Distrax、TensorFlow Probability、Gymnasium 等直接依赖。
5. 最后安装经校验的 Agentlace wheel。
6. 执行 `pip check`。
7. 执行不依赖硬件的 import/CPU 构建时测试。
8. 将 GPU/HIL 硬件测试放到容器运行阶段；Docker build 阶段通常不能假定 GPU 和真实设备可用。

建议固定当前已验证核心版本，而不是使用无上限的 `--upgrade`：

```text
python=3.10.20
numpy==1.26.4
scipy==1.15.3
jax==0.4.35
jaxlib==0.4.35
jax-cuda12-plugin==0.4.35
jax-cuda12-pjrt==0.4.35
flax==0.10.2
optax==0.2.4
chex==0.1.88
distrax==0.1.5
tensorflow-probability==0.25.0
gymnasium==1.2.2
agentlace==0.1.3
```

此清单是当前状态基线，不代表最终依赖决策；尤其要先解决 Agentlace 元数据和应用实际需求。

### 阶段 C：设计新 Compose 服务

- 服务名/容器名使用 `hilserl`。
- 新服务引用新镜像，不复用 `assembly` 容器的可写层。
- 根据功能测试决定是否沿用：
  - NVIDIA runtime；
  - host 网络；
  - X11；
  - `/dev` 与 USB；
  - cuDNN 9.3.0 只读挂载；
  - ROS 环境变量和工作空间挂载。
- 避免让 `hilserl` 与 `assembly` 同时争用独占硬件、ROS 节点名或固定端口。
- 将 64 MiB `/dev/shm` 纳入压力测试；强化学习、图像管线或多进程场景可能需要更大共享内存。

### 阶段 D：分层验收

建议按以下顺序验收，失败时更容易定位：

1. 镜像/架构：确认 ARM64、Ubuntu、CUDA 和 Conda 路径。
2. Python 依赖：核心包 import、版本断言、`pip check`。
3. JAX J1：GPU 发现、JIT、矩阵乘法。
4. JAX J2：带 `XLA_FLAGS=--xla_gpu_autotune_level=0` 的卷积、梯度和参数更新。
5. Agentlace：服务端/客户端回环测试及序列化测试。
6. Gym 环境：reset/step API 与 HIL-SERL 调用兼容性。
7. ROS：Master 连接、消息收发和 Conda Python/ROS Python 互操作。
8. 相机、USB、执行器等 HIL 设备逐项接入。
9. HIL-SERL 最小训练/推理闭环。
10. 长时间稳定性、GPU 内存、温度、功耗和容器重启恢复测试。

## 7. 后续实施前的待确认项

- HIL-SERL 源码的具体版本、分支或 commit。
- HIL-SERL 官方 requirements/安装说明及其 JAX 版本范围。
- `agentlace` wheel 的源码仓库、构建方式和许可证。
- 新镜像是直接继承 `ros1_docker:latest`，还是用现有 Dockerfile从基础镜像完整重建。为了可追溯性，优先建议完整重建并固定基础镜像 digest。
- 是否必须同时保留 `assembly` 的全部视觉模型环境；若只用于 HIL-SERL，可考虑后续拆分以降低约 41.75 GB 镜像带来的构建与分发成本。
- cuDNN 9.3.0 应继续依赖宿主机文件挂载，还是在合法且兼容的前提下固化进新镜像。
- 是否接受将 `XLA_FLAGS=--xla_gpu_autotune_level=0` 设为默认，及其性能影响。
- ROS 网络地址是否应继续写死为当前地址，还是通过 `.env`/部署参数注入。
- 新容器是否确实需要完整 `privileged: true` 和整个 `/dev` 映射。

## 8. 推荐的下一步

下一次实际更改前，先完成以下最小决策：

1. 确定 HIL-SERL 和 Agentlace 的源码版本。
2. 确定“继承现有镜像”或“从现有 Dockerfile完整重建”的策略。
3. 根据源码真实 import 解决 Agentlace 的 4 项 `pip check` 问题。
4. 形成固定版本的直接依赖文件、constraints/lock 文件和 wheel 校验清单。
5. 再开始编写新 Dockerfile 与 Compose 服务，创建 `hilserl` 容器和 `hil-serl` Conda 环境。

在以上输入未固定前，不建议直接把当前 `pip freeze` 全量复制到 Dockerfile，因为这样会把传递依赖、偶然版本和当前不一致状态一起固化。
