# HIL-SERL WA2 Actor 镜像部署

## 1. 目标和边界

部署产物分为两部分：

```text
hilserl:actor-v1 镜像
  └── ROS/CUDA/Conda、hil-actor、Python 依赖、Agentlace、SERL Launcher

Git 仓库/catkin_ws
  └── 运行时挂载到容器 /root/catkin_ws，用于继续开发
```

镜像不是容器。新 Orin 加载镜像后，由 Compose 创建名为 `hilserl` 的容器。
`docker save` 包含派生镜像及其父层，但不包含 bind mount 中的 `catkin_ws`、模型、训练数据和日志。

Compose 使用 `../catkin_ws` 相对路径，因此 Git 仓库可以位于任意用户目录，也可以命名为 `orin_hilserl` 或 `HILSERL-WA2`，无需修改挂载路径。

## 2. 旧 Orin 构建镜像

保留当前已验收的旧容器作为回退，不要先删除它。进入仓库根目录：

```bash
cd /home/naviai/hilserl_orin

export HILSERL_SOURCE_REVISION="$(git rev-parse HEAD)"

docker build \
  --network host \
  --build-arg BASE_IMAGE=ros1_docker:latest \
  --build-arg IMAGE_VERSION=actor-v1 \
  --build-arg SOURCE_REVISION="$HILSERL_SOURCE_REVISION" \
  -f docker/dockerfile \
  -t hilserl:actor-v1 \
  .
```

Dockerfile 会创建 `/opt/conda/envs/hil-actor` 并执行无硬件的依赖与导入检查。不要同时启动第二个能够控制机器人的容器。

检查镜像元数据：

```bash
docker image inspect hilserl:actor-v1 \
  --format 'id={{.Id}} arch={{.Architecture}} version={{index .Config.Labels "org.opencontainers.image.version"}} revision={{index .Config.Labels "org.opencontainers.image.revision"}}'
```

## 3. 导出与传输

建议在仓库之外保存大镜像文件：

```bash
mkdir -p /home/naviai/hilserl_images

docker save hilserl:actor-v1 \
  | gzip -1 \
  > /home/naviai/hilserl_images/hilserl-actor-v1-arm64.tar.gz

sha256sum /home/naviai/hilserl_images/hilserl-actor-v1-arm64.tar.gz \
  > /home/naviai/hilserl_images/hilserl-actor-v1-arm64.tar.gz.sha256
```

传输到新 Orin：

```bash
scp /home/naviai/hilserl_images/hilserl-actor-v1-arm64.tar.gz* \
  naviai@NEW_ORIN_IP:/home/naviai/HILSERL-WA2/
```

## 4. 新 Orin 加载镜像

假设代码已通过 Git 放在 `/home/naviai/HILSERL-WA2`：

```bash
cd /home/naviai/HILSERL-WA2

sha256sum -c hilserl-actor-v1-arm64.tar.gz.sha256
gzip -dc hilserl-actor-v1-arm64.tar.gz | docker load

docker image inspect hilserl:actor-v1 \
  --format 'id={{.Id}} arch={{.Architecture}}'
```

新 Orin 不需要重新安装 `hil-actor` 或执行 `pip install`。

## 5. 创建 hilserl 容器

同型号 WA2、相同 ROS 网络基线可以直接使用 Compose 默认值：

```bash
cd /home/naviai/HILSERL-WA2

docker compose -f docker/docker-compose.hilserl.yml config
docker compose -f docker/docker-compose.hilserl.yml \
  up -d --no-build hilserl
```

Compose 会自动完成：

```text
当前仓库/catkin_ws -> /root/catkin_ws
镜像 hilserl:actor-v1 -> 容器 hilserl
```

如果新机器 ROS IP 不同，才需要复制模板并修改：

```bash
cp docker/.env.example .env
docker compose --env-file .env -f docker/docker-compose.hilserl.yml \
  up -d --no-build hilserl
```

## 6. 进入开发环境

```bash
docker exec -it hilserl bash
```

镜像的 entrypoint 和交互式 shell 都会加载 `hil-actor`。如需手动激活：

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate hil-actor
```

确认工作目录和环境：

```bash
cd /root/catkin_ws
which python
python --version
echo "$PYTHONPATH"
```

## 7. 新 Orin 首次构建 catkin 工作空间

Git 仓库不应携带旧机器生成的 `build` 和 `devel`。进入容器后重新构建：

```bash
source /opt/ros/noetic/setup.bash
source /ros_noetic/catkin_ws/devel/setup.bash

cd /root/catkin_ws
catkin_make
source /root/catkin_ws/devel/setup.bash
```

## 8. 可选的机器级配置

只有以下条件发生变化时才需要额外配置：

- ROS Master 或新 Orin IP 变化：修改 `.env`；
- Learner/Actor IP 变化：同步修改两端 `configs/network/local.yaml`；
- 新 WA2 机械零位或相机安装变化：重新标定 scene 和相机配置；
- 开始 R13 训练：另行放置 `/root/.serl/resnet10_params.pkl`、Classifier checkpoint、demo 和训练 checkpoint。

这些内容属于机器人实例和实验资产，不属于通用 Actor 依赖镜像。

## 9. 安全限制

- 不要让 `assembly`、旧 `hilserl` 和新 `hilserl` 同时发布机器人动作；
- 新容器先做离线、ROS 只读和相机验收，再进行 ServoL 与 reset；
- 不配置 `restart: always`，避免机器人控制容器未经人工确认自动恢复；
- 当前 cuDNN 挂载要求新 Orin 具备与现有 R36.4.4 基线一致的宿主机库文件。
