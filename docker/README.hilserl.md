# hilserl容器

## 当前阶段

当前目标是在现有 `hilserl` 容器及其 `hil-actor` Conda环境中完成：

- WA2 Gymnasium Env。
- ROS机器人接口。
- SpaceMouse干预。
- Orin Actor单机运行。
- Orin Actor与笔记本Learner跨机联调。
- 小规模真机训练闭环。

全流程验收完成前不构建 `hilserl:actor-v1`。

当前Compose继续使用：

```text
镜像：ros1_docker:latest
容器：hilserl
Conda环境：hil-actor
```

## 启动现有容器

```bash
docker start hilserl
docker exec -it hilserl bash
```

进入后：

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate hil-actor
```

## 工作空间同步

```text
宿主机 /home/naviai/hilserl_orin/catkin_ws
  ↕
容器 /root/catkin_ws
```

源码、WA2适配和机器人控制代码都在该目录中开发。

## Compose

如果容器不存在，可使用基础镜像重新创建：

```bash
cd /home/naviai/hilserl_orin

docker compose -f docker/docker-compose.hilserl.yml \
  up -d --no-build hilserl
```

注意：新建容器不会包含当前手工安装的 `hil-actor`。全流程完成前不要删除当前已验收容器。

## Dockerfile状态

`docker/dockerfile` 当前只保留为环境固化草案，Compose不会读取或构建它。

完成真机全流程后再执行：

1. 冻结最终Python、ROS和系统依赖。
2. 删除实验期间不再需要的依赖。
3. 固定HIL-SERL和WA2适配源码版本。
4. 重写并评审最终Dockerfile。
5. 构建版本化镜像。
6. 从新镜像重新创建容器并执行完整回归验收。

## 当前环境验收

```bash
python -m pip check
python /opt/hilserl/verify/verify_hil_actor.py
python /opt/hilserl/verify/verify_agentlace.py
```

## 使用限制

- 不要删除当前 `hilserl` 容器，其Conda环境位于容器可写层。
- 不要同时从 `assembly` 和 `hilserl` 控制同一机器人。
- 不要同时占用相机、SpaceMouse或相同Agentlace端口。
- `/home/naviai/ros_docker_test` 不由本项目修改。
