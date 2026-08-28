# HIL-SERL Orin

独立的 HIL-SERL Actor 复现与 WA2 适配仓库。

## 目录

```text
artifacts/          固定 wheel、模型及校验文件
catkin_ws/           与容器 /root/catkin_ws 双向同步
  src/
    hil-serl-main/   独立HIL-SERL上游源码
    hilserl_wa2/     WA2 Env、ROS适配和人工干预代码
    naviai_controller/ 已复制的机器人控制接口
configs/            Actor、Learner和实验配置
docker/             hilserl镜像、Compose、依赖锁和验收脚本
docs/               接口、环境契约、安全与部署文档
scripts/            构建、准备和验收脚本
tests/              单元、集成和硬件测试
```

## 容器目录映射

Compose 使用：

```text
/home/naviai/hilserl_orin/catkin_ws → /root/catkin_ws
```

只有 `catkin_ws` 内的内容会通过该挂载与容器双向同步。`docker`、`docs`、`artifacts` 等仓库管理文件不会覆盖容器工作空间。

`/home/naviai/ros_docker_test` 始终作为只读来源，不由本仓库修改。

## 构建入口

参见 [`docker/README.hilserl.md`](docker/README.hilserl.md)。
