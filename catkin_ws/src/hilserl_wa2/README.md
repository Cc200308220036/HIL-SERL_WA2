# HIL-SERL WA2 适配层

后续代码按职责组织：

```text
envs/           Gymnasium WA2Env
ros_adapters/   现有ROS topic/service/action封装
interventions/  SpaceMouse动作映射和人工干预
experiments/    WA2任务配置与Actor入口
```

这里存放自研代码；尽量不直接修改相邻的 `hil-serl-main` 上游快照。
