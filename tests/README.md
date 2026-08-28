# 测试目录

```text
unit/         不依赖ROS和硬件的纯逻辑测试
integration/  Agentlace、ROS接口与Gym Env集成测试
hardware/     需要Orin、相机、SpaceMouse或机器人的测试
```

硬件测试必须显式运行，不纳入默认测试命令。
