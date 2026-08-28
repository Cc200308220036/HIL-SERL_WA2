# left_wrist 状态（R1）

- topic: `/zj_humanoid/sensor/left_wrist/image_raw`
- 2026-08-11 实测：topic 名存在于 graph，但 **Publishers: None**，无消息（未启动）
- 契约：`enabled: false`，`missing_policy: zero_image`
- 启动后只需改 YAML `enabled: true` 并复测 hz（升 version）
