# 文档规划

- `ros-interface-inventory.md`：现有机器人 topic/service/action 清单。
- `wa2-env-contract.md`：observation/action、reset/step 和异常语义。
- `safety-checklist.md`：实机运行前安全门禁。
- `数据采集优化控制方案.md`：SpaceMouse 采集卡顿定位、第一刀 hold-last、第二刀 latch + 50 Hz 按时间积分（代码已合入，真机手感待烟测）。
- `solution/SpaceMouse-20Hz试验说明.md`：独立遥操把 Servo 降到 20 Hz 的最小改法（不动 Actor/Learner）。
- 阶段验收：`调试日志/阶段验收日志/2026-08-19_R12验收.md` → **R12 GATE: PASS**。
- 阶段方案：`docs/solution/R12_方案.md`（已完成）；`docs/solution/R13_方案.md`（当前，论文级 HIL-SERL：脱手抓放瓶）。
- 根目录 `hil-serl部署.md`：总体部署计划。
- 根目录 `hil-serl-wa2开发.md`：早期环境分析记录。
