# 阶段验收日志（R1～R12）

本目录集中存放 WA2Env / HIL-SERL 主线的验收结论、原始输出与证据，与日常调试笔记（上级目录 `0805`～`0814_*.md` 等）分开。R10 日更见 `../0814调试日志.md`（原 `R10调试日志.md` 已并入）。

| 阶段 | 结论文件 | 主要证据 |
|---|---|---|
| R1 | `2026-08-11_R1验收.md` | `R1_ROS接口盘点.md`、`r1_samples/` |
| R2 | `2026-08-11_R2验收.md` | `2026-08-11_R2验收_raw.txt`、`r2_contract_sha256.txt` |
| R3 | `2026-08-11_R3验收.md` | `2026-08-11_R3验收_raw.txt`、`2026-08-11_R3_soak.txt` |
| R4 | `2026-08-11_R4验收.md` | `2026-08-11_R4验收_raw.txt`、`2026-08-11_R4_gates.txt` |
| R5 | `2026-08-12_R5验收.md` | `verify_r5_reset.py --real --n 10` |
| R6 | `2026-08-12_R6验收.md` | `catkin_ws/r6_samples/` |
| R7 | `2026-08-12_R7验收.md` | Intervention offline/live |
| R8 | `2026-08-13_R8验收.md` | fake_env / mapping / agent+buffer |
| R9 | `2026-08-13_R9验收.md` | 本地 Actor/Agentlace；readonly / live-zero / 断连 / SIGINT |
| R10 | `2026-08-14_R10验收.md` | `runs/wa2_bottle_pick/20260814_153513_r10/`（Actor+Learner） |
| R11 | `2026-08-18_R11验收.md` | live20 `demo.pkl` + Learner demo buffer |
| R12 | `2026-08-19_R12验收.md` | `0819_1426_r12_clean` 分类器 ckpt + `R12_LIVE_EVAL: PASS` |

后续验收记录命名保持 `YYYY-MM-DD_Rx验收.md`。

