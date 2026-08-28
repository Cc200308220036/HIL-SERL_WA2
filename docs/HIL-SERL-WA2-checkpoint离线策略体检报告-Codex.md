# HIL-SERL-WA2 checkpoint 离线策略体检报告（Codex）

> 检查日期：2026-08-26  
> 对象：`20260825_224215_r13` 的 `checkpoint_2000/5000/7856/10000/11509`  
> 检查性质：只读恢复与 GPU 离线推理；未连接 ROS、Agentlace 或机器人，未执行训练更新，未覆盖 checkpoint、buffer 或 metrics。

## 1. 最终结论

本轮离线检查确认：**真机原地抖动的直接来源是 Actor 在 50 Hz 下使用 `argmax=False`，从一个始终很宽的 Gaussian policy 中逐步独立采样。** 同一批连续观测上，策略的确定性 `mode` 很平滑，但随机 `sample` 每一步都有约 0.69 的平均动作变化和约 50% 的方向翻转。

不过，当前不能仅把 Actor 改成 `argmax=True` 就认定策略已经可用。五个 checkpoint 的 critic 都没有形成足够的动作方向区分能力，而且越往后平均区分度越弱：对每个轴施加 `±0.5` 动作扰动时，Q 范围从 checkpoint 2000 的 `0.00304` 降到 checkpoint 11509 的 `0.00112`。在成功 demo 上，checkpoint 11509 的 Q 在轨迹前 50% 仍约为 0，直到最后 10% 才从约 `0.086` 突升到终点 `0.935`。

因此当前状态可以概括为：

```text
策略均值 mode：相邻状态下平滑，但没有证据证明形成了可靠目标方向
策略随机 sample：方差很大，50 Hz 独立采样直接造成抖动
连续 critic：能识别成功终点，但几乎不能区分中前段动作优劣
夹爪 critic：checkpoint 10000/11509 已在抽检状态上全部选择 hold
```

**五个 checkpoint 均不建议直接作为自主真机策略。checkpoint 11509 不是“最接近可用”的稳定策略；它只是终点 Q 更高，同时仍保持很宽的随机分布、很弱的动作区分度和已经塌缩的夹爪选择。**

## 2. 检查方法

### 2.1 固定输入

为保证 checkpoint 之间可比，全部模型使用相同数据：

- 从当前 online cache 均匀选择 32 个固定有效 observation；
- 从 online cache 最后一个连续片段选择 96 个 observation，用于相邻动作平滑性检查；
- 从 20 条初始成功 demo 中，每条按进度提取 11 个 observation，共 220 个点；
- observation 按 `MemoryEfficientReplayBuffer` 的图像堆叠语义重建：有效索引 `i` 的当前图像来自缓存 `i-1`，state 来自 `i`。

### 2.2 每个 checkpoint 的检查项

1. `dist.mode()`：确定性连续 6D 动作。
2. 同一 observation 上采样 32 次：估计策略自身随机标准差。
3. 连续 96 个 observation：比较 mode 和单次 sample 的相邻变化、符号翻转率。
4. 连续 critic：比较 mode、buffer executed action 和 zero action 的 Q。
5. 动作敏感度：分别对 6 个轴在 mode 基础上施加 `-0.5/+0.5`，统计 12 个候选动作的 Q range。
6. demo 价值传播：比较 20 条成功轨迹在 0%～100% 进度处的 executed-action Q。
7. 夹爪 critic：统计 `{-1,0,+1}` 三个动作的 Q 和贪心选择。

推理使用本机 `cuda:0`，网络结构由 `fake_env=True` 重建。脚本没有调用 `agent.update()`。

## 3. checkpoint 总表

表中各轴指标均为 6D 平均值。

| Step | α | mode 绝对幅值 | 同状态 sample 标准差 | 相邻 mode 变化 | 相邻 sample 变化 | mode 翻转率 | sample 翻转率 | Q 扰动区分范围 | 夹爪 hold |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2,000 | 0.050000 | 0.0372 | 0.5848 | 0.00056 | 0.7053 | 0.0% | 49.5% | 0.003041 | 6/32 |
| 5,000 | 0.017123 | 0.0317 | 0.5775 | 0.00123 | 0.6965 | 1.4% | 50.2% | 0.003030 | 3/32 |
| 7,856 | 0.006425 | 0.0690 | 0.5676 | 0.00288 | 0.6917 | 0.4% | 50.5% | 0.001990 | 8/32 |
| 10,000 | 0.003078 | 0.1184 | 0.5645 | 0.00353 | 0.6926 | 1.9% | 52.1% | 0.001595 | 32/32 |
| 11,509 | 0.001837 | 0.1122 | 0.5537 | 0.00528 | 0.6927 | 7.4% | 49.6% | 0.001122 | 32/32 |

## 4. 关键发现

### 4.1 确定性 mode 平滑，随机 sample 剧烈抖动

checkpoint 11509 在连续 96 个 observation 上：

```text
mode 每轴相邻绝对变化均值：   0.0031～0.0074
sample 每轴相邻绝对变化均值： 0.5968～0.8073

mode 平均方向翻转率：          7.4%
sample 平均方向翻转率：       49.6%
```

其他 checkpoint 的 sample 结果基本相同，平均翻转率一直约为 50%。这不是机器人控制器造成的轻微噪声，而是近似独立随机样本的典型表现。

Actor 当前在 `_sample_action()` 中固定使用：

```python
argmax=False
```

并且 live 默认控制频率为 50 Hz。因此每 20 ms 都可能重新选一个方向相反的大幅动作，直接对应现场“原地抖动”。

### 4.2 α 从 0.05 降到 0.00184，但策略分布只缩小约 5%

从 checkpoint 2000 到 11509：

```text
α：                    0.0500 → 0.00184，下降约 96%
sample 平均标准差：    0.5848 → 0.5537，只下降约 5.3%
sample 相邻变化：      0.7053 → 0.6927，几乎不变
```

原因是 α 是训练损失中的熵权重，并不是推理时直接乘在 Gaussian 标准差上的缩放因子。当前 Q 对动作差异太不敏感，Actor 没有足够价值梯度把分布收窄。因此：

- 单独继续降低 α 不能解决抖动；
- 把 α 强行顶回 0.05 也不能建立目标方向，反而可能继续鼓励宽分布；
- 必须先解决 critic 的时间信用分配和动作区分能力。

### 4.3 critic 几乎不能区分 mode、zero 和 buffer action

checkpoint 11509 的固定 observation 结果：

```text
Q(mode) 中位数：          -0.0107713
Q(zero) 中位数：          -0.0107864
Q(mode)-Q(executed)中位数： 0.0001472
单轴±0.5的Q range均值：     0.0011225
```

mode、zero 和实际 buffer action 的价值几乎相同。即使策略均值在 observation 之间发生变化，也没有证据表明这些变化是由可靠的“哪个方向更接近目标”价值差驱动。

动作扰动区分度还随训练下降：

```text
step 2000：  0.003041
step 5000：  0.003030
step 7856：  0.001990
step 10000： 0.001595
step 11509： 0.001122
```

从 step 2000 到 11509 下降约 63%。这说明继续沿当前数据和时间尺度训练，并没有让 critic 更明确地区分动作方向。

### 4.4 成功价值只集中在轨迹末端

20 条成功 demo 的 executed-action Q 均值：

| 轨迹进度 | ckpt 2000 | ckpt 5000 | ckpt 7856 | ckpt 10000 | ckpt 11509 |
|---:|---:|---:|---:|---:|---:|
| 0% | 0.0186 | -0.0068 | -0.0204 | 0.0030 | -0.0119 |
| 50% | 0.1953 | 0.0747 | 0.0153 | 0.0131 | -0.0035 |
| 90% | 0.1098 | 0.0715 | 0.0449 | 0.0701 | 0.0857 |
| 100% | 0.3015 | 0.5837 | 0.7613 | 0.9486 | 0.9346 |

checkpoint 11509 确实能识别成功终点，但其价值形状是：

```text
0%～60%：约 -0.013～0.001
80%：    约 0.023
90%：    约 0.086
100%：   约 0.935
```

也就是说，动态规划只把奖励向前传播了很短距离。开局、接近、抓取和大部分运输阶段没有形成稳定的价值坡度，自然无法给策略提供持续朝目标逼近的方向。

这与当前 50 Hz、平均 762.5-step demo、`discount=0.97` 的时间尺度问题完全一致。

### 4.5 夹爪策略在后期塌缩为 hold

32 个固定 online observation 上的贪心夹爪选择：

| Step | release `-1` | hold `0` | grasp `+1` |
|---:|---:|---:|---:|
| 2,000 | 15 | 6 | 11 |
| 5,000 | 20 | 3 | 9 |
| 7,856 | 16 | 8 | 8 |
| 10,000 | 0 | 32 | 0 |
| 11,509 | 0 | 32 | 0 |

checkpoint 11509 的三类平均 Q：

```text
release：-0.00061
hold：    0.06477
grasp：   0.00307
```

因此训练终端中的 `grasp=g` 只表示 grasp critic 参与了梯度更新，不表示夹爪策略已经学会开合。训练期 `grasp_eps=0.15` 只能偶尔强制探索开/合，贪心策略本身已经选择 hold。

## 5. checkpoint 逐项判定

| Checkpoint | 判定 | 原因 |
|---|---|---|
| 2000 | 不可用于真机自主 | mode 接近零，策略 sample 最宽；critic ensemble 分歧较高，仍接近初始化/早期阶段 |
| 5000 | 不可用于真机自主 | Q 能识别终点，但中前段价值明显下降；sample 仍约 0.58 标准差 |
| 7856 | 不可用于真机自主 | 训练谱系处在 dirty buffer 归档边界；动作 Q 区分度继续下降 |
| 10000 | 不可用于真机自主 | 终点 Q 很高，但前半段仍近零；夹爪已全部选择 hold |
| 11509 | 不可用于真机自主 | sample 仍高噪声、Q 动作区分度最低、夹爪 hold 塌缩；仅终点识别较强 |

没有发现一个可以通过“回退到更早 checkpoint”直接解决问题的版本。

## 6. 对下一步的直接指导

### 6.1 可以进行的下一项离线检查

在任何真机运动前，建议基于本脚本增加两个只读输出：

1. 在固定 demo/online 图像上导出 mode 6D 时间序列和 policy empirical std 曲线；
2. 对每个轴绘制 Q(action) 扫描曲线，而不只比较 `±0.5` 两点。

这可以进一步确认 mode 的方向是否与人工成功动作一致。但本轮结果已经足以判定当前 checkpoint 不能自主上线。

### 6.2 训练结构修复方向

1. ServoL 保持 50 Hz，策略推理和 transition 调整为 10 Hz；一个高层动作由底层连续执行 5 tick。
2. 将 50 Hz demo 按相同执行语义重建成 10 Hz transition，夹爪边沿和区间 reward/done 必须保留。
3. 7D hybrid 任务优先对齐论文的 `discount=0.98`，episode 先控制在约 200～250 个 10 Hz step。
4. 使用全新 run、全新参数和干净 buffer，不从 checkpoint 11509 resume。
5. 增加 deterministic shadow/eval 路径；正式评估使用 mode 与 `grasp_eps=0`。
6. 重新检查夹爪 reward/penalty 和成功轨迹中的开合边沿，避免 hold 再次成为唯一贪心选择。

## 7. 产物与复现

新增只读诊断脚本：

```text
HILSERL_Learner/src/hilserl_wa2/scripts/offline_checkpoint_audit.py
SHA-256: 1929620fd6d15a0c2014ad6db62ac0c09943b884b7d2886cf23b1bf221e964fa
```

原始结果：

```text
HILSERL_Learner/runs/wa2_bottle_pick/20260825_224215_r13/offline_policy_audit/audit.json
SHA-256: ce4b9d0d0dd31bcddc9726cb45803ec66224f3f67e05537c3a4cacd619056aab
```

并为每个 checkpoint 单独保存：

```text
checkpoint_2000.json
checkpoint_5000.json
checkpoint_7856.json
checkpoint_10000.json
checkpoint_11509.json
```

本次没有修改或删除任何训练代码、checkpoint、buffer、metrics、manifest 或机器人端文件。新增内容仅为离线诊断脚本、诊断 JSON 和本报告。
