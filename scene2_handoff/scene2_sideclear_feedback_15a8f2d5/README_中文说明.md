# Scene2：side_clear 真实反馈微步候选

## 目的

定位结果是：IK 静态解正确，但原 side_clear 一次约 11.5° 的多关节动作在新服务器上严重超调。此候选只修抓取前 side_clear 和安全返程，不改视觉、抓取点、20°姿态门槛、夹爪、下降、闭爪及闭爪后反向回放。

## 修改

- side_clear 每个活动臂关节命令最大 1.20°；
- 每步读取 /sensors_data_raw 真实关节并重新计算 FK；
- 每步关节跟踪误差不得超过 0.50°，末端误差不得超过 8 mm；
- 低进度重试和反馈重规划同样受 1.20°门槛约束；
- 返程按已实际经过的关节端点倒放，每步同样不超过 1.20°；
- 失败时不执行原先会继续跑偏的大幅开环回退，锁存实测状态后交回官方自动摆臂。

## 校验

- 基线脚本 SHA256：`0664B0B400C2A6F79C1D23B1857CC8B045A8535EFFBE4B41C16BCBB88FC308D6`
- 候选脚本 SHA256：`15A8F2D5B50908A50FB5084CA7F307E9A26B73C063059C343DEAC43D8E607930`
- ZIP SHA256：`502DCF686505905ACA547635DD11110714C3DDAFE400B186E872EA8EB51C3BBB`

## 必须先跑 10% 往返

```zsh
python3 -u "$SCRIPT" \
  --restricted-side-execute \
  --restricted-side-fraction 0.10 \
  --move-time 3.0 \
  2>&1 | tee /tmp/scene2_sideclear_feedback_10pct.log
```

只有出现 `restricted side restored`，且无 `tracking error`、`IK subdivision exhausted` 时，才运行 100% 侧移。