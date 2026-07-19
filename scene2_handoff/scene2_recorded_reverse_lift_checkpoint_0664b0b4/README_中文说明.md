# Scene2 红色螺丝刀：闭爪后反向回放抬升 3 cm 验证包

## 这版做了什么

- 保留现有视觉、PREGRASP、最终对齐和闭爪流程。
- 闭爪后不再在线求解新的抬升 IK。
- 记录本次最终连续垂直下降的实测关节状态，并在闭爪后反向回放。
- 实测夹持中心抬升达到 3 cm 后保持夹紧，暂不执行投箱。
- 保留 8°关节步长、20°姿态、5°非工作手臂、3 mm 掉落/滑移等保护。
- 最终必须重新识别到真实红件；预测点不能作为成功依据。

这是只验证“是否能稳定夹住并抬起 3 cm”的 checkpoint 候选，不是最终完整搬运提交版。

## 校验值

- 上一候选脚本 SHA256：`618CD2614EABBC01434843958827E2A40216707CC32AB7331FB34EBF21914489`
- 本候选脚本 SHA256：`0664B0B400C2A6F79C1D23B1857CC8B045A8535EFFBE4B41C16BCBB88FC308D6`
- ZIP SHA256：`C60426A6A67ECD7C498AC99BA6CC2FA1B989216F512668B07C9D8D1CC667A387`

## Docker 内运行

```zsh
cd /root/kuavo_ws
source /opt/ros/noetic/setup.zsh
source devel/setup.zsh

SCRIPT=/root/scene2_recorded_reverse_lift_checkpoint_0664b0b4/challenge_cup_task_template/scripts/scene2_pick_nearest_red.py

set -o pipefail
python3 -u "$SCRIPT" \
  --restricted-single-loop-execute \
  --replay-lift-checkpoint-only \
  --grasp-center-clearance 0.000 \
  --bin-release-clearance 0.10 \
  --restricted-high-step 0.05 \
  --restricted-approach-step 0.02 \
  --move-time 3.0 \
  2>&1 | tee /tmp/scene2_recorded_reverse_lift_checkpoint_0664b0b4.log
```

看到 `REPLAY_VERIFY` 后，程序会故意保持夹紧并持续运行，这不是卡死。确认画面后按 `Ctrl+C`；下一次测试前重启官方容器或明确恢复控制模式。

