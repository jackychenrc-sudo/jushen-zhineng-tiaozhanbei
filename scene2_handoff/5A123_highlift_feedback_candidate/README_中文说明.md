# Scene2 5A123 高位抬升反馈修复候选

## 来源与状态

- 原始闭爪基线脚本 SHA-256：`5A123F3FD1C64F795F3773C3F4BFF5D1CD7A6ABC295C5FB53A05369EC0A39338`
- 本候选脚本 SHA-256：`A3F5C75B86220D0BFD0F30D79E88BC5401E2626709B25D1B008E4C553F144C94`
- 原始 5A123 基线已单独保留，本目录不会覆盖回滚版本。
- 当前候选已通过 Python 语法编译检查，尚待官方 Docker 实机复跑。

## 为什么修改

两次官方 Docker 运行都在闭爪之前的 `high_lift_13_end_end` 被终止：

- 第一次末端位置误差：15.2 mm；
- 干净重启复跑误差：15.1 mm；
- 原安全门槛：15.0 mm。

原代码仅在带姿态目标的路径超差时进行反馈重规划；`high_lift_*` 使用位置 IK，超差后会直接退出。

## 最小修改

- 仅允许 `high_lift_*` 在超过原 15 mm 门槛时，从真实关节/FK反馈重新规划一次；
- 一次反馈后仍未进入 15 mm 内，仍按原逻辑失败；
- 细分递归传递重试计数，避免重试预算被重置；
- 兼容 high-lift 没有姿态误差值时的日志输出。

以下内容全部保持不变：抓取点、Z偏移、轨迹、速度参数、15 mm位置门槛、20°姿态门槛、8°关节步长保护、IK预测误差保护和夹爪参数。

## 运行

先通过 `challenge_task.py --scene scene2 --seed 3` 启动场景，再运行：

```bash
python3 -u challenge_cup_task_template/scripts/scene2_pick_nearest_red.py \
  --restricted-single-loop-execute \
  --grasp-center-clearance 0.000 \
  --bin-release-clearance 0.10 \
  --restricted-high-step 0.05 \
  --restricted-approach-step 0.02 \
  --move-time 3.0
```

测试时应重点确认日志出现 `feedback replan=1`，随后误差降至 15 mm 内并出现 `HIGH_LIFT_COMPLETE`。
