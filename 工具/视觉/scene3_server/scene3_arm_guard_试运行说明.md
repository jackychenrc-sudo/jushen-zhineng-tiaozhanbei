# Scene3 手臂 IK 安全试运行

本版本不修改已经验证成功的视觉闭环移动。机器人移动并完成二次识别后，手臂阶段默认只计算，不接触料盘、不闭合夹爪。

## 文件

将以下文件放在同一目录：

- `challenge_task_3.py`
- `scene3_vision_debug.py`
- `scene3_arm_guard.py`
- YOLO 模型文件

## 第一次：只做 IK/FK 分析

```bash
export SCENE3_ARM_STAGE=analysis
unset SCENE3_ARM_CONFIRMATION
export SCENE3_ARM_REPORT=/tmp/scene3_arm_guard.json

python3 challenge_task_3.py \
  --scene scene3 \
  --seed "$SEED" \
  --scene3-model-path "$MODEL"
```

正在做什么：复用原移动流程，到达后连续读取 5 个新的 `base_link` 目标，取中值；对预抓取、接触和抓取三个腕位分别做中心及 XYZ 正负 2 cm 的 IK，并用 FK 回算。

成功标准：

- `status` 为 `single_pregrasp_ready`；
- `stable_target.sample_count` 为 5；
- `stable_target.maximum_3d_spread_m` 不超过 `0.010`；
- `ik_fk_robustness.passed` 为 `true`；
- 21 项 IK/FK 检查全部通过；
- `arm_command_sent` 和 `claw_command_sent` 都为 `false`。

上述命令标志只描述“移动完成后的手臂准入门”；原任务在移动前已有的准备姿态和张爪动作不计入这两个标志。

只要任一项失败，就停在当前姿态，不进入预抓取。

## 第二次：仅执行一次预抓取

只有第一次报告通过并人工检查目标、姿态和关节解后才能执行：

```bash
export SCENE3_ARM_STAGE=pregrasp
export SCENE3_ARM_CONFIRMATION=SCENE3_SINGLE_PREGRASP
export SCENE3_ARM_REPORT=/tmp/scene3_arm_guard_pregrasp.json

python3 challenge_task_3.py \
  --scene scene3 \
  --seed "$SEED" \
  --scene3-model-path "$MODEL"
```

正在做什么：重新完成同一套 5 帧和 21 项 IK/FK 检查，然后只发布一次已经通过检查的预抓取关节轨迹。

成功标准：

- `status` 为 `single_pregrasp_completed`；
- `post_execution_fk.passed` 为 `true`；
- 实际手腕位置误差不超过 `0.02 m`；
- 实际手腕姿态误差不超过 `12°`；
- `claw_command_sent` 仍为 `false`。

到这里必须停止，观察手腕相对料盘的前后、左右和高度误差。当前版本没有开放接触、闭爪、抬升或出库。

## 离线回归

```bash
python3 -m unittest test_scene3_arm_guard -v
```
