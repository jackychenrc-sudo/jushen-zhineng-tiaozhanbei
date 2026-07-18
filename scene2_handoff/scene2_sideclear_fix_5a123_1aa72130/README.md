# Scene2 只读站位判断候选包

这是基于精确闭爪基线 `5A123F3FD1C64F795F3773C3F4BFF5D1CD7A6ABC295C5FB53A05369EC0A39338` 制作的预演和最小修复版本。

## 本次只验证什么

- 继续使用原有 RGB/深度红色零件识别和“选择最近有效红件”逻辑。
- 根据红件中心在 `base_link` 中的实时 `y` 坐标，输出站位建议：左移、右移或保持。
- 默认舒适中心为 `y=0.150 m`，死区为 `±0.020 m`。
- 单次建议最多 `0.060 m`，较远目标需要移动后重新识别再判断。

使用 `--stance-plan-only` 时不会发布 `/cmd_vel`，也不会发布头部、手臂或夹爪命令。使用执行类参数时仍会运行相应的原手臂流程；本包目前只是测试候选，不是最终提交版。

## `side_clear` 最小修复

原代码把优化器内部 `3 mm` 收敛标志放在调用方已经配置的 `5 mm` 预测容差之前，导致 `xyz_err=3.2 mm` 的有效解提前退出，后续 `8°` 关节自动分段没有机会执行。

本候选只修正判断顺序：

- 非收敛状态且预测误差不超过现有 `5 mm` 时，继续使用原有关节分段和反馈保护；
- 单段最大关节变化仍为 `8°`；
- 超过 `5 mm` 的预测结果仍立即拦截；
- 未修改抓取点、姿态门、实际跟踪容差和夹爪参数。

## Docker 内运行

```bash
cd /root/kuavo_ws
source /opt/ros/noetic/setup.zsh
source devel/setup.zsh

python3 -u \
  /替换为实际解压路径/challenge_cup_task_template/scripts/scene2_pick_nearest_red.py \
  --stance-plan-only \
  2>&1 | tee /tmp/scene2_stance_plan_only.log
```

只看包含 `STANCE_PLAN_ONLY` 的日志：

- `MOVE_BASE_RIGHT`：规划认为机器人应沿自身右侧移动。
- `MOVE_BASE_LEFT`：规划认为机器人应沿自身左侧移动。
- `HOLD_POSITION`：目标已在暂定舒适区内。
- `desired_base_dy`：理论总位移，正数为左、负数为右。
- `first_step_dy`：将来真实执行层第一步的限幅建议，目前不会执行。

## 重要限制

当前服务器上的 `/cmd_vel` 输入轴与实际运动方向尚未完成标定。因此这里仅输出 `base_link` 几何方向，不把建议转换为 `Twist.linear.x/y`，避免机器人再次向错误方向移动。
