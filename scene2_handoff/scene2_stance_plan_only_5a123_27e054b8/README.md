# Scene2 只读站位判断候选包

这是基于精确闭爪基线 `5A123F3FD1C64F795F3773C3F4BFF5D1CD7A6ABC295C5FB53A05369EC0A39338` 制作的预演版本。

## 本次只验证什么

- 继续使用原有 RGB/深度红色零件识别和“选择最近有效红件”逻辑。
- 根据红件中心在 `base_link` 中的实时 `y` 坐标，输出站位建议：左移、右移或保持。
- 默认舒适中心为 `y=0.150 m`，死区为 `±0.020 m`。
- 单次建议最多 `0.060 m`，较远目标需要移动后重新识别再判断。

此版本不会发布 `/cmd_vel`，也不会发布头部、手臂或夹爪命令。它不是完整抓取提交版。

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
