# Scene2 红色螺丝刀抓取测试包（A8F798F3）

## 状态

- 用途：官方 Docker 中验证红色螺丝刀完整抓取、抬升与放入紫色目标框。
- 当前状态：已通过 Python 语法、静态安全和用户失败日志数值复核；尚未在官方 Docker 中实跑成功，不能标记为最终提交版。
- 本轮使用直接测试入口 `scripts/scene2_pick_nearest_red.py`；统一入口 `challenge_task.py --scene scene2` 尚未接入本候选。
- 原始可靠闭爪回滚基线未覆盖，SHA-256：`5A123F3FD1C64F795F3773C3F4BFF5D1CD7A6ABC295C5FB53A05369EC0A39338`。

## 本次失败根因

旧版在闭爪前的 `high_lift_48_feedback` 失败：

- 实际末端：约 `[0.0465, 0.3478, 0.1093]`
- 旧目标：约 `[0.0461, 0.3364, 0.1196]`
- 三维误差：约 `15.3 mm`
- 原门槛：`15.0 mm`

因此该次运行没有进入 PREGRASP、最终对齐或闭爪；视觉检测不是本次失败原因。

## 最小修改

1. 高抬每一步从最新真实 FK 的 XY 继续竖直抬升，不再让长期累计的横向控制偏差占满 15 mm 跟踪预算。
2. 仍严格要求实测高度达到动态最低安全平面以上 5 mm；末段仅命令 3 mm 竖直超调帮助反馈跨过门槛。
3. 新增相对侧移实际点 15 mm 的累计 XY 漂移保护。
4. 修复第 60 步恰好达到安全高度却被 `for-else` 误判失败的边界情况。
5. 闭爪后持续重发夹紧命令，先竖直抬升 5 cm 并验证，再通过安全高位直接搬向运行时 RGB-D 检出的紫色框，下降、张爪并竖直撤离。

以下保护未放宽：

- 实际位置误差门槛：15 mm（高抬）；
- IK 关节分段上限：8°；
- 最终抓取姿态保护：20°；
- 动态安全高度、视觉/FK反馈与禁用接口限制。

## 哈希

- 主脚本 SHA-256：`A8F798F3B44CF3FD9ECCE4160BF0B520B3F87DE38DBF47B91CB4E8A2840BAE83`
- 完整 ZIP SHA-256：`B47B70249859302645AD03114E1E8F2B63AE829FCE06B29E9B10C26C5E17A546`
- ZIP 已使用 POSIX `/` 路径重新打包，可在官方 Linux 容器中直接解压。

## 官方容器测试命令

```bash
cd /root/kuavo_ws
source /opt/ros/noetic/setup.zsh
source devel/setup.zsh

set -o pipefail

python3 -u \
  /root/scene2_direct_highlift_terminal_a8f798f3/challenge_cup_task_template/scripts/scene2_pick_nearest_red.py \
  --restricted-single-loop-execute \
  --grasp-center-clearance 0.000 \
  --bin-release-clearance 0.10 \
  --restricted-high-step 0.05 \
  --restricted-approach-step 0.02 \
  --move-time 3.0 \
  2>&1 | tee /tmp/scene2_direct_highlift_terminal_a8f798f3.log
```

重点阶段：`HIGH_LIFT_COMPLETE`、`PREGRASP`、`ALIGN_COMPLETE`、`CLAW_CLOSED`、`POST_GRASP_LIFT`、`ABOVE_PURPLE_BIN`、`RELEASED`、`RETREATED`。
