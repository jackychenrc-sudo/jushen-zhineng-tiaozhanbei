# Scene2 高位路线锁腕测试候选（未完成 Docker 实机验证）

这个目录只用于服务器测试，不是最终提交版本，也没有覆盖可靠闭爪基线。

## 本次只改什么

- `side_clear`、`high_lift_*`、`high_forward_*`、`over_part_*` 使用肩肘四关节 IK，并保持腕部三个关节为实时测量值。
- `approach_*`、`pinch_align_*`、`grasp_lift_*`、搬运和放箱阶段继续启用腕部 IK。
- 保留上一版对 `solver_success=False` 但预测误差不超过既有 5 mm 门槛时继续进入原有关节细分与反馈检查的修复。
- 未修改抓取点、8° 单段关节跳变门槛、20° 姿态保护、实际跟踪容差、夹爪参数和安全高度。

目标是修复最新日志中 `side_clear` 在 joint 6 附近反复跳 IK 分支并最终耗尽细分的问题。

## 校验值

- ZIP SHA256：`06F438FC060EB2B3F51944D7C88BEDC7A39306263C81C92F39AFD36A6A0DFAC4`
- 主脚本 SHA256：`618CD2614EABBC01434843958827E2A40216707CC32AB7331FB34EBF21914489`
- 上一候选主脚本 SHA256：`1AA721300082692F53C130DCDA5AEDDEDFB412C456960D0EAC08D71BD8108432`
- 可靠闭爪基线 SHA256：`5A123F3FD1C64F795F3773C3F4BFF5D1CD7A6ABC295C5FB53A05369EC0A39338`

## 已完成的离线检查

- 7 个 Python 文件 AST 语法检查通过。
- 所有递归 `_mid`、`_end`、`_feedback` 标签会继承原阶段的锁腕策略。
- 模拟求解测试确认锁腕模式下 joint 5/6/7 与输入测量值完全相同。
- ZIP 完整性、Linux 正斜杠路径和包内主脚本 SHA 均已验证。

## 服务器运行前提

必须先启动官方 Scene2 环境，并在容器内执行：

```bash
cd /root/kuavo_ws
source /opt/ros/noetic/setup.zsh
source devel/setup.zsh
```

本候选首先只验证能否通过 `side_clear` 和 `high_lift`；若日志仍在高位路线报腕部 joint 6/7 跳变，应立即停止并保存完整日志，不修改安全阈值。
