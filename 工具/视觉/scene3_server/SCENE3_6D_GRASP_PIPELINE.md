# Scene3 三阶段 6D 抓取流程

这个入口只负责机械臂和最终夹爪动作，不控制底盘。底盘仍使用已经验证过的学长闭环移动，先到料盘前约 0.65 m，并完成近距离世界目标锁定。

流程固定为：

1. **预处理位置**：只强约束 TCP 位置，夹爪方向不锁死；
2. **原地转正**：固定物理 TCP，用 6D IK 将夹爪前向对准料盘、上方向对准 `+Z`；
3. **直线伸入**：保持转正后的完整方向，沿固定直线到最终 TCP；
4. **腕部确认**：右腕 RGB-D 使用同一个近侧抓取边缘，连续三帧通过后才允许闭爪；不再错误地用料盘中心代替抓取边缘；
5. **闭爪**：必须另有 `--close-claw` 明确授权。

## 关键控制规则

- 绝不把 `/sensors_data_raw` 的绝对关节角直接发布成命令；
- 每一段都读取当前 `/joint_cmd[13:27]` 作为绝对命令起点；
- 官方 IK 只提供“求解角减去实测角”的相对增量；
- 新命令 = 当前底层命令 + IK 相对增量；
- 每段使用五次曲线，运动后检查物理 TCP、方向、左臂和料盘身份；
- 任一门禁失败只回退当前小段，夹爪保持打开。

## 建议运行顺序

先做只读审计，不发送任何控制命令：

```bash
python3 -u scene3_6d_grasp_pipeline.py --stage audit
```

只计算下一段预处理计划，不发送命令：

```bash
python3 -u scene3_6d_grasp_pipeline.py --stage preprocess
```

人在仿真窗口前确认后，执行到腕部确认，但不闭爪：

```bash
timeout 360 python3 -u scene3_6d_grasp_pipeline.py \
  --stage all \
  --execute \
  --confirmation SCENE3_6D_PIPELINE \
  | tee /tmp/scene3_6d_pipeline.log
```

确认腕部视觉输出 `SIX_D_WRIST_VERIFY_OK` 后，可在验证有效期内闭爪：

```bash
python3 -u scene3_6d_grasp_pipeline.py \
  --stage close \
  --execute \
  --confirmation SCENE3_6D_PIPELINE \
  --close-claw
```

需要一次完成时，可以在 `--stage all` 命令末尾加入 `--close-claw`。首次实测不建议这样做。

旧入口 `scene3_safe_turn_position.py` 已经停用并强制拒绝执行，避免再次触发错误的绝对关节角接管。
