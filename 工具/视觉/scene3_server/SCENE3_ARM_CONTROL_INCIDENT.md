# Scene3 机械臂控制异常记录与当前安全路线

日期：2026-07-17

## 结论

视觉坐标、底盘接近和 IK/FK 计算均不是这次异常的主要矛盾。异常发生在一次
**尚未合入 GitHub 的本地 `/kuavo_arm_target_poses` 执行实验**中：规划器根据
FK 预测一个 5 mm 物理 TCP 小段，但实际机械臂运动约 51 mm，并带动了本应保持
不动的左臂。

这份记录不能证明官方接口自身存在 Bug。严谨的表述是：当前集成方式未能安全、
可重复地使用该定时轨迹链路，因此比赛代码继续统一使用已经通过分段运动验证的
`/kuavo_arm_traj`。

## 直接证据

实验计划：

```text
Planned TCP step: 5.0mm
Right shoulder/elbow command delta: [1.292, -0.844, -0.029, -3.252]deg
Right wrist command delta: [0.0, 0.0, 0.0]deg
```

实际运动：

```text
Observed TCP delta: [0.04922, -0.01244, 0.00434]m
Progress: 32.3mm
Cross-track error: 39.4mm
left_arm_held: False
```

也就是规划约 5 mm，但实际三维位移约 51 mm。

失败回退后的只读状态：

```text
Left arm joint 3:  85.189deg
Right arm joint 3: 85.176deg
```

执行前对应关节约为 29 deg，说明回退并未恢复原始构型。该姿态必须通过仿真
Reset 恢复，禁止继续叠加机械臂命令。

## 已排除的原因

同一时刻读取 MPC observation 和原始传感器：

```text
MPC arm joint 3:    85.192deg / 85.178deg
Sensor arm joint 3: 85.192deg / 85.178deg
Maximum difference: about 0.001deg
```

因此可以排除“读取了错误的 MPC 手臂片段”“MPC 与传感器使用不同角度单位”
这两个假设。

仍未定位到官方链路内部的单一代码根因。可能范围包括首次轨迹起点、上一目标状态、
控制权接管或绝对目标的使用语义。完成比赛任务不依赖继续冒险验证该链路。

## 夹爪方向问题

此前的位置阶段锁住了三个腕部关节，只用肩肘把手送到料盘附近。这样能够改善
`x/y/z`，但不能同时保证夹爪姿态，实测曾出现约 34--62 deg 的夹爪轴误差，
并触发以下腕部几何门禁：

```text
target_in_corridor: False
target_between_fingers: False
segment_distance: False
tcp_error: False
```

因此腕部相机只能用于转正后的最终确认，不能代替安全预抓取位置和完整 6D 姿态
规划。

## 当前安全路线

1. Reset 恢复异常关节；
2. 头部视觉锁定料盘世界身份；
3. 使用学长闭环底盘代码停在约 0.65 m；
4. 通过 `/kuavo_arm_traj` 到达料盘前 0.16 m、上方 0.02 m 的预抓取点；
5. 在安全点用 6D IK 同时计算肩、肘和手腕，使夹爪方向正确；
6. 继续通过同一 `/kuavo_arm_traj` 通道平滑执行；
7. 保持方向直线伸入；
8. 腕部 RGB-D 连续三帧确认；
9. 明确授权后闭爪、抬起、后退。

比赛路径中不混用 `/kuavo_arm_target_poses`、`/joint_cmd` 直接写入和
`/kuavo_arm_traj`。

## 只读审计

重启或 Reset 后，先运行：

```bash
python3 -u scene3_arm_control_audit.py
```

只有输出 `SCENE3_ARM_CONTROL_AUDIT_OK` 才允许继续预抓取动作。脚本只订阅状态并
查询当前模式，不创建控制发布器、不切换模式、不写参数。
