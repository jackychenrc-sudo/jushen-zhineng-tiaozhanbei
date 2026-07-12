# Scene3 Day2：视觉与动作联调方案

更新时间：2026-07-12 下午

这份文档给闫帅辰和王森今天下午使用。目标不是把历史 `pytrees_actions` 整包接进比赛代码，而是从里面借动作思路，尽快打通 Scene3 上层料盘闭环。

## 今天下午总目标

先只做上层料盘。

```text
保存 RGB/depth -> 标清楚上层料盘 -> 算出 3D 点 -> 动作骨架能开夹爪/动手臂/回撤 -> 人工把视觉点交给动作测试
```

今天能做到下面任意一条都算有效进展：

```text
A. 视觉：上层料盘能稳定输出 bbox、center_pixel、camera_xyz。
B. 动作：机器人能执行预备姿态、夹爪开合、伸手/回撤中的任意两个动作。
C. 联调：动作脚本能读取一份视觉 JSON，哪怕先只打印或手动代入。
```

## pytrees 里值得借的部分

| 能力 | 参考文件 | 借什么 | 不要照搬什么 |
| --- | --- | --- | --- |
| Scene3 总流程 | `py_tree_smt.json` | 任务顺序：选目标、靠近、抓、后退、去放置区、放下 | 不要直接接整套行为树 |
| 抓料盘流程 | `tree/smt_grasp_a_tray.json` | 预备姿态、找目标、开夹爪、抓取、闭夹爪、抽出 | 不要相信旧 row/ID 参数完全匹配比赛场景 |
| 放料盘流程 | `tree/smt_place_a_tray.json` | 到放置姿态、开夹爪、后撤、夹爪复位 | 旧放置高度和角度要重新试 |
| 底盘移动 | `node/MoveToTarget.py` | `relative_pose` 后退、`slam` 目标点移动的写法 | 旧 SLAM 坐标不要直接用 |
| 夹爪控制 | `node/MoveClaw.py` | `target_positions`、`velocity`、`torque` 的调用思路 | 夹爪开合数值要在仿真里试 |
| 手臂关节轨迹 | `node/MoveArmBaseJointTrajectories.py` | `control_arm_joint_trajectory(times, q_frames)` | 旧关节角只能当初始参考 |
| 手臂目标位姿 | `node/MoveArmBaseTargetPose.py` | 目标 pose + wrench 交给 SDK 执行 | 直接用旧 TargetTag 逻辑可能和你视觉接口不一致 |
| 抓取姿态计算 | `node/CalcArmPose.py` | 先预抓点，再抓取点，再回撤点的思想 | 文件里的层高、偏移、关节角都要重调 |
| 感知循环 | `node/FindAllTag.py` | 循环收集感知结果、写入共享变量/黑板 | 它找 AprilTag，不等于你现在的 RGBD 料盘检测 |

## 今天不要做的事

```text
1. 不要把 pytrees_actions.zip 整个复制进仓库。
2. 不要今天就重构官方 challenge_task.py 成完整行为树。
3. 不要同时追上层、下层、扫码、完整放置精度。
4. 不要写死仿真真值坐标、/mujoco/qpos、/ground_truth/state。
```

## 闫帅辰今天做视觉

第一步，进入服务器项目目录并更新仓库：

```bash
cd ~/code/leju-kuavo-challenge-cup-2026-master/tools

git pull
```

如果服务器上中文路径不好用，复制视觉工具到英文目录：

```bash
cd ~/code/leju-kuavo-challenge-cup-2026-master
mkdir -p tools_vision
cp tools/工具/视觉/*.py tools_vision/
```

第二步，启动 Scene3 后保存 RGB：

```bash
python3 tools/工具/视觉/save_compressed_image.py \
  --topic /cam_h/color/image_raw/compressed \
  --output vision_debug/scene3_day2/scene3_head_rgb.jpg \
  --count 1
```

第三步，标清楚料盘：

```bash
python3 tools/工具/视觉/label_smt_trays.py \
  --image vision_debug/scene3_day2/scene3_head_rgb.jpg \
  --output vision_debug/scene3_day2/scene3_trays_labeled.jpg \
  --json-output vision_debug/scene3_day2/scene3_trays_labeled.json \
  --debug-mask vision_debug/scene3_day2/scene3_trays_mask.jpg
```

看 `scene3_trays_labeled.jpg`，如果框偏了，优先调：

```text
--roi            搜索货架区域
--split-y        上层/下层分界线
--dark-threshold 深色阈值
```

第四步，保存 depth：

```bash
python3 tools/工具/视觉/save_depth_image.py \
  --topic /cam_h/depth/image_raw/compressedDepth \
  --output vision_debug/scene3_day2/scene3_head_depth.npy \
  --png-output vision_debug/scene3_day2/scene3_head_depth_vis.png \
  --count 1
```

第五步，从 `scene3_trays_labeled.json` 里拿 `best.upper.center_pixel` 和 `best.upper.bbox`，转 3D：

```bash
python3 tools/工具/视觉/detect_tray_3d.py \
  --rgb vision_debug/scene3_day2/scene3_head_rgb.jpg \
  --depth vision_debug/scene3_day2/scene3_head_depth.npy \
  --center 640,228 \
  --bbox 453,159,828,298 \
  --level upper \
  --output-json vision_debug/scene3_day2/scene3_upper_tray_3d.json \
  --output-vis vision_debug/scene3_day2/scene3_upper_tray_3d.jpg
```

这里的 `640,228` 和 `453,159,828,298` 要换成你实际 JSON 里的值。

## 王森今天做动作骨架

先不接视觉，先用假目标点测试基础动作。

优先级：

```text
1. 能控制夹爪打开/闭合。
2. 能让手臂走到一个安全预备姿态。
3. 能做一个小幅前伸/回撤动作。
4. 能底盘后退 0.2m。
5. 最后再读视觉 JSON。
```

参考 pytrees 的动作拆法，不照搬旧坐标：

```text
MoveClaw.py                         -> 夹爪开合
MoveArmBaseJointTrajectories.py     -> 预备姿态、收手姿态
MoveArmBaseTargetPose.py            -> 后面接视觉 3D 点
MoveToTarget.py                     -> 抓完后退、去放置区
CalcArmPose.py                      -> 预抓点/抓取点/回撤点的偏移设计
```

建议动作流程先写成最小版本：

```text
stand/准备
打开夹爪
手臂到预备姿态
手臂小幅靠近
闭合夹爪
手臂回撤
底盘后退
打开夹爪
手臂收回
```

## 你们两个的接口

视觉先输出 JSON，不直接控制机器人。

动作先读取这个文件：

```text
vision_debug/scene3_day2/scene3_upper_tray_3d.json
```

重点字段：

```json
{
  "level": "upper",
  "bbox": [453, 159, 828, 298],
  "center_pixel": [640, 228],
  "depth": 1.23,
  "camera_xyz": [0.0, -0.41, 1.23]
}
```

今天联调时，动作组可以先只读 `camera_xyz` 并打印出来，确认接口通了；真正抓取偏移可以明天继续调。

## 今天结束前要留下的东西

闫帅辰留下：

```text
scene3_head_rgb.jpg
scene3_trays_labeled.jpg
scene3_trays_labeled.json
scene3_head_depth.npy
scene3_upper_tray_3d.json
```

王森留下：

```text
能运行的动作测试脚本
夹爪开/合截图或视频
手臂预备姿态截图或视频
失败日志或卡点说明
```

窦欣悦记录：

```text
今天跑了哪个 seed
上层料盘有没有检测出来
动作有没有开夹爪/动手臂
卡在视觉、深度、SDK、动作安全里的哪一步
```
