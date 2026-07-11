# 具身智能挑战杯协作仓库

本仓库用于管理 2026 挑战杯具身智能操作赛 / Kuavo 仿真初赛的团队代码、视觉工具、任务记录和提交材料。

当前策略：先做可得分闭环，再做稳定性优化。所有代码和文档都围绕三个仿真场景服务：

| 场景 | 任务 | 负责人 | 第一阶段目标 |
| --- | --- | --- | --- |
| Scene1 | 包裹称重与入箱 | 王仔俊（$$） | 先完成 1 个包裹抓取、称重、入箱，再扩展到 4 个 |
| Scene2 | 零件识别、分类、放入对应盒子 | 史智涛（真空期）+ Kevin | 先识别并放对 1 个零件，再扩展到 6 个 |
| Scene3 | SMT 料盘出库 | 闫帅辰 + 王森（易水寒） | 先完成上层料盘，再尝试下层料盘 |
| 视觉与坐标 | 相机读取、目标检测、深度、3D 坐标、抓取点偏移 | 闫帅辰，王仔俊辅助 | 先打通 RGB + depth -> camera_xyz -> JSON |
| 测试与提交 | 测试表、README、截图视频、最终打包 | 窦欣悦（精准控温洗澡水） | 每天记录可运行结果，最后整理提交包 |

## 仓库结构

```text
.
├── README.md
├── docs/
│   ├── day2_plan.md              # 明天环境配置好后的具体任务板
│   ├── git_workflow.md           # 团队 GitHub 协作规则
│   ├── submit_checklist.md       # 最终提交检查清单
│   ├── team_roles.md             # 成员分工和交付物
│   └── test_record.md            # 每天测试记录模板
├── src/
│   ├── scene1/README.md          # Scene1 工作区说明
│   ├── scene2/README.md          # Scene2 工作区说明
│   ├── scene3/README.md          # Scene3 工作区说明
│   └── vision/README.md          # 视觉模块接口说明
├── submission/
│   └── README.md                 # 最终提交包整理说明
└── tools/
    └── vision/
        ├── README.md
        ├── save_compressed_image.py
        ├── save_depth_image.py
        ├── detect_tray_opencv.py
        └── detect_tray_3d.py
```

## 当前进展

更新时间：2026-07-12

已经完成：

1. 远程服务器仿真环境可以启动 Scene3。
2. 已确认头部、左手、右手相机话题，均有 RGB 和 depth。
3. 头部 RGB 图可以保存，能看到货架和 SMT 料盘。
4. `detect_tray_opencv.py` 已经能粗框货架 / 料盘候选区域。
5. GitHub 到服务器的代码同步流程已跑通。

下一步只盯一个小闭环：

```text
Scene3 RGB 图检测料盘 bbox
-> 保存 depth
-> 取 bbox center 附近深度中位数
-> 用 camera_info 转 camera_xyz
-> 输出 JSON 给动作组
```

## 明天优先级

先看 [docs/day2_plan.md](docs/day2_plan.md)。

每个人都要产出可运行结果，不要只研究方案：

- Scene1：能检测 / 接近 1 个包裹。
- Scene2：能识别 1 个零件并输出 3D 点。
- Scene3：能输出上层料盘 3D 点，动作骨架能动手臂和夹爪。
- 测试：有测试表、有截图、有失败原因。

## 服务器下载工具

在远程 Ubuntu 的比赛仓库根目录执行：

```bash
cd ~/code/leju-kuavo-challenge-cup-2026-master
mkdir -p tools/vision
cd tools/vision

curl -L -O https://raw.githubusercontent.com/jackychenrc-sudo/jushen-zhineng-tiaozhanbei/repo-structure-day2/tools/vision/save_compressed_image.py
curl -L -O https://raw.githubusercontent.com/jackychenrc-sudo/jushen-zhineng-tiaozhanbei/repo-structure-day2/tools/vision/save_depth_image.py
curl -L -O https://raw.githubusercontent.com/jackychenrc-sudo/jushen-zhineng-tiaozhanbei/repo-structure-day2/tools/vision/detect_tray_opencv.py
curl -L -O https://raw.githubusercontent.com/jackychenrc-sudo/jushen-zhineng-tiaozhanbei/repo-structure-day2/tools/vision/detect_tray_3d.py

chmod +x *.py
```

## 比赛规则提醒

正常方案只能使用机器人自身传感器和允许的控制接口。不要使用：

- `/mujoco/qpos`
- `/ground_truth/state`
- `/set_object_position`
- 评分逻辑或场景模型修改
- 固定 seed 写死位置

本仓库视觉工具只读相机图像和相机参数，属于正常视觉方案。

## 协作原则

- GitHub 管代码和文档。
- 网盘管 rosbag、大视频、大模型权重。
- `main` 保持稳定。
- 每个人用自己的分支开发。
- 每天晚上把能跑的东西合并。
- 最终提交包以官方 `challenge_cup_task_template` 为准，本仓库用于协作和整理。