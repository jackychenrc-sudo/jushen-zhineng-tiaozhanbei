# 具身智能挑战杯协作仓库

本仓库用于管理 2026 挑战杯具身智能操作赛 / Kuavo 仿真初赛的团队代码、视觉工具、任务记录和提交材料。

## 队友先看这里

第一次用这个仓库的队友，先打开：

[先看这个_队友操作说明.md](先看这个_队友操作说明.md)

里面已经写好：

```text
怎么 clone 仓库
每个人切哪个分支
每天怎么 pull / commit / push
每个人改哪个目录
Scene1 / Scene2 / Scene3 怎么启动
视觉工具怎么保存 RGB、depth、输出 3D JSON
晚上怎么开 PR 和合并
```

当前策略：先做可得分闭环，再做稳定性优化。所有代码和文档都围绕三个仿真场景服务。

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
├── 先看这个_队友操作说明.md
├── 文档/
│   ├── 第二天任务板.md
│   ├── GitHub协作规则.md
│   ├── 提交检查清单.md
│   ├── 团队分工.md
│   └── 测试记录.md
├── 任务代码参考/
│   ├── 场景一_包裹称重入箱/README.md
│   ├── 场景二_零件分拣/README.md
│   ├── 场景三_SMT料盘出库/README.md
│   └── 视觉与坐标/README.md
├── 工具/
│   └── 视觉/
│       ├── README.md
│       ├── save_compressed_image.py
│       ├── save_depth_image.py
│       ├── detect_tray_opencv.py
│       └── detect_tray_3d.py
└── 最终提交包/
    └── README.md
```

说明：目录和文档尽量用中文，方便队友理解；真正要运行的 Python 脚本保留英文文件名，避免命令行、Python import、ROS 环境出现编码问题。

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

先看 [文档/第二天任务板.md](文档/第二天任务板.md)。

每个人都要产出可运行结果，不要只研究方案：

- Scene1：能检测 / 接近 1 个包裹。
- Scene2：能识别 1 个零件并输出 3D 点。
- Scene3：能输出上层料盘 3D 点，动作骨架能动手臂和夹爪。
- 测试：有测试表、有截图、有失败原因。

## 服务器下载工具

如果是整个仓库 clone：

```bash
cd ~/code/leju-kuavo-challenge-cup-2026-master
git clone https://github.com/jackychenrc-sudo/jushen-zhineng-tiaozhanbei.git team_repo
cd team_repo
git checkout 中文结构整理
```

如果只下载视觉工具，可以执行：

```bash
cd ~/code/leju-kuavo-challenge-cup-2026-master
mkdir -p tools/vision
cd tools/vision

curl -L -o save_compressed_image.py "https://raw.githubusercontent.com/jackychenrc-sudo/jushen-zhineng-tiaozhanbei/中文结构整理/工具/视觉/save_compressed_image.py"
curl -L -o save_depth_image.py "https://raw.githubusercontent.com/jackychenrc-sudo/jushen-zhineng-tiaozhanbei/中文结构整理/工具/视觉/save_depth_image.py"
curl -L -o detect_tray_opencv.py "https://raw.githubusercontent.com/jackychenrc-sudo/jushen-zhineng-tiaozhanbei/中文结构整理/工具/视觉/detect_tray_opencv.py"
curl -L -o detect_tray_3d.py "https://raw.githubusercontent.com/jackychenrc-sudo/jushen-zhineng-tiaozhanbei/中文结构整理/工具/视觉/detect_tray_3d.py"

chmod +x *.py
```

如果 curl 对中文 URL 不稳定，就用 `git clone` 方式。

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