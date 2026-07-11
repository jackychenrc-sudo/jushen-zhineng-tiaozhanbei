# 提交检查清单

最终提交以前逐项检查。

## 官方要求方向

最终核心是能被官方评测运行的 ROS 功能包：

```text
challenge_cup_task_template/
├── CMakeLists.txt
├── package.xml
├── README.md
└── scripts/
    └── challenge_task.py
```

如果修改了其他官方功能包，也要一起提交，并在 README 说明。

## 代码检查

- [ ] `challenge_cup_task_template` 包名没有改。
- [ ] `scripts/challenge_task.py` 入口存在。
- [ ] 三个场景都能通过 `--scene scene1/scene2/scene3` 启动。
- [ ] 没有读取禁用真值话题。
- [ ] 没有调用禁用摆放服务。
- [ ] 没有修改评分、场景、模型来降低难度。
- [ ] 没有针对固定 seed 写死位置。
- [ ] 没有提交 `build/`、`devel/`、`log/`。
- [ ] 没有提交 rosbag、大视频、大模型权重。

## 文档检查

- [ ] README 写清楚环境。
- [ ] README 写清楚编译方式。
- [ ] README 写清楚运行方式。
- [ ] README 写清楚每个 scene 当前能力。
- [ ] README 写清楚额外依赖。
- [ ] 有测试记录。
- [ ] 有截图或视频说明。

## 交付命名

按赛事通知，作品电子文件建议按：

```text
题目编号+学校或单位+团队负责人
```

例如：

```text
XH-202611-学校名-闫帅辰.zip
```

最终以最新通知为准。

## 打包前命令

在官方环境里至少跑一次：

```bash
rosrun challenge_cup_task_template challenge_task.py --scene scene1 --seed 0
rosrun challenge_cup_task_template challenge_task.py --scene scene2 --seed 0
rosrun challenge_cup_task_template challenge_task.py --scene scene3 --seed 3
```

能启动、有输出、有日志，再打包。
