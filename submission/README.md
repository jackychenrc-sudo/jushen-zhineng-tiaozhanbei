# 最终提交包说明

本目录用于记录最终提交包应该包含什么。

注意：本仓库不是官方比赛工作空间本身。最终要从官方工作空间里整理可运行的 ROS 功能包。

## 最小提交结构

```text
参赛团队名称+挑战杯仿真赛/
└── challenge_cup_task_template/
    ├── CMakeLists.txt
    ├── package.xml
    ├── README.md
    └── scripts/
        └── challenge_task.py
```

## 如果新增了辅助模块

可以放在 `challenge_cup_task_template` 内部，例如：

```text
challenge_cup_task_template/
├── scripts/
│   ├── challenge_task.py
│   ├── scene1_task.py
│   ├── scene2_task.py
│   ├── scene3_task.py
│   └── vision_utils.py
└── config/
    └── params.yaml
```

## README 必须说明

- 修改了哪些文件。
- 如何编译。
- 如何运行三个 scene。
- 是否有额外依赖。
- 当前测试结果。
- 已知限制。

## 不要打包

```text
build/
devel/
log/
rosbag
大视频
大模型权重
缓存文件
```
