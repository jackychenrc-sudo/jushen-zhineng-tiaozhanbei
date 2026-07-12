# 测试记录

测试负责人：窦欣悦（精准控温洗澡水）

## 每日汇总

| 日期 | Scene | Seed | 负责人 | 是否启动 | 是否识别 | 是否有 3D 点 | 是否动作执行 | 是否得分 | 失败原因 | 截图/视频 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-07-13 | scene1 | 0 | 王仔俊 | 待测 | 待测 | 待测 | 待测 | 待测 |  |  |
| 2026-07-13 | scene2 | 0 | 史智涛 + Kevin | 待测 | 待测 | 待测 | 待测 | 待测 |  |  |
| 2026-07-13 | scene3 | 3 | 闫帅辰 + 王森 | 待测 | 待测 | 待测 | 待测 | 待测 |  |  |

## 单次测试记录模板

```text
日期：
测试人：
场景：scene1 / scene2 / scene3
seed：
代码分支：
运行命令：
是否成功启动：
是否有视觉输出：
是否有 3D 坐标：
是否执行动作：
是否得分：
截图路径：
视频路径：
失败现象：
下一步建议：
```

## 运行命令记录

Scene1：

```bash
rosrun challenge_cup_task_template challenge_task.py --scene scene1 --seed 0
```

Scene2：

```bash
rosrun challenge_cup_task_template challenge_task.py --scene scene2 --seed 0
```

Scene3：

```bash
rosrun challenge_cup_task_template challenge_task.py --scene scene3 --seed 3
```

## 最小通过标准

Day2 不要求满分，只要求有证据：

- 有截图。
- 有命令。
- 有输出文件。
- 有失败原因。
- 有下一步。
