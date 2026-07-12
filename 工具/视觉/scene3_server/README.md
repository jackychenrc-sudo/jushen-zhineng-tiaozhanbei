# Scene3 服务器视觉首轮测试

这个目录用于 Scene3「SMT 料盘出库」的第一轮服务器测试。

当前目标只有一个：

- 从头部相机读取 RGB-D
- 标出上层/下层 SMT 料盘候选框
- 读取候选点深度
- 输出相机坐标系下的 `camera_xyz_m`

注意：这里的脚本只做视觉，不会控制机器人移动。

## 文件说明

```text
label_smt_trays.py              # 离线图片标注：给保存下来的图片画料盘候选框
scene3_environment_probe.py     # 服务器环境检查：ROS 话题、服务、TF、Python 包
scene3_live_perception.py       # 在线视觉测试：读取相机 RGB-D，输出 JSON 和标注图
```

## 服务器更新代码

如果服务器上已经 clone 过本仓库，一般在比赛项目目录下执行：

```bash
cd /home/shx/code/leju-kuavo-challenge-cup-2026-master/tools
git pull
```

如果服务器上还没有 clone：

```bash
cd /home/shx/code/leju-kuavo-challenge-cup-2026-master
git clone https://github.com/jackychenrc-sudo/jushen-zhineng-tiaozhanbei.git tools
```

## 启动 Scene3

在 Docker / ROS 终端 1：

```bash
cd /root/kuavo_ws
source devel/setup.zsh
rosrun challenge_cup_task_template challenge_task.py --scene scene3 --seed 3
```

这个终端保持运行，不要关。

## 跑环境检查

在 Docker / ROS 终端 2：

```bash
cd /root/kuavo_ws
source devel/setup.zsh
python3 tools/工具/视觉/scene3_server/scene3_environment_probe.py | tee /tmp/scene3_environment.json
```

如果提示找不到文件，先检查：

```bash
ls /root/kuavo_ws/tools/工具/视觉/scene3_server
```

## 跑视觉检测

继续在 Docker / ROS 终端 2：

```bash
cd /root/kuavo_ws
source devel/setup.zsh
python3 tools/工具/视觉/scene3_server/scene3_live_perception.py --level upper --output-dir /tmp/scene3_upper | tee /tmp/scene3_perception.log
```

输出文件：

```text
/tmp/scene3_upper/tray_detection.json
/tmp/scene3_upper/tray_candidates.jpg
```

查看 JSON：

```bash
cat /tmp/scene3_upper/tray_detection.json
```

## 下一步判断

如果 `tray_candidates.jpg` 框到了正确的上层料盘，就继续做：

- 反复跑不同 seed
- 调 ROI 和阈值
- 用 TF 把 `camera_xyz_m` 转到机器人基座坐标
- 再接机械臂预抓取动作

如果框错了，先把下面三个东西发出来再改参数：

```text
/tmp/scene3_environment.json
/tmp/scene3_upper/tray_detection.json
/tmp/scene3_upper/tray_candidates.jpg
```
