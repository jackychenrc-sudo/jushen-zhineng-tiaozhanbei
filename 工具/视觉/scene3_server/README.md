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
scene3_live_perception.py       # 旧版轮廓检测与 RGB-D 调试
scene3_upper_tray_perception.py  # 正式上层模板检测：输出深度和 base_link 坐标
scene3_dataset_collector.py        # 按 seed 采集 RGB、深度、内参和 TF
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
source devel/setup.bash
rosrun challenge_cup_task_template challenge_task.py --scene scene3 --seed 3
```

这个终端保持运行，不要关。

## 跑环境检查

在 Docker / ROS 终端 2：

```bash
cd /root/kuavo_ws
source devel/setup.bash
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
source devel/setup.bash
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

## 正式上层料盘感知

完成一次上层料盘模板选择后，在容器终端运行：

```bash
cd /root/kuavo_ws
source devel/setup.bash
python3 tools/工具/视觉/scene3_server/scene3_upper_tray_perception.py \
  --template vision_debug/scene3_seed3_live/upper_tray_template.png \
  --output-dir vision_debug/scene3_upper_live
```

正式节点会：

- 以低阈值召回上层候选，再用高分料盘建立 RGB-D 货架平面并过滤假框
- 使用候选框内第 10 百分位深度，避免读到料盘后方背景
- 输出相机坐标和 `base_link_xyz_m`
- 保存 `upper_trays.jpg`、`upper_candidates.jpg`、`upper_trays.json`
- 发布只读结果话题 `/scene3/upper_trays`

## 采集跨 seed RGB-D 数据

启动指定 seed 的 Scene3 后，在容器终端运行：

```bash
cd /root/kuavo_ws
source devel/setup.bash
python3 tools/工具/视觉/scene3_server/scene3_dataset_collector.py \
  --seed 3 --camera head --count 10
```

数据按 `seed/camera/run` 保存到 `vision_dataset/scene3`。每帧包含 RGB、米制浮点深度、16 位毫米深度 PNG、深度预览、相机内参、相机到 `base_link` 的 TF 和 JSON 清单。采集器只读取相机与 TF，不控制机器人；原始数据不要提交到 GitHub。

## 下一步判断

当前流程已经在 seed 3 找到 3 个上层料盘并完成 `base_link` 坐标转换。接入机械臂前，至少再测试 3 个不同 seed；只有候选数量、深度和坐标都稳定后，才开始预抓取动作。
