# 具身智能挑战杯项目记录

这个仓库用于存放我们参加具身智能挑战杯初赛时的辅助代码、视觉调试脚本、场景记录和后续说明。当前阶段先服务于 Scene3「SMT 料盘出库」任务，重点是把相机图像读取、料盘识别、坐标/抓取点输出这条链路先跑通。

## 当前进展

更新时间：2026-07-12

目前已经完成的前期验证：

1. 服务器仿真环境已能启动 Scene3。
2. 已确认机器人有头部、左手、右手三组相机话题，并且有 RGB 图和深度图。
3. 头部相机 RGB 图可以正常保存，已经能看到货架和 SMT 料盘。
4. 远程服务器可以访问 GitHub raw 链接，因此后续脚本可以通过 GitHub 下载到服务器。
5. 本次上传了两个 Scene3 视觉调试脚本，先用于“看图、存图、检测大概位置”，不是最终完整方案。

## 本次上传文件说明

| 文件 | 作用 | 当前用途 |
| --- | --- | --- |
| `save_compressed_image.py` | 从 ROS 的压缩相机话题中保存图片 | 用来保存头部/手部相机画面，方便后续调试、标注、比较不同 seed 下的场景 |
| `detect_tray_opencv.py` | 用 OpenCV 对保存下来的图片做初步检测 | 先尝试从头部相机图像里找到货架/料盘的大致区域，输出候选框、中心点和可视化结果 |

简单理解：

- `save_compressed_image.py` 负责“把机器人看到的画面存下来”。
- `detect_tray_opencv.py` 负责“在存下来的图里先粗略找料盘在哪里”。

这两个脚本主要是视觉调试工具，目的是让我们先验证：相机能不能用、图像是否清楚、料盘在图里大概长什么样、能不能稳定找出候选区域。

## 服务器下载方式

在远程 Ubuntu 的项目目录下执行：

```bash
cd ~/code/leju-kuavo-challenge-cup-2026-master
mkdir -p vision_debug/scene3_tools
cd vision_debug/scene3_tools

curl -L -o detect_tray_opencv.py https://raw.githubusercontent.com/jackychenrc-sudo/jushen-zhineng-tiaozhanbei/main/detect_tray_opencv.py
curl -L -o save_compressed_image.py https://raw.githubusercontent.com/jackychenrc-sudo/jushen-zhineng-tiaozhanbei/main/save_compressed_image.py

chmod +x *.py
ls -lh
```

## 后续建议流程

Scene3 视觉部分建议按这个顺序推进：

1. 保存不同 seed 下的头部相机图像。
2. 用 `detect_tray_opencv.py` 跑保存图，观察能不能稳定框出料盘/货架区域。
3. 如果 OpenCV 阈值法不稳定，再考虑换 YOLO 或 SAM 辅助分割。
4. 视觉模块最终要给动作模块输出：物体类别、图像中心点、候选框、深度/距离、建议抓取点。
5. 动作模块再根据这些信息完成靠近、抓取、抬起、转移、放置。

## 重要规则提醒

比赛中不能为了图方便读取仿真真值坐标，尤其不要使用这些内容：

- 不要订阅 `/mujoco/qpos`
- 不要订阅 `/ground_truth/state`
- 不要调用 `/set_object_position` 或类似摆物体服务
- 不要修改评分逻辑
- 不要修改仿真场景来降低难度
- 不要针对固定 seed 写死位置

我们目前这些脚本只使用相机图像，属于正常视觉方案。

## 后续提交说明建议

大家后面往仓库里加代码时，建议在 README 或单独日志里写清楚：

```text
日期：
负责人：
修改文件：
这次解决了什么问题：
怎么运行/怎么测试：
当前还没解决的问题：
```

这样后面别人接手时不会断档，也方便最后整理 README、测试记录和提交材料。
