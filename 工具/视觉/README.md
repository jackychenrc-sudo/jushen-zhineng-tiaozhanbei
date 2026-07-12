# 视觉工具

这些脚本用于离线调试视觉，不是最终完整比赛代码。

目录名用中文方便查找；脚本名保留英文，方便复制命令运行。

## 保存 RGB

```bash
python3 工具/视觉/save_compressed_image.py \
  --topic /cam_h/color/image_raw/compressed \
  --output vision_debug/scene3_seed3/scene3_head_rgb.jpg \
  --count 1
```

如果服务器终端对中文路径不稳定，可以把脚本复制到英文临时目录再运行：

```bash
mkdir -p tools/vision
cp 工具/视觉/*.py tools/vision/
```

## 保存 depth

```bash
python3 工具/视觉/save_depth_image.py \
  --topic /cam_h/depth/image_raw/compressedDepth \
  --output vision_debug/scene3_seed3/scene3_head_depth.npy \
  --png-output vision_debug/scene3_seed3/scene3_head_depth_vis.png \
  --count 1
```

## 离线检测料盘 2D 候选框

```bash
python3 工具/视觉/detect_tray_opencv.py \
  --image vision_debug/scene3_seed3/scene3_head_rgb.jpg \
  --output vision_debug/scene3_seed3/scene3_detected.jpg
```

## 2D + depth 转 3D JSON

先用检测脚本输出的候选中心点，例如 `640,368`，再运行：

```bash
python3 工具/视觉/detect_tray_3d.py \
  --rgb vision_debug/scene3_seed3/scene3_head_rgb.jpg \
  --depth vision_debug/scene3_seed3/scene3_head_depth.npy \
  --center 640,368 \
  --bbox 487,311,794,426 \
  --level upper \
  --output-json vision_debug/scene3_seed3/scene3_tray_result.json \
  --output-vis vision_debug/scene3_seed3/scene3_detected_3d.jpg
```

默认相机参数先使用头部相机估计值：

```text
fx = 392.871
fy = 392.871
cx = 640
cy = 360
```

如果保存了 camera_info，可以后续改脚本从 JSON 读取。
