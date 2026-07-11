# 视觉模块

负责人：闫帅辰，王仔俊辅助采样。

## 目标

视觉模块不直接控制机器人，只负责回答：

```text
物体是什么？
在图像哪里？
深度是多少？
相机坐标是多少？
建议抓取点在哪里？
```

## 统一输出格式

```json
{
  "scene": "scene3",
  "object": "smt_tray",
  "level": "upper",
  "bbox": [487, 311, 794, 426],
  "center_pixel": [640, 368],
  "depth": 1.23,
  "camera_xyz": [0.0, 0.025, 1.23],
  "grasp_hint": "front_center",
  "confidence": 0.8
}
```

## Day2 优先级

1. Scene3 上层料盘 3D 点。
2. Scene2 至少一个零件 3D 点。
3. Scene1 至少一个包裹 3D 点。

## 不做什么

暂时不要上复杂模型。先用 OpenCV + depth 打通闭环。
