# Scene2：零件分拣

负责人：史智涛（真空期）+ Kevin

## 第一阶段目标

不要先做 6 个零件。先完成 1 个零件闭环：

```text
识别一个零件 -> 判断类别 -> 得到 3D 点 -> 放入对应箱子附近
```

## 输出格式

```json
{
  "scene": "scene2",
  "part_type": "a",
  "center_pixel": [0, 0],
  "depth": 0.0,
  "camera_xyz": [0.0, 0.0, 0.0],
  "target_bin": "purple"
}
```

## Day2 交付

- 保存 Scene2 RGB/depth/camera_info。
- 至少识别一个零件。
- 至少输出一个零件 3D 点。
- 尝试接近或放入对应箱附近。
