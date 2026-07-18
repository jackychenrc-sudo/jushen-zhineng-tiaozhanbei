# Scene2 学长交接包（2026-07-17）

## 这次推了什么

本目录保存 **Scene2 红色螺丝刀抓取的完整学长交接材料**，原样归档，没有修改抓取算法。

- `学长交接包_scene2_20260717_230742.zip`：完整原始交接包，包含当前实验源码、5A123 闭爪基线、运行日志、掉件诊断、失败偏移试验、视觉图片和官方注意事项。
- `5A123_baseline/challenge_cup_task_template/`：从交接材料整理出的可直接查看/运行的 5A123 闭爪基线功能包。
- 红色抓取主脚本：`scripts/scene2_pick_nearest_red.py`
- 红色主脚本 SHA-256：`5A123F3FD1C64F795F3773C3F4BFF5D1CD7A6ABC295C5FB53A05369EC0A39338`
- 完整交接 ZIP SHA-256：`1B4BE5F813120221E1C6995815B424CD5DA6C7D9F4C54BA5DFB682F2B4895222`

## 当前能力与限制

交接记录显示已经实现 RGB/深度识别红色零件、高位移动、PREGRASP 和中心位置闭爪。闭爪后的稳定抬升、搬运及放置仍需在官方 Docker 中继续验证。这里上传的是学弟/学长交接代码，不是官方 collect_scene2_dataset 演示答案。

`challenge_task.py --scene scene2` 在这份基线里负责启动场景；抓取脚本需要在第二个终端单独运行。

## 服务器获取

```bash
git clone --branch vision-yanshuaichen --single-branch \
  https://github.com/jackychenrc-sudo/jushen-zhineng-tiaozhanbei.git
cd jushen-zhineng-tiaozhanbei/scene2_handoff
sha256sum 5A123_baseline/challenge_cup_task_template/scripts/scene2_pick_nearest_red.py
```

校验结果必须是：

```text
5a123f3fd1c64f795f3773c3f4bff5d1cd7a6abc295c5fb53a05369ec0a39338
```
