# GitHub 协作规则

## 分支命名

每个人不要直接改 `main`。

建议分支：

```text
scene1-wangzijun
scene2-shizhitao-kevin
scene3-wangsen
vision-yanshuaichen
docs-douxinyue
```

## 每天开工

```bash
git checkout main
git pull
```

新建自己的分支：

```bash
git checkout -b vision-yanshuaichen
```

如果分支已经存在：

```bash
git checkout vision-yanshuaichen
git pull origin vision-yanshuaichen
```

## 提交代码

```bash
git status
git add <改过的文件>
git commit -m "vision: save depth and export tray xyz"
git push origin vision-yanshuaichen
```

## 不要提交这些

```text
build/
devel/
log/
*.bag
*.mp4
*.avi
*.pt
*.pth
大型数据集
大模型权重
```

大文件放网盘，只在 README 或测试记录里贴路径。

## 每晚合并

1. 每个人 push 自己分支。
2. 在 GitHub 开 Pull Request。
3. 测试同学看能否运行。
4. 能跑就合并到 main。
5. main 必须保持能启动。

## 关键原则

不要多人同时改同一个大文件。

推荐结构：

```text
Scene1 组主要改 src/scene1 或官方包里的 scene1_task.py
Scene2 组主要改 src/scene2 或官方包里的 scene2_task.py
Scene3 组主要改 src/scene3 或官方包里的 scene3_task.py
视觉组主要改 src/vision 和 tools/vision
测试同学主要改 docs
```

最终接入官方 `challenge_task.py` 时，只保留一个统一入口：

```python
if scene == "scene1":
    run_scene1()
elif scene == "scene2":
    run_scene2()
elif scene == "scene3":
    run_scene3()
```
