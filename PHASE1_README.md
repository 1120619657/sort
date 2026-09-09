# SORT 阶段一测试说明（Checkpoint 3）

## 当前范围

在前两个缺陷修复基础上，新增坐标转换、`Sort.update` 基本跨帧场景以及 demo 检测文件读取测试。累计 32 条自动化用例，已经达到课程对阶段一测试用例数量的基本要求。

新增重点：
- `convert_bbox_to_z` / `convert_x_to_bbox` 的代表性矩形与往返转换。
- `Sort.update` 的单目标连续运动、多目标、新目标进入、目标离开。
- demo 最小合法输入：只有一条 detection 的 MOT 格式 `det.txt`。

## 运行

```bash
pytest tests_phase1 -v
```

也可单独运行新增模块：

```bash
pytest tests_phase1/test_bbox_conversion.py -v
pytest tests_phase1/test_sort_update.py -v
pytest tests_phase1/test_demo_input.py -v
```

## 当前预期

该 checkpoint 故意保留原始 demo 文件读取逻辑，不包含 P1-BUG-003 的修复。因此单行 detection 文件测试应稳定失败，其余阶段一用例应通过。这里用于保留“新增测试先暴露缺陷”的证据。
