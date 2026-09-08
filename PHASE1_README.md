# SORT 阶段一测试说明（Checkpoint 1）

## 当前范围

第一批测试覆盖 `linear_assignment`、`iou_batch` 和 `associate_detections_to_trackers` 的基础功能、空输入边界和典型匹配行为。当前使用 SORT 原始基线代码，不包含任何缺陷修复。

## 测试方法

- 等价类：单元素/普通矩阵、重合/不重合矩形、正常 detection 输入。
- 边界值：空代价矩阵、空 tracker、矩形边界接触。

## 运行环境

建议 Python 3.9+，安装 `requirements.txt` 以及 `pytest`。

```bash
python -m pip install -r requirements.txt
python -m pip install pytest
```

## 运行第一批测试

```bash
pytest tests_phase1/test_linear_assignment.py -v
pytest tests_phase1/test_iou.py -v
pytest tests_phase1/test_association.py -v
pytest tests_phase1 -v
```

## 当前预期

原始基线代码应暴露两个边界返回契约问题：空代价矩阵的分配结果形状，以及空 tracker 分支的未匹配轨迹索引形状。此 checkpoint 只负责复现，不修复。
