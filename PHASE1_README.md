# SORT 阶段一测试说明（Checkpoint 2）

## 当前范围

在 Checkpoint 1 基础上完成第一批测试：`linear_assignment`、`iou_batch`、`associate_detections_to_trackers`，共 19 条自动化用例。

## 本次修复

- P1-BUG-001：空 tracker 时 `unmatched_trackers` 从二维空框数组统一为一维空索引数组。
- P1-BUG-002：空代价矩阵时 `linear_assignment` 统一返回 `(0, 2)` 整数数组。

两项均采用“失败用例 → 最小修复 → 目标回归 → 第一批全量回归”的方式验证。

## 安装与运行

```bash
python -m pip install -r requirements.txt
python -m pip install pytest
pytest tests_phase1 -v
```

单独验证两个回归用例：

```bash
pytest tests_phase1/test_linear_assignment.py::test_linear_assignment_empty_matrix_has_pair_shape -v
pytest tests_phase1/test_association.py::test_association_no_detections_and_no_trackers -v
```
