# P1-BUG-001 / P1-BUG-002 修复与回归记录

## 目的

本记录对应第一批测试暴露的两个返回契约缺陷，保存修复依据和回归方式。

## P1-BUG-001

- 触发：0 detection、0 tracker。
- 原始结果：`unmatched_trackers.shape == (0,5)`。
- 预期：未匹配轨迹是索引集合，空集合应为 `(0,)`。
- 修复：早返回第三项改为一维空整数数组。

## P1-BUG-002

- 触发：`0×0` 空代价矩阵。
- 原始结果：线性分配返回 shape `(0,)`。
- 预期：空匹配仍保持两列 `[row, column]`，shape `(0,2)`。
- 修复：函数入口显式处理空矩阵。

## 回归命令

```bash
pytest tests_phase1/test_linear_assignment.py::test_linear_assignment_empty_matrix_has_pair_shape -v
pytest tests_phase1/test_association.py::test_association_no_detections_and_no_trackers -v
pytest tests_phase1 -v
```

Checkpoint 2 实际结果：第一批 19 条测试全部通过。
