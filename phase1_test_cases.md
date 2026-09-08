# 阶段一测试用例记录（Checkpoint 1）

本 checkpoint 共 11 条自动化用例，作为第一批人工基线测试。

| 范围 | 用例数 | 主要方法 | 重点 |
|---|---:|---|---|
| linear_assignment | 4 | 等价类、边界值 | 最小代价、矩形矩阵、空矩阵 |
| iou_batch | 5 | 等价类、边界值 | 重合、不重合、部分重叠、包含、接触 |
| associate_detections_to_trackers | 2 | 边界值 | 双方为空、仅有 detection |

当前关键缺陷触发用例：
- `test_linear_assignment_empty_matrix_has_pair_shape`
- `test_association_no_detections_and_no_trackers`
