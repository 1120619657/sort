# 阶段一测试用例记录（Checkpoint 2）

当前累计 19 条自动化用例。

| 范围 | 累计用例数 | 本 checkpoint 增量 | 说明 |
|---|---:|---:|---|
| linear_assignment | 4 | 0 | 保留 BUG-002 回归用例 |
| iou_batch | 7 | +2 | 增加浮点坐标与批量广播 |
| associate_detections_to_trackers | 8 | +6 | 增加成功/失败、多目标、数量不等和阈值边界 |
| 合计 | 19 | +8 | 两个已确认缺陷已修复 |

缺陷关联：
- P1-BUG-001 → `test_association_no_detections_and_no_trackers`
- P1-BUG-002 → `test_linear_assignment_empty_matrix_has_pair_shape`
