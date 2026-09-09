# 阶段一缺陷记录（Checkpoint 2）

## P1-BUG-001 空 tracker 分支返回错误形状

- 触发：`detections.shape=(0,5)`，`trackers.shape=(0,5)`。
- 预期：`unmatched_trackers` 是一维索引数组，空值 shape 为 `(0,)`。
- 基线实际：shape 为 `(0,5)`。
- 根因：早返回把 tracker 框的二维列结构误用于索引集合。
- 最小修复：改为 `np.empty((0,), dtype=int)`。
- 回归：目标测试及第一批全量测试通过。

## P1-BUG-002 空代价矩阵返回格式不稳定

- 触发：`linear_assignment(np.empty((0,0)))`。
- 预期：空匹配仍维持 `[row_index, column_index]` 两列结构，即 `(0,2)`。
- 基线实际：SciPy 回退路径形成 shape `(0,)`。
- 根因：函数未对空矩阵显式规范返回契约。
- 最小修复：入口检测 `cost_matrix.size == 0` 并返回 `(0,2)` 整数数组。
- 回归：目标测试及第一批全量测试通过。

## 第一批测试状态

两项已确认软件缺陷已完成闭环。其余测试用于回归正常输入、IoU 典型边界以及 detection-tracker 的常见匹配组合。
