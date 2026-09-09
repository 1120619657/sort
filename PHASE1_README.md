# SORT 阶段一测试说明（Checkpoint 4 / 最终状态）

## 最终范围

阶段一累计 41 条自动化用例，覆盖：
- `linear_assignment`
- `iou_batch`
- `convert_bbox_to_z`
- `convert_x_to_bbox`
- `associate_detections_to_trackers`
- `KalmanBoxTracker`
- `Sort.update`
- demo 单行 detection 文件读取路径

测试设计采用等价类、边界值和场景法。NaN/Inf、Hypothesis、随机/模糊测试、系统性质测试和长序列压力仍保留到阶段二。

## 三个已确认缺陷

1. P1-BUG-001：空 tracker 分支返回二维空框而非一维索引。
2. P1-BUG-002：空代价矩阵返回 shape `(0,)` 而非 `(0,2)`。
3. P1-BUG-003：单行合法 detection 文件被 `np.loadtxt` 读取为一维数组，demo 后续二维索引崩溃。

## 运行

```bash
python -m pip install -r requirements.txt
python -m pip install pytest coverage
pytest tests_phase1 -v
coverage erase
coverage run -m pytest tests_phase1
coverage report -m --include='sort.py'
```

## 缺陷回归

```bash
pytest tests_phase1/test_linear_assignment.py::test_linear_assignment_empty_matrix_has_pair_shape -v
pytest tests_phase1/test_association.py::test_association_no_detections_and_no_trackers -v
pytest tests_phase1/test_demo_input.py::test_demo_accepts_single_detection_row -v
```
