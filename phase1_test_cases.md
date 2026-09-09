# SORT 阶段一最终测试用例概览

阶段一最终累计 41 条自动化测试用例。

| 模块 | 用例数 |
|---|---:|
| linear_assignment | 4 |
| iou_batch | 7 |
| 坐标转换 | 8 |
| associate_detections_to_trackers | 8 |
| KalmanBoxTracker | 5 |
| Sort.update | 8 |
| demo 单行检测输入 | 1 |
| 合计 | 41 |

测试设计覆盖等价类、边界值和场景法。三个缺陷分别由空 tracker、空代价矩阵和单行 detection 文件边界触发，并均保留自动化回归用例。
