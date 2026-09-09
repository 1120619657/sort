# 阶段一测试用例记录（Checkpoint 3）

当前累计 32 条自动化用例。

| 范围 | 累计用例数 | 新增用例数 |
|---|---:|---:|
| 第一批：linear_assignment / iou / association | 19 | 0 |
| 坐标转换 | 8 | +8 |
| Sort.update 基本跨帧场景 | 4 | +4 |
| demo 单行检测文件 | 1 | +1 |
| 合计 | 32 | +13 |

新增测试方法同时覆盖等价类、边界值和场景法。P1-BUG-003 对应 `test_demo_accepts_single_detection_row`，属于“最小合法数据规模”的边界值与场景测试。
