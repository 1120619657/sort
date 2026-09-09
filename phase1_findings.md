# 阶段一缺陷记录（Checkpoint 3）

## 已关闭缺陷

### P1-BUG-001 空 tracker 分支返回错误形状
已在 Checkpoint 2 最小修复并回归通过。

### P1-BUG-002 空代价矩阵返回格式不稳定
已在 Checkpoint 2 最小修复并回归通过。

## 新发现：P1-BUG-003 单行检测文件导致 demo 读取维度异常

- 场景：MOT 格式 `det.txt` 仅包含一条合法 detection。
- 触发数据示例：`1,1,10,20,30,40,0.9,-1,-1,-1`。
- 合理预期：demo 能处理最小规模合法检测文件并生成输出。
- 实际：`np.loadtxt` 对单行文件返回一维 ndarray，随后 `seq_dets[:,0]` 使用二维索引导致 `IndexError`。
- 根因：demo 默认 `np.loadtxt` 始终返回二维检测矩阵，没有统一单行输入维度。
- 当前状态：已由 `test_demo_accepts_single_detection_row` 稳定复现，尚未修复。

本 checkpoint 的目的就是保存 BUG-003 修复前状态，便于后续形成清晰的测试发现过程。
