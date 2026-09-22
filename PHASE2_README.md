# SORT 模块二：AI 辅助测试

模块二采用“AI 测”方案：保持阶段一 SORT 被测对象不变，由 AI 分析 `sort.py` 与阶段一覆盖缺口，生成新的测试场景、测试数据与 pytest 代码；失败结果仍需人工判断是有效缺陷、测试预期错误还是算法设计限制。

## Checkpoint 规划

1. AI 异常输入与数值鲁棒性：6 条
2. AI 参数组合与状态边界：6 条
3. AI 性质测试：5 条
4. AI 复杂/长序列场景：5 条

最终共 22 条阶段二 AI 测试。阶段一已有 41 条测试保留在 `tests_phase1/`，阶段二代码放在 `tests_phase2/`，避免重复计算阶段一用例。

## 运行

```bash
pytest tests_phase2 -v
pytest tests_phase1 tests_phase2 -v
```

性质测试 Checkpoint 引入 Hypothesis：

```bash
pip install hypothesis
```

本仓库原始 `sort.py` 使用 `TkAgg`。在无图形界面的 CI/Linux 环境中，可临时将 Matplotlib 后端设置为 `Agg` 后执行测试；该环境兼容调整不属于 SORT 业务修复。

## 当前进度

当前 Checkpoint：4/4，阶段二 22 条 AI 测试代码已全部生成：鲁棒性 6 条 + 参数组合 6 条 + 性质测试 5 条 + 复杂/长序列 5 条。

异常输入测试采用保守判定：被测函数可以明确 `ValueError` 拒绝异常数值，也可以返回有限数值；静默产生 NaN/Inf 视为测试失败，需要人工复核后才能计为有效缺陷。
