# Issue tracker: Local Markdown

工单与规格保存在本仓库 `.scratch/`，使用本地文件发布与读取。

## 发布与读取

- 每个工作主题一个目录：`.scratch/<feature-slug>/`。
- 每票一个文件：`.scratch/<feature-slug>/issues/<NN>-<slug>.md`，从 `01` 按依赖顺序编号。
- `README.md` 保存该主题的实验合同、工单索引及决策；独立规格需要时保存为 `spec.md`。
- 工单顶部记录 `What to build`、`Blocked by`、`Status`；阻塞项引用同一主题内的编号及标题。
- 用户引用路径或票号时读取对应完整工单；补充讨论追加到 `Comments`，执行证据追加到 `Execution`。
- 状态词汇见 [triage-labels.md](triage-labels.md)。`ready-for-agent` 表示已定义清楚，不表示依赖完成或工作已完成。

## 执行前沿

- 只领取所有阻塞项均已完成且通过必要准入条件的票；独立任务链可以分别推进。
- 执行时追加实际进度、验收勾选和产物证据；完成时记录 `Completion: completed`，与 triage 状态区分。
- 暂停或准入未通过时记录原因及待决事项，保留未完成验收项，下游保持阻塞。
- 旧主题工单保留原有记录方式；读取其局部合同解释历史状态。

## Wayfinding

使用 wayfinder 时，主题内 `map.md` 保存决策地图，子票仍各自保存；`Type` 记录类型，`Status` 使用 `claimed` / `resolved`。读取依赖均为 `resolved` 的未领取前沿，先领取再工作，解决后向地图追加结论指针。这是决策票流程，与实施票的 triage 状态区分。
