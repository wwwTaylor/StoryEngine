# 输出与恢复

## 运行目录

每次运行都保存在 `project.output_root/<run_id>/`：

```text
<run_id>/
├── request.json
├── config.redacted.json
├── state.json
├── events.jsonl
├── artifacts/
│   └── sha256/...
└── output/
    ├── OPEN_ME.html
    ├── README.md
    ├── review_manifest.json
    ├── 00_input/
    ├── 01_story/
    ├── 02_assets/
    ├── 03_spatial/
    ├── 04_render_plan/
    ├── 05_first_frames/
    ├── 06_shot_videos/
    ├── 07_evaluation_summary/
    ├── 08_final/
    └── 09_audit/
```

## 权威数据

- `request.json`：规范化后的项目请求。
- `config.redacted.json`：不含凭据的权威运行配置，Resume 直接从这里恢复。
- `state.json`：当前项目、步骤、镜头和异步任务状态。
- `events.jsonl`：追加式运行事件。
- `artifacts/sha256/`：内容寻址的权威产物。

`output/` 是随流程逐阶段发布的人工审阅投影，不是恢复缓存。

运行目录还会创建内容为 `*` 的 `.gitignore`，覆盖自定义输出根目录。该规则只影响 Git 新文件收录，不会清除历史提交，也不会过滤手动 ZIP 打包。

`config.redacted.json` 用于合法配置的恢复，Provider 参数另有递归脱敏保护；它的文件名不是整个运行目录的公开许可。`request.json`、Prompt、原始模型响应和附件可能包含项目私有内容，分享前需要单独审阅。

## 人工审阅

`output/OPEN_ME.html` 是审阅入口。各媒体目录保存：

- 实际发送给 Provider 的 Prompt
- 输入附件和上下文摘要
- 所有候选媒体
- 技术验证结果
- 语义评价及证据
- 最终选择记录

`07_evaluation_summary/` 汇总跨阶段结果，不替代媒体目录旁的原始评价记录。

## 最终交付

`08_final/` 包含：

- `final.mp4`：按 StoryPlan 顺序组装并通过技术验证的成片。
- `manifest.json`：交付状态、镜头结果、Requirement、Provider 和产物来源。

交付状态：

- `delivered`
- `delivered_degraded`
- `process_failed`

## Resume

恢复命令：

```bash
uv run story-engine resume runs/<run_id>
```

恢复规则：

1. 读取 `request.json`、`config.redacted.json` 和 `state.json`，不根据输出文件名推断完成状态。
2. 比较步骤的 StepKey、Provider 指纹和输入哈希。
3. 验证所有权威产物的内容哈希。
4. 对复用的选中媒体重新执行技术验证。
5. 对已提交但未完成的异步视频任务继续轮询原 Provider 任务 ID。
6. 输入或实现版本变化时，只失效受影响的步骤。

恢复时仍需在环境中提供 `config.redacted.json` 所声明的凭据变量。原始 YAML 后续发生变化不会覆盖运行目录内的配置；需要不同参数时应创建新运行。

`process_failed` 是终态，不能通过 resume 继续；需要创建新的运行。

## Prompt 追踪

正式运行在 Provider 调用前保存对应 Prompt。旧运行如果缺少历史 Prompt，只会生成明确的未记录标记，不会反推或伪造当时发送的内容。
