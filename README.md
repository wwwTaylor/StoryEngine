# StoryEngine

> 正式版本：`1.2.0` · Python：`>= 3.12`

StoryEngine 是一个“证据优先”的故事转视频生产引擎。它先冻结故事语义，再生成并验证参考素材与空间证据，随后形成可执行镜头计划，最终完成逐镜头生成、评估、选择、组装和交付。

[架构说明](docs/architecture.md) · [代码地图](docs/code-map.md) · [配置说明](docs/configuration.md) · [输出与恢复](docs/outputs-and-resume.md) · [安全说明](SECURITY.md) · [版本记录](CHANGELOG.md)

## 核心能力

- 以严格类型的 `ProjectRequest` 接收故事、镜头和交付要求。
- 以 `StoryPlan` 保存故事、实体状态、镜头顺序和约束的唯一计划真值。
- 生成不可变的 `ReferenceLibrary`，管理人物、道具状态和 2:1 场景全景。
- 支持人物/场景上传图、名称和描述；人物图严格约束身份，场景图保留可见布局并补全不可见区域。
- 提供自动或人工素材语义对齐；不匹配时返回结构化问题，不会静默创建额外故事对象。
- 从全景图确定性生成六方向探针，完成 grounding、视场角和相机求解。
- 首帧同时使用选定机位视图和与其匹配的完整场景全景，分别约束构图与全局布局。
- 只在真实素材及空间证据齐备后冻结可执行的 `RenderPlan`。
- 分开保存候选媒体、技术验证、语义评估和确定性选择结果。
- 通过原子状态、内容哈希和 StepKey 支持中断恢复。
- 支持异步视频任务恢复、FFmpeg 成片组装和最终 MP4 技术验证。
- 为每个阶段生成可人工审阅的输出和最终交付 manifest。
- 每个运行定义 YAML 同时保存故事输入、Provider 选择和执行策略。

## 快速开始

### 1. 准备环境

需要：

- Python 3.12 或更高版本
- [uv](https://docs.astral.sh/uv/)
- FFmpeg
- FFprobe
- 可用的文本、图片、视频和评估 Provider

安装锁定的正式运行依赖：

```bash
uv sync
```

检查本机媒体工具：

```bash
uv run story-engine doctor
```

`doctor` 只有在 FFmpeg 和 FFprobe 均可用时才返回成功。

### 2. 准备运行定义与凭据

[tiny_request.yaml](examples/tiny_request.yaml) 是完整运行定义，包含双场景故事、交付要求、Provider 配置和生成策略。不同 YAML 可以选择不同模型、候选数量、并发和空间失败策略，不需要额外的配置文件。

先复制一份被 Git 忽略的本地运行定义，保留公开示例中的占位地址：

```bash
cp examples/tiny_request.yaml tiny_request.local.yaml
```

当前示例通过一个 OpenAI 兼容的统一网关（例如 LiteLLM Proxy）调用 Provider。示例中的 `base_url` 是占位地址 `https://your-gateway.example.com`，请在 `tiny_request.local.yaml` 中替换为自己的网关地址。凭据只通过环境变量提供：

```bash
export STORY_ENGINE_GATEWAY_KEY="..."
```

不要把密钥值写入运行定义、Provider `options` 或 URL。YAML 中只保存环境变量名称 `api_key_env`。

### 3. 校验输入

```bash
uv run story-engine validate tiny_request.local.yaml
```

`validate` 同时校验故事请求和 `runtime` 配置结构。Provider 可达性、模型能力以及请求与能力的组合约束会在正式运行的 preflight 阶段检查。

当前示例声明的尺寸、时长和媒体能力来自配套运行配置；这些声明必须始终与实际模型能力一致。

### 4. 启动项目

```bash
uv run story-engine run tiny_request.local.yaml \
  --run-id my-production-run
```

`--run-id` 可以省略。省略时，系统会根据任务 ID、UTC 时间和请求哈希生成安全的运行 ID。

包含素材的定义可以在启动前单独执行语义对齐：

```bash
uv run story-engine align-assets definition.yaml --mode auto
```

`auto` 只自动接受唯一且置信度不低于阈值的同类型匹配；`manual` 始终要求上层界面确认。正式运行可将确认结果写入 `provided_asset_bindings`，并把 `provided_asset_policy` 设为 `require_all`。

### 5. 恢复项目

```bash
uv run story-engine resume runs/my-production-run
```

`resume` 从运行目录读取已持久化的请求和无凭据运行配置，以状态、StepKey 和产物哈希为准，不根据文件名是否存在推断完成状态。恢复时仍需提供配置声明的凭据环境变量。

## 正式工作流

```text
ProjectRequest + Provider Capabilities
    → Preflight
    → StoryPlan
    → ReferenceLibrary
    → Panorama / Probes / Grounding / Camera
    → RenderPlan
    → First Frames
    → Shot Videos
    → Evaluation / Selection
    → Assembly
    → Delivery
```

三个 Agent 是固定职责边界，不会相互调用：

- `ExecutionAgent`：故事规划、结构化决策和可执行计划编译。
- `AssetAgent`：生成图片与视频，不修改故事真值。
- `EvaluationAgent`：根据证据报告评估结果，不修改计划或伪造通过。

所有阶段由确定性的 `Workflow` 统一编排。

## 仓库结构

```text
StoryEngine/
├── story_engine/                     # 可安装的 Python 正式包
│   ├── agents/                       # 固定职责的执行、资产与评估 Agent
│   ├── domain/                       # 核心数据模型与状态真值
│   ├── media/                        # 媒体验证、尾帧与组装
│   ├── planning/                     # 故事、参考素材与渲染计划编译
│   ├── prompts/                      # 各 Provider 操作的 Prompt 渲染
│   ├── providers/                    # Provider 端口、注册与正式 Adapter
│   ├── spatial/                      # 全景、grounding 与相机求解
│   ├── workflow.py                   # 固定正式流程的统一编排
│   └── ...                           # CLI、配置、状态、存储与输出发布
├── examples/
│   └── tiny_request.yaml             # 故事与运行参数完整示例
├── docs/
│   ├── architecture.md
│   ├── code-map.md
│   ├── configuration.md
│   └── outputs-and-resume.md
├── README.md
├── CHANGELOG.md
├── pyproject.toml
└── uv.lock
```

以下内容属于本机状态，不进入正式版本：

- `.venv/`：本机 Python 运行环境。
- `runs/`：项目运行产物。
- `API_KEY.txt`：不是正式运行输入，不应发布；正式流程只从环境变量读取凭据。

## Provider 配置

正式版本支持：

| 职责 | Adapter |
| --- | --- |
| Planner | `openai_responses`、`gemini_native` |
| Image | `openai_images` |
| Video | `openai_videos` |
| Judge | `openai_responses`、`gemini_native` |

配置必须如实声明模型支持的：

- 结构化输出协议
- Prompt 字符容量
- 附件数量与媒体类型
- 图片宽高比和精确分辨率
- 视频输入方式、时长、分辨率与帧率
- Judge 可接收的媒体数量和字节容量
- 异步视频任务轮询间隔与超时

[tiny_request.yaml](examples/tiny_request.yaml) 的 `runtime.providers` 展示当前正式 Adapter、模型和能力声明，其中 `base_url` 为占位端点。创建其他故事 YAML 时应复制并根据实际 Provider 修改该区块。

StoryEngine 不会通过发送失败请求来猜测模型能力。场景空间链路必须使用 Provider 真正支持的精确 2:1 全景输出。

## 项目请求

每个正式运行定义由两部分组成。顶层故事请求至少包含：

- 任务 ID 和故事创意
- 目标镜头数量
- 视觉风格
- 输出分辨率
- 必需实体和禁止内容
- 允许的视频时长、分辨率、帧率和音频要求

同一文件的 `runtime` 区块保存输出根目录、四类 Provider 和生成策略。运行参数不进入故事 Request Hash；请求的交付参数必须落在所选 Provider 的真实能力范围内。字段定义和生成策略详见 [配置说明](docs/configuration.md)。

## 输出

```text
runs/<run_id>/
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
    │   ├── final.mp4
    │   └── manifest.json
    └── 09_audit/
```

`output/OPEN_ME.html` 是人工审阅入口。内容寻址的 `artifacts/sha256/` 是恢复流程的权威产物存储。

`output/09_audit/` 保存当前项目运行产生的正式审计记录。

## 交付状态

- `delivered`：所有选中 Requirement 均通过。
- `delivered_degraded`：已生成技术有效的成片，但至少一项语义 Requirement 为 `FAIL` 或 `UNKNOWN`。
- `process_failed`：缺少必需的技术有效素材、计划无法冻结，或最终组装验证失败。

系统会对参考素材、首帧和完整镜头视频进行评估，并单独验证确定性的计划约束。最终组装后的 `final.mp4` 接受严格技术验证，但不表示已经完成第二次整体语义评估。

## 一致性与安全边界

- Provider 凭据仅从环境变量读取；配置拒绝嵌套凭据字段，Adapter 参数在创建运行文件前完成校验，配置导出另有递归脱敏保护。
- 提供素材的 URI 不接受用户名、密码、查询参数或片段。签名资源应先下载到本地，再通过 `path` 提供。
- 媒体下载跨域重定向会移除认证头与 Cookie；含请求体的跨域重定向及 HTTPS 降级会被拒绝。
- 错误响应先脱敏再截断，配置和 YAML 校验错误不回显输入值。
- Provider transport 只重试网络失败、超时、HTTP 429 和可重试 5xx。
- HTTP 400 和认证错误不会作为逻辑尝试重复发送。
- 项目输入、配置、运行目录和提供的本地素材均受路径边界检查。
- 技术失败不能被评价文字重新标记为成功。
- 观测到的媒体错误不会反向修改 `StoryPlan` 中的计划世界真值。
- 不使用黑屏、静态帧或复制旧镜头伪造成片成功。

这些边界不负责识别故事正文或图片中的个人信息。运行目录会保存故事、Prompt、附件和模型响应；即使配置文件名含 `redacted`，也不代表整个目录适合公开。自定义运行目录会自动生成忽略文件；手动打包仍需遵循 [安全说明](SECURITY.md)。

## 当前限制

- 当前 Workflow 不交付音频；请求必须声明 `audio: false`。
- 生产输出目录必须位于当前项目目录内。
- `openai_responses` 当前用于文本规划和图片判断。
- `gemini_native` 可以根据已声明的附件能力处理图片及短视频判断。
- 超过内联容量的视频需要文件上传 Adapter，当前版本未提供该能力。

## 版本与许可

正式版本变化记录见 [CHANGELOG](CHANGELOG.md)。

Python 包版本和 CLI 版本均为 `1.2.0`。

本仓库暂不声明软件许可证，当前不对使用、修改或再分发权作出许可授权声明。后续如确定许可证，将在本节公布。
