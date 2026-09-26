# StoryEngine 运行定义

StoryEngine 使用单个 YAML 描述一次可重复运行。文件顶层保存故事请求，`runtime` 保存本次运行使用的 Provider、能力声明和执行策略，不再使用独立的 `--config` 文件。

## 文件结构

```yaml
task_id: example
idea: A structured story idea.
shot_target: 4
visual_style: Cinematic realism.

resolution:
  width: 1280
  height: 720

generation_requirements:
  output_language: English
  required_entities: []
  forbidden_content: []
  asset_alignment_mode: auto
  provided_asset_policy: optional
  delivery_requirements:
    allowed_video_duration_seconds: [4, 6, 8]
    video_resolution:
      width: 1280
      height: 720
    video_fps: 24
    audio: false

provided_assets:
  - asset_id: asset_character_01
    kind: character
    name: Customer-facing display name
    description: Visual identity and clothing details
    path: assets/character.png

provided_asset_bindings:
  - asset_id: asset_character_01
    kind: character
    canonical_alias: provided_character_01
    semantic_role: The courier in the idea
    idea_mentions: [courier]

runtime:
  project:
    output_root: runs
  providers:
    planner: {}
    image: {}
    video: {}
    judge: {}
  generation: {}
```

完整实例见 [tiny_request.yaml](../examples/tiny_request.yaml)。

人物和场景素材支持 PNG、JPEG、WebP，单个本地文件不超过 50 MiB。`asset_alignment_mode` 为 `auto` 或 `manual`；它表达上层交互策略。`provided_asset_policy: require_all` 会拒绝任何未写入 `provided_asset_bindings` 的素材。绑定的 `canonical_alias` 是 Planner 必须原样使用的 ASCII 标识，不依赖上传显示名称与 idea 的字面一致性。

普通 `scene` 图片会被用来生成并验证完整 2:1 场景全景；已是 2:1 等距柱状投影的素材可声明为 `panorama`。人物素材会作为权威身份输入，生成规范的正、侧、背参考图。

## 请求与运行参数边界

加载时，StoryEngine 先从 YAML 中取出 `runtime`，再用其余字段构造严格的 `ProjectRequest`：

- 故事内容、镜头数、交付要求和提供素材参与 Request Hash。
- Provider、重试、候选数、并发和输出目录不参与 Request Hash。
- 运行配置单独持久化到 `config.redacted.json`，用于恢复和步骤失效判断。

因此，仅调整运行策略不会把相同故事误认为新的故事请求。

## Provider

四个 Provider 职责使用统一结构：

```yaml
adapter: openai_responses
model: MODEL_NAME
base_url: https://provider.example/v1
api_key_env: PROVIDER_API_KEY
timeout_seconds: 120
options: {}
```

正式 Adapter：

| Adapter | 可承担职责 |
| --- | --- |
| `openai_responses` | Planner、Judge |
| `openai_images` | Image |
| `openai_videos` | Video |
| `gemini_native` | Planner、Judge |

不同职责即使使用相同端点或凭据，也必须分别声明模型和能力。

## 凭据

`api_key_env` 只保存环境变量名称：

```yaml
api_key_env: STORY_ENGINE_GATEWAY_KEY
```

运行前在当前终端设置真实值：

```bash
export STORY_ENGINE_GATEWAY_KEY="..."
```

以下位置禁止出现凭据值：

- 运行定义 YAML
- Provider `options`
- `base_url` 查询参数、用户名或密码
- 项目请求和提供素材 URI
- 状态、事件、manifest、StepKey 和产物元数据

Provider `options` 中的凭据字段按大小写和分隔符归一化检查，并递归检查嵌套对象、列表以及恢复配置中的编码参数。`password`、`access_key`、`refresh_token`、`apiToken` 等写法也会被拒绝。所有 Adapter 参数会在创建运行文件前完成本地结构校验，`validate` 不需要真实 API Key，也不会发起 Provider 请求。

提供素材的 `uri` 禁止任何查询参数和片段，以免访问令牌或下载签名进入请求记录。需要签名访问的素材，请先下载为本地文件，再使用 `path`。普通无查询参数的 HTTP(S) URL 仍受支持。

建议将个人运行定义命名为 `*.local.yaml`、`*.local.yml` 或 `*.local.json`；这些文件已加入 Git 忽略规则。公开示例应始终保留占位网关地址与环境变量名。

## 能力声明

StoryEngine 在生产调用前验证能力。`runtime.providers.*.options` 必须如实声明模型支持的：

- 结构化输出协议
- Prompt 字符容量
- 附件数量与媒体类型
- 图片宽高比和精确分辨率
- 视频输入方式、时长、分辨率与帧率
- 异步任务轮询间隔和超时
- Judge 可接收的媒体数量与字节容量

请求分辨率必须同时被 Image 和 Video Provider 支持；每个镜头的可选时长是请求允许时长和 Video Provider 支持时长的交集。场景空间链路还需要 Provider 真正支持的精确 2:1 全景输出。

StoryEngine 不会通过发送失败生产请求来猜测能力，也不能为了通过 preflight 而声明模型实际不支持的尺寸。

## 生成策略

每个 YAML 可以独立设置：

```yaml
runtime:
  generation:
    attempts: 2
    story_planning_attempts: 3
    grounding_attempts: 2
    candidates_per_attempt: 1
    provider_retries: 1
    max_concurrency: 2
    spatial_failure_policy: strict
    spatial_repair_attempts: 2
    novel_station_attempts: 2
    max_hfov_degrees: 110
    safe_fallback_hfov_min: 60
    safe_fallback_hfov_max: 90
```

- `attempts`：每项媒体操作的逻辑尝试上限。
- `story_planning_attempts`：故事草案生成和结构修订上限。
- `grounding_attempts`：场景 grounding 的逻辑尝试上限。
- `candidates_per_attempt`：每次逻辑尝试生成的候选数量。
- `provider_retries`：网络故障、超时、HTTP 429 和可重试 5xx 的传输重试次数。
- `max_concurrency`：四类 Provider 共享的全局并发上限。
- `spatial_failure_policy`：空间无解时使用 `strict` 或 `degrade`。
- `spatial_repair_attempts`：已有站位的空间修复尝试次数。
- `novel_station_attempts`：新站位尝试次数。
- `max_hfov_degrees`：相机允许的最大水平视场角。
- `safe_fallback_hfov_min`、`safe_fallback_hfov_max`：降级策略的安全视场角范围。

Transport retry 和逻辑尝试是两类独立机制，不会混用。媒体操作找到可接受候选后会提前停止，不一定耗尽全部上限。

## 路径与恢复

`runtime.project.output_root` 可以是相对路径或绝对路径，但解析后必须位于 StoryEngine 项目目录内。运行 ID 必须是安全的单一路径段。

项目请求引用的本地素材路径相对于运行定义 YAML 解析，并接受可读性与路径边界验证。

新运行从 YAML 读取 `runtime`；Resume 从运行目录中的 `config.redacted.json` 恢复同一份无凭据配置。修改原始 YAML 不会覆盖已经开始运行的配置，如需使用新参数应创建新的运行。
