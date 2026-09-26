# StoryEngine 架构

## 设计目标

StoryEngine 将“计划事实”和“媒体观测”分离。系统只在相应证据已经存在时冻结决定，避免下游 Provider 通过自由文本补写故事、空间或状态真值。

正式流程只有一条固定执行路径，不提供通用 DAG、多 Agent 消息总线或 Agent 自主互调。

## 运行定义边界

每次新运行由一个 YAML 完整定义。文件顶层的故事字段构造成 `ProjectRequest`，`runtime` 区块构造成独立的 `AppConfig`。运行参数不参与故事 Request Hash，也不能修改 StoryPlan、空间证据或媒体评价的所有权边界。

运行开始后，请求和无凭据配置分别持久化为 `request.json` 与 `config.redacted.json`。Resume 使用这两份权威记录，不重新读取原始 YAML。

## 数据流

```text
ProjectRequest
    ↓
StoryPlan
    ↓
ReferenceLibrary
    ↓
Spatial Evidence
    ↓
RenderPlan
    ↓
Candidate Media
    ↓
Evaluation + Selection
    ↓
Delivery
```

### ProjectRequest

`ProjectRequest` 保存用户输入、输出规格、必需实体和交付约束。请求使用严格 Schema，并通过规范化 JSON 计算稳定哈希。

### StoryPlan

`StoryPlan` 在参考素材生成前冻结，只包含不依赖未来像素的事实：

- 场景、人物和道具身份
- 初始状态与镜头边界状态变化
- 镜头顺序、动作和空间意图
- Requirement 归属和结构化顺序约束

所有结构化状态变化由统一的 WorldReducer 计算。媒体评估结果不能修改这份计划真值。

### ReferenceLibrary

`ReferenceLibrary` 是真实参考素材的不可变索引，包括：

- 人物多视图参考
- 道具状态版本
- 场景 2:1 全景图
- 源站位和新站位空间参考

只有通过技术验证和选择策略的产物才能进入 ReferenceLibrary。

### 空间证据

每个场景全景确定性投影为六个方向探针。Grounding 只定位当前场景所需的锚点和区域，相机求解器再根据真实投影、输出画幅和结构化意图计算站位、朝向与视场角。

无法满足的空间约束会形成结构化失败证据。系统根据配置选择严格失败或有边界的降级方案，不会伪造几何可达性。

### RenderPlan

`RenderPlan` 只在参考素材和空间证据齐备后冻结。它保存每个镜头真正可执行的：

- 媒体输入和身份参考
- 起始状态和目标状态
- 构图、相机和空间证据
- 首帧、持续动作和安全 Requirement
- 跨镜头顺序约束

Prompt 由针对具体操作的闭合视图生成，不直接转储完整上游对象。

首帧生成固定将选定机位的 `scene_view` 作为第一张输入，并将产生该视图的完整场景全景作为第二张输入。前者定义开场机位和可见背景，后者只补充全局房间布局与静态建筑信息。

### 媒体生成与选择

每个生成操作可以产生多个候选。候选记录、技术验证、语义评估和选择决定是分开的持久化对象：

1. 技术验证决定媒体是否可消费。
2. 语义评估根据正式 Requirement 报告 `PASS`、`FAIL` 或 `UNKNOWN`。
3. 确定性选择策略选择通过结果，或在允许时选择技术有效的降级结果。

技术失败不能被评估文字改写为成功。

### Assembly 与 Delivery

每个必需镜头至少存在一个技术有效视频后，FFmpeg 才会执行组装。最终 MP4 必须通过分辨率、时长、帧率、解码和媒体结构验证。

交付 manifest 汇总 Provider、产物哈希、Requirement 结果、镜头状态和最终交付状态。

## Agent 边界

### ExecutionAgent

负责结构化规划、修订、grounding 请求和 RenderPlan 编译。它是计划决策的唯一所有者。

### AssetAgent

负责调用图片和视频 Provider 并保存候选媒体。它不解释或修改故事。

### EvaluationAgent

负责把正式评价视图发送给 Judge Provider 并保存证据。它不选择流程分支，也不修改计划。

### Workflow

`Workflow` 按固定顺序调用三个 Agent、空间组件、验证器和组装器。Agent 之间没有直接调用关系。

## 一致性与恢复

- 运行状态通过原子写入持久化。
- 每个步骤使用输入哈希、实现版本和 Provider 指纹形成 StepKey。
- 已完成步骤只有在 StepKey 和产物哈希一致时才可复用。
- 异步视频 Provider 的任务 ID 会被保存，恢复时继续轮询原任务。
- 已选择媒体在复用前重新执行技术验证。
- 内容寻址产物存储是恢复依据，人工审阅目录不是缓存真值。
