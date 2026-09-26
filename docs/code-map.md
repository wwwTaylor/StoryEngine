# StoryEngine 代码地图

正式源码位于仓库根目录的 `story_engine/`，安装后的导入命名空间也是 `story_engine`。仓库目录可以被任意改名，不影响 Python 包名和 `story-engine` 命令。

## 入口与主流程

```text
story-engine 命令
└── story_engine/cli.py
    ├── validate → story_engine/run_spec.py
    ├── run → story_engine/bootstrap.py → run_spec.py
    └── resume → story_engine/bootstrap.py → 已持久化的请求与配置
                                              ↓
                                  story_engine/workflow.py
                                              ↓
                         Agent + Planning + Spatial + Provider + Media
```

- `cli.py`：解析 `validate`、`doctor`、`run` 和 `resume` 命令。
- `run_spec.py`：从单个 YAML 分离并验证 `ProjectRequest` 与 `runtime` 配置。
- `bootstrap.py`：装配配置、Provider、Agent、存储和 Workflow。
- `workflow.py`：按固定顺序编排正式生产流程。
- `version.py`：保存软件版本及影响恢复判定的实现版本标识。
- `__init__.py`：提供稳定的包入口和软件版本。

`validate` 和 `doctor` 保持轻量，不会启动生产 Workflow；`run` 和 `resume` 才会延迟加载运行组件。

## 业务目录

| 目录 | 职责 |
| --- | --- |
| `agents/` | Execution、Asset、Evaluation 三个固定职责 Agent |
| `domain/` | 请求、故事、参考素材、渲染、状态、评价和 manifest 等严格模型 |
| `planning/` | 故事规划、结构修订、参考素材编译、视图构造和 RenderPlan 编译 |
| `spatial/` | 全景投影、方向探针、grounding、连续性、视场角和相机求解 |
| `providers/` | Provider 协议端口、能力 Schema、注册表、传输和正式 Adapter |
| `prompts/` | 规划、grounding、图片、视频、评价等操作的 Prompt 渲染 |
| `media/` | 媒体技术验证、视频尾帧提取和 FFmpeg 组装 |

## 根目录运行模块

| 文件 | 职责 |
| --- | --- |
| `config.py` | 每次运行的 Provider、生成策略和凭据边界 |
| `run_spec.py` | 单 YAML 运行定义加载与请求/配置分离 |
| `errors.py` | 正式错误类型 |
| `ids.py` | 稳定 ID、哈希和路径安全辅助逻辑 |
| `run_state.py` | 运行步骤、恢复状态和事件持久化 |
| `workflow_records.py` | Workflow 使用的结构化运行记录 |
| `task_pool.py` | 全局异步任务并发控制 |
| `storage.py` | 内容寻址产物存储 |
| `stage_output.py` | 各阶段人工审阅输出和最终交付投影 |
| `selection.py` | 候选媒体的确定性选择策略 |

这些模块保留在包根目录，以维持现有导入路径和流程结构。目录整理不拆分 Workflow，不改变 Agent、Provider、Planning、Spatial 或 Media 的职责边界。

## Provider 子结构

```text
providers/
├── ports.py               # Provider 抽象协议
├── schemas.py             # 能力与响应结构
├── registry.py            # 配置到正式 Adapter 的映射
├── transport.py           # 网络传输与重试边界
└── adapters/
    ├── openai.py
    └── gemini.py
```

正式版本不包含 Fake Provider。Provider 能力必须由配置如实声明，并在 Workflow preflight 阶段验证。

## 数据与执行方向

```text
ProjectRequest
    → StoryPlan
    → ReferenceLibrary
    → Spatial Evidence
    → RenderPlan
    → Candidate Media
    → Evaluation + Selection
    → Assembly + Delivery
```

领域模型保存流程真值；Planning 和 Spatial 形成可执行证据；Provider 只执行边界操作；Agent 不相互调用；Workflow 是正式流程的唯一统一编排者。
