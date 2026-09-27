# StoryEngine

StoryEngine 是一个“证据优先”的故事转视频引擎，提供故事规划、参考素材生成、空间与机位求解、逐镜头生成和评估，以及最终视频组装。

版本：`1.2.0` · Python：`>= 3.12`

## 项目结构

```text
StoryEngine/
├── story_engine/              # 核心 Python 源码
├── examples/
│   └── tiny_request.yaml      # 故事与 Provider 配置示例
├── README.md
├── pyproject.toml             # 安装配置、依赖和命令入口
├── uv.lock                    # 锁定的依赖版本
└── .gitignore
```

## 安装

需要 Python 3.12 或更高版本、[uv](https://docs.astral.sh/uv/)、FFmpeg 和 FFprobe，以及可用的文本、图片、视频和评估 Provider。

```bash
git clone https://github.com/wwwTaylor/StoryEngine.git
cd StoryEngine
uv sync
uv run story-engine doctor
```

`doctor` 检查 FFmpeg 和 FFprobe 是否可用。

## 配置与运行

复制示例作为本地配置：

```bash
cp examples/tiny_request.yaml tiny_request.local.yaml
```

Windows PowerShell 可使用 `Copy-Item examples/tiny_request.yaml tiny_request.local.yaml`。

编辑 `tiny_request.local.yaml` 中的故事、Provider 模型和能力参数，将占位地址 `https://your-gateway.example.com` 替换为自己的网关。示例使用 OpenAI 兼容接口及 Gemini 原生接口；模型名称、尺寸和时长声明需与实际服务相符。

凭据通过环境变量提供。Bash 使用：

```bash
export STORY_ENGINE_GATEWAY_KEY="<your-api-key>"
```

Windows PowerShell 使用：

```powershell
$env:STORY_ENGINE_GATEWAY_KEY = "<your-api-key>"
```

校验配置并运行：

```bash
uv run story-engine validate tiny_request.local.yaml
uv run story-engine run tiny_request.local.yaml --run-id my-run
```

输入校验不访问真实 Provider，也不验证其可达性。正式运行需要有效凭据和支持所选能力的服务。

中断后可恢复已有运行：

```bash
uv run story-engine resume runs/my-run
```

默认输出位于 `runs/`。恢复时仍需提供对应的凭据环境变量。

## 数据与凭据

不要把真实密钥写入 YAML、Provider options、故事正文或 URL。配置中仅保存环境变量名称 `api_key_env`；本地配置使用被 Git 忽略的 `*.local.yaml`、`*.local.yml` 或 `*.local.json`。

程序会向你配置的 Provider 发送生成所需的故事、提示词和素材。运行目录包含输入、Prompt、附件及模型响应，应作为私有资料保存；配置脱敏不代表整个运行目录可以公开，手动打包也不会应用 Git 忽略规则。

提供素材的 URI 不接受用户名、密码、查询参数或片段。签名素材应先下载到本地，再通过 `path` 引用。代码中的凭据保护不会自动识别故事或媒体中的个人信息。

## 许可

本仓库暂不声明软件许可证，当前不对使用、修改或再分发权作出许可授权声明。后续如确定许可证，将在本节公布。
