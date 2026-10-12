# Credit Review Agent

一个基于 **LangGraph 业务工作流、自定义 Agent Harness 和 MCP 工具接入**的可审计企业授信评审项目。保留 `mcp-agent` 的模型适配和结构化抽取能力，由 LangGraph 组织案件主流程，Harness 执行专业审查任务。

项目面向企业材料种类多、财务数据口径不统一、人工核验耗时以及大模型结论难以追溯的问题，构建从材料解析、专业审查、证据校验、风险计算到报告生成和人工审批的完整链路，输出带原文依据的授信报告、风险项和补件清单。

**核心原则：LLM 负责材料理解、候选事实提取和文字建议；确定性程序负责证据校验、财务计算、风险评分及状态转换，最终审批由人工完成。**

> 文档范围：架构按当前项目描述更新；文件名、CLI 命令及 Provider 配置沿用原 README，尚未依据最新版源码验证。新增编排模块按逻辑职责说明，不推定其实际文件位置。

## 项目结构

### 分层与职责

| 工程层         | 模块或组件                                       | 主要职责                                                     |
| -------------- | ------------------------------------------------ | ------------------------------------------------------------ |
| 案件编排层     | LangGraph；原流程入口 `workflow.py`              | 组织案件阶段、条件路由、Checkpoint 与人工审批中断/恢复       |
| 专业审查执行层 | 自定义 Agent Harness                             | 组织 Planner、任务依赖、Worker 执行、结果汇总、Verifier 校验与定向修复 |
| 数据与上下文层 | `document.py`、`privacy.py`                      | 文档解析、表格处理、脱敏、证据定位及模型上下文准备           |
| 模型与工具层   | `llm.py`、`extraction.py`、`mcp-agent`、MCP 工具 | 模型适配、结构化抽取、案件文档检索和计算工具调用             |
| 领域规则层     | `rules.py`、`models.py`                          | 数据结构约束、财务口径校验、风险规则与额度计算               |
| 存储与接口层   | `storage.py`、`cli.py`                           | 案件产物保存、命令行入口与人工审批操作                       |
| 质量层         | `tests/`、`evals/`                               | 单元测试、历史数据评测、证据校验与报告审计                   |

表中的现有文件名来自原 README；LangGraph、Harness 与新增任务组件的具体目录位置需要对照当前代码补齐。

### LangGraph、Harness 与 MCP 的关系

**LangGraph 是案件主流程的管理者。** 它组织材料解析、专业审查、证据校验、规则计算、报告生成及人工审批之间的状态流转。LangGraph 提供状态图、持久化和人工中断等编排能力，但业务规则仍由项目实现。[^langgraph]

**Harness 是专业审查阶段内部的执行模块，不是第二套案件状态机。** 它组织 Planner 生成的审查任务，按照依赖执行 Worker，并复用上下文组装、模型调用、工具执行、结果回传和输出校验逻辑。各 Worker 主要定义自己的专业任务、可用工具及输出要求。

**`mcp-agent` 与 MCP 不等同于业务编排层。** 本项目保留前者的模型接入与抽取能力；MCP 用于暴露和调用文档读取、字段提取、规则查询及计算能力。MCP 是工具与上下文接入协议，本身不负责审批决策或保证结果真实性。[^mcp]

这种分层是为了复用执行代码、明确职责，并不是因为 LangGraph 单独无法实现 Agent。案件阶段由 LangGraph 推进，专业任务在 Harness 内执行，不在两个层次重复维护同一套案件状态。

## 工作流

```text
案件材料
   ↓
解析文本/表格/OCR → 脱敏 → 整理可定位证据
   ↓
LangGraph：进入专业审查阶段
   ↓
Agent Harness
   ├─ Planner：根据材料和审查要求生成任务及依赖
   ├─ Scheduler：组织可并行任务
   ├─ Worker：身份 / 财务 / 信用 / 合规
   │      └─ 加载 Skill → 组装上下文 → 模型调用
   │                         ↑              ↓
   │                      结果反馈 ← MCP 工具执行
   └─ Reducer 汇总 → Verifier 校验
          ├─ 可修复：定向修复后重新校验
          ├─ 材料不足：返回补件需求
          └─ 证据冲突：返回人工复核需求
   ↓
LangGraph：依据审查结果继续推进或转入补件/人工复核
   ↓
证据门禁 → 风险规则 → 额度计算 → 报告生成与审计
   ↓
人工审批中断 → 接收审批意见 → 恢复流程
   ↓
通过 / 拒绝 / 补件
```

Planner 根据当前材料生成身份、财务、信用和合规任务；Worker 通过结构化结果及 Artifact 协作，而不是依靠自由对话传递关键事实。项目已沉淀 **13 个可复用 Skill**，描述任务步骤、输入输出和质量要求；具体检索、计算及校验由工具代码执行。

专业审查中的 Verifier 检查任务输出结构、引用、数字及事实一致性；后续证据门禁判断汇总结果能否进入风险与额度计算。材料缺失不通过反复调用模型“补全”，而是形成补件项。

## 文档解析、财务口径与证据检索

### 文档解析与结构化

支持 DOCX、PDF、TXT、原生表格及 OCR 解析，并保留跨页表格标识。长材料按字符预算分块处理，通过 Map-Reduce 抽取并汇总候选事实，使用 Pydantic 约束输出结构，再由确定性合并器整理数据。

财务快照按照以下维度隔离：

```text
企业主体 × 合并/单体口径 × 报告期间
```

禁止直接把某年前三季度与另一年全年数据作为同期数据比较。字段结构合法不等于财务含义正确，仍需校验主体、期间、单位及原文支持。

### 原文证据与案件内检索

根据文档哈希、段落/表格位置及文本内容生成稳定 `evidence_id`，将抽取字段和关键结论绑定到可定位的原文片段。

Worker 通过案件文档检索工具获取与任务相关的证据，再将证据加入模型上下文：

```text
审查任务 → 案件材料检索 → 原文片段与位置
        → 模型分析/工具计算 → 带证据的结构化结果
```

当前项目描述的是**案件范围内的证据检索与引用校验**，不将选股项目的 LlamaIndex、向量数据库、BM25/RRF 或 Reranker 算作本项目已实现的能力。

检索增强的关键是先获取相关外部证据，再用于模型分析或生成；解析分块本身不等于 RAG，也不要求一定使用向量数据库。实际检索算法应以案件检索工具的实现为准。[^retrieval]

### 脱敏与数据边界

姓名、手机号、地址等敏感字段先在本地转换为稳定 Token，脱敏映射独立保存。发送给外部模型的是分块后的 `masked_text`，不直接发送原始文档。

脱敏不等于已经完成全部安全措施。原件、映射文件、日志和中间产物仍需访问控制；生产部署前还需要密钥管理、日志脱敏及供应商数据处理评估。

## 如何限制大模型幻觉

| 控制环节       | 实现方式                                                     |
| -------------- | ------------------------------------------------------------ |
| 输入与数据口径 | 完整解析材料并脱敏；财务数据按主体、合并/单体、报告期间隔离  |
| 结构化输出     | 使用 Pydantic 约束字段；不把格式正确视为事实正确             |
| 证据与引用     | 校验标量、财务快照及报告正文的 Evidence ID；无效引用不进入正式报告，必要时回退为确定性文本 |
| 风险与额度     | 风险实际值、阈值、得分、等级、系数及建议额度由代码计算，不由模型自由生成 |
| 报告与复核     | 展示冲突、规则依据、计算过程及原文位置；Reviewer 检查数字矛盾、无依据陈述与隐私泄漏，最终由人工审批 |

没有充分原文支持的风险判断标记为待核实，并生成具体补件项。Verifier 与 Reviewer 的检查用于发现和阻断错误，不等于对所有可能错误提供绝对保证。

## 状态管理与流程恢复

当前主线使用 **LangGraph State、Checkpoint 与 `interrupt`** 管理案件进度和人工审批。Harness 记录专业任务结果及中间 Artifact，不另行接管案件生命周期。

| 数据                     | 用途                                             |
| ------------------------ | ------------------------------------------------ |
| Graph State / Checkpoint | 保存业务流程状态及可恢复的执行进度               |
| 案件 `state.json`        | 保存结构化事实、风险、额度、审批结果和审计事件   |
| Artifact                 | 保存专业任务中间结果及其引用，便于检查和后续处理 |

**案件 JSON、Artifact 和 LangGraph Checkpoint 不是同一种存储。** 原 README 未给出当前 Checkpointer 的具体后端及落盘位置，部署说明需要与实际配置对应，不能仅凭存在 `state.json` 就认为已经支持跨进程恢复。

人工审批时暂停流程，接收操作人、动作和意见后继续执行。跨进程恢复需要持久化 Checkpointer 和稳定的执行标识；仅使用内存保存不能保留进程退出后的状态。[^checkpoints]

恢复粒度受节点和任务边界影响。`interrupt` 恢复时可能重新执行所在节点前面的代码，因此不能承诺所有已完成的 LLM/Tool 调用都绝不重复。涉及外部写入的操作需要幂等处理；复用中间产物也需要对应的完成记录和读取逻辑。[^interrupts]

## 输出产物

沿用原 README 的默认输出约定：

```text
output/
├── cases/
│   └── <case_id>/
│       ├── credit_report.md          # 授信辅助报告
│       ├── state.json                # 事实、风险、额度、审批与审计记录
│       └── pii_mapping.private.json  # 脱敏映射，权限 0600
└── metrics.jsonl                     # 运行指标
```

通过 `--runtime <目录>` 覆盖输出根目录。Graph Checkpoint 与 Harness Artifact 的实际存储位置以当前实现为准，不假定它们都保存在 `state.json` 中。

## 初始化与离线运行

> 以下命令沿用现有 CLI。`<document.docx>`、`<报告目录>`、`<user>@<server>` 等均为占位符，执行前替换为实际路径或账号。新版 CLI 与 LangGraph 的接线尚未依据源码核验。

### 初始化（uv）

```bash
cd agent/credit-review-agent
uv sync --frozen
```

项目尚未生成 lock 文件时，首次执行：

```bash
uv sync
```

### 离线基线

`mock` 是不依赖模型或 API Key 的确定性抽取基线，用于检查文档处理、规则计算、报告及审批链路。它不代替真实模型下的 Planner、工具选择和生成质量评测。

```bash
uv run credit-review review \
  '../mcp-agent-main/评审报告-无项目编码/评审报告-无项目编码/300万以下项目/深圳xx生物科技有限公司评审报告.docx' \
  --provider mock
```

查看案件及提交人工审批：

```bash
uv run credit-review show CR-xxxxxxxxxx

uv run credit-review approve CR-xxxxxxxxxx \
  --action approve \
  --operator interviewer-demo \
  --comment '材料核验通过'
```

案件编号替换为运行结果中的真实编号。`approve` 命令能更新案件状态，不代表它已经接通 LangGraph 的恢复调用；二者的连接应由当前 CLI 实现确认。

## 模型接入

模型配置沿用 `mcp_agent.config.yaml`、Provider 适配器及项目原有密钥配置。`mcp-agent` 用于模型调用和结构化抽取，不替代 LangGraph 的案件级编排。

原版 Extractor、Risk Analyst/Writer、Independent Reviewer 分别对应抽取、分析/撰写、独立审阅功能；当前整体流程以上面的工作流为准，不再用某个 Provider 的固定调用顺序代表整套系统。

### 本地 LLM

沿用原项目的 LM Studio 连接方式：本机服务监听 `127.0.0.1:1234`，通过 SSH 将服务器 `3456` 端口转发至本机模型服务。

```text
Agent 访问地址：http://localhost:3456/v1
转发目标：本机 127.0.0.1:1234
```

在 `mcp_agent.config.yaml` 中设置实际加载的模型名称，然后运行：

```bash
uv run credit-review review <document.docx> --provider lm_studio
```

原 README 也列出了 Ollama 的接入方式：调整 OpenAI-compatible `base_url`，或扩展 `MCPAgentExtractor` 增加独立 Provider；这不等于当前已经提供了独立的 `--provider ollama` 命令。

```text
Ollama 的原文配置地址：http://localhost:11434/v1
```

### OpenAI

沿用项目的私有配置文件方式，创建不提交到 Git 的 `mcp_agent.secrets.yaml`：

```yaml
openai:
  api_key: your-key
```

```bash
uv run credit-review review <document.docx> --provider openai
```

这里的 YAML 是项目适配层读取的配置，不是 OpenAI 通用配置文件。API Key 不应进入仓库、报告或客户端代码；生产环境应由服务端环境变量或密钥管理服务提供。[^openai]

### DeepSeek Anthropic-compatible API

项目原实现不读取 `~/.claude/settings.json`。可在当前 shell 设置：

```bash
export DEEPSEEK_API_KEY='your-key'
uv run credit-review review <document.docx> --provider deepseek
```

也可以按照原项目约定复制 `.env.example` 为 `.env` 并填写 Key；确认 `.env` 保持在 `.gitignore` 中。

原 README 记录的 Provider 配置如下；端点与模型名称也见 DeepSeek 官方接口文档。实际启用值仍以当前项目配置为准。[^deepseek]

```yaml
base_url: https://api.deepseek.com/anthropic
model: deepseek-v4-pro
```

原实现针对 Thinking 与强制 `tool_choice` 的结构化调用冲突，在 `mcp_agent.config.yaml` 中使用 `structured_output_thinking: disabled`，仅对相关结构化调用关闭 Thinking，并为 Writer/Reviewer 提供确定性降级。

该项属于项目的兼容处理记录，不应推定所有模型版本均存在同样限制；Provider 或依赖升级后，需要回归验证结构化输出与工具调用行为。

### SSH 反向隧道与 HTTP 代理

服务器需要借助本机 `7890` HTTP 代理访问外部模型时，从本机建立隧道：

```bash
ssh -NT \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -R 127.0.0.1:1234:127.0.0.1:7890 \
  -R 127.0.0.1:3456:127.0.0.1:1234 \
  <user>@<server>
```

在服务器配置：

```dotenv
HTTP_PROXY=http://127.0.0.1:1234
HTTPS_PROXY=http://127.0.0.1:1234
```

以上沿用原项目的端口约定：服务器 `1234` 是 HTTP 代理入口，`3456` 是本地模型转发入口。使用 DeepSeek 时仍选择 `--provider deepseek`，保留远端 API 的 `base_url`，不要把代理端口填成模型地址。

发送外部请求前，仍需检查输入脱敏、日志内容和数据授权范围。

## 测试与历史数据评测

### 单元测试

```bash
uv run pytest
```

### 历史数据集

```bash
uv run credit-review prepare-dataset <报告目录> <项目清单及授信金额.xlsx>
uv run credit-review run-dataset evals/dataset.jsonl --split test --provider mock
uv run credit-review evaluate evals/dataset.jsonl evals/predictions.jsonl
```

原有历史数据按企业稳定划分，共 50 份文档：development 28、validation 9、test 13。

`gold` 中为 `null` 的字段是待人工标注项，不能将原报告自动抽取结果当作人工金标准。使用 `mock` 得到的是确定性基线结果，真实模型结果需要单独运行、记录 Provider 与配置版本。

### 已记录的离线结果

| 指标                   | 现有记录 | 解释边界                                               |
| ---------------------- | -------: | ------------------------------------------------------ |
| 必填字段覆盖率         |   98.67% | 衡量字段覆盖，不是字段正确率或审批正确率               |
| 结论证据支持检测通过率 |     100% | 表示相应测试集上的检测结果，不是对所有结论正确性的保证 |
| 报告篡改测试检出率     |     100% | 表示既定篡改测试中的检出结果，不代表覆盖全部篡改方式   |

上述结果来自现有项目评测描述。本 README 尚未给出对应运行版本、样本分母和逐项判定规则，不能默认这些百分比都以 50 份历史文档为分母。

**当前不据此声明“审批正确率达到 98%”。** 该结论需要独立案件级人工标注，比较人工修改前的 Agent 建议，并明确补件、转人工案件是否纳入分母。

已记录的财务口径案例：在包含 688 个证据片段的报告中识别跨期间错配，将错误同比 `-32.10%` 修正为同期口径下的 `-1.64%`。这是个案验证，不应解释为全数据集准确率。

## 监控

```bash
uv run credit-review metrics --metrics-file output/metrics.jsonl
```

沿用原项目的监控范围：阶段延迟、模型失败率、Token 估算、配置化成本估算、PII 残留率和人工修改率。

字符推算的 Token 与成本用于粗略观测，不能当作实际账单；精确统计需要接入对应 Provider 返回的 `usage` 和计费记录。专业审查任务覆盖率、检索质量、恢复成功率等指标应单独验证，不由现有字段覆盖率替代。

## Temporal：历史适配说明

早期版本包含 Temporal Worker、案件提交及审批 Signal 适配。**当前架构以 LangGraph 管理案件主流程，不将 Temporal 与 LangGraph 同时描述为同一层的流程管理者。**

下面保留原 README 的历史命令，供对应旧实现使用；它们不是当前 LangGraph 主线的默认启动方式，其是否仍在当前分支可用需检查 CLI：

```bash
uv run credit-review temporal-worker
uv run credit-review temporal-submit CASE-001 <document.docx> --provider mock
uv run credit-review temporal-approve CASE-001 --action approve --operator reviewer
```

历史实现由 Temporal 保存工作流事件并等待审批 Signal，业务状态另存于案件仓库。这里不声明已实现 Temporal 与 LangGraph 的嵌套部署，也不推定两者使用相同的恢复粒度。

## 当前功能与下一阶段

当前文档覆盖：多格式材料解析与脱敏、财务口径隔离、稳定证据引用、LangGraph 案件编排、Harness 专业任务执行、MCP 工具调用、13 个 Skill、确定性风险与额度计算、报告审计、人工审批、离线基线及历史数据评测。

生产化下一步：

- 完成人工字段和案件结论标注，补充评测样本、分母、配置版本，以及补件/转人工的统计口径。
- 将文件案件库升级为 PostgreSQL，以对象存储保存原件；明确 Checkpointer 后端与 Artifact 存储，验证进程中断恢复和重复写入处理。
- 增加 FastAPI、审批 UI、RBAC、JWT、KMS 及日志脱敏机制，完善身份认证、权限隔离和敏感数据保护。
- 接入 OpenTelemetry、Provider 原始 `usage` 与账单数据，完善模型调用、阶段延迟、失败恢复及成本监控。

## 技术说明参考

以下官方文档仅用于核对通用技术边界，不作为本项目代码实现或评测结果的证明。

[^langgraph]: LangGraph overview：`https://docs.langchain.com/oss/python/langgraph/overview`
[^mcp]: Model Context Protocol architecture：`https://modelcontextprotocol.io/docs/learn/architecture`
[^retrieval]: LangChain retrieval：`https://docs.langchain.com/oss/python/langchain/retrieval`
[^checkpoints]: LangGraph checkpointers：`https://docs.langchain.com/oss/python/langgraph/checkpointers`
[^interrupts]: LangGraph interrupts：`https://docs.langchain.com/oss/python/langgraph/interrupts`
[^openai]: OpenAI API authentication：`https://developers.openai.com/api/reference/overview`
[^deepseek]: DeepSeek API：`https://api-docs.deepseek.com/`
