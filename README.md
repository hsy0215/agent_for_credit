# Credit Review Agent

一个基于 `mcp-agent` 的、可审计的企业授信评审项目。它不是让 LLM 直接决定贷款，而是把 LLM 放在擅长的文档理解环节，把计算、规则、状态转换和最终审批留给确定性程序与人。

## 为什么这是完整的 Agent 项目

普通后端主要接收结构化请求并执行预定义逻辑；Agent 工程还要处理非结构化上下文、模型不确定性、工具调用、证据约束与评测。项目分层如下：

| Agent 工程层 | 本项目模块 | 对应传统后端概念 |
|---|---|---|
| 数据与上下文层 | `document.py`, `privacy.py` | DAO/数据链路 |
| 模型能力层 | `llm.py`, `extraction.py` | 外部服务适配器 |
| 领域规则层 | `rules.py`, `models.py` | Domain/Service |
| 编排与记忆层 | `workflow.py`, `storage.py` | Application Service/状态机 |
| 接口层 | `cli.py` | Controller |
| 质量层 | `tests/`，后续 `evals/` | 测试、监控与审计 |

核心原则：模型只产生“候选事实和文字建议”，确定性代码负责验证、评分、额度计算和状态转换。

## 工作流

```text
解析 → 脱敏 → 结构化抽取 → 证据校验 → 风险规则
     → 额度计算 → 报告生成 → 待人工审批 → 通过/拒绝/补件
```

默认产物写入项目内 `output/`：

- 报告：`output/cases/<case_id>/credit_report.md`
- 结构化事实、风险、额度、审核结果和审计事件：`output/cases/<case_id>/state.json`
- 脱敏映射：`output/cases/<case_id>/pii_mapping.private.json`（权限 `0600`）
- 运行指标：`output/metrics.jsonl`

可通过 `--runtime <目录>` 覆盖输出根目录。原文段落有稳定的 `evidence_id`；LLM 抽取字段必须引用真实证据。生产环境必须进一步使用 KMS 加密和独立权限服务。

## 初始化（uv）

```bash
cd agent/credit-review-agent
uv sync --frozen
```

如果项目还没有 lock 文件，第一次执行：

```bash
uv sync
```

## 离线运行

`mock` 是确定性抽取基线，不需要模型或 API Key，适合验证整条链路：

```bash
uv run credit-review review \
  '../mcp-agent-main/评审报告-无项目编码/评审报告-无项目编码/300万以下项目/深圳裕策生物科技有限公司评审报告.docx' \
  --provider mock
```

查看案件：

```bash
uv run credit-review show CR-xxxxxxxxxx
```

人工审批：

```bash
uv run credit-review approve CR-xxxxxxxxxx \
  --action approve --operator interviewer-demo --comment '材料核验通过'
```

## 使用本地 LLM

1. 在本机启动 LM Studio，使其监听 `127.0.0.1:1234`。
2. 通过 SSH 将服务器的 `3456` 反向映射到本机 LM Studio 的 `1234`。
3. Agent 使用 `http://localhost:3456/v1` 访问 LM Studio。
4. 在 `mcp_agent.config.yaml` 修改模型名称。
5. 执行：

```bash
uv run credit-review review <document.docx> --provider lm_studio
```

也可以将 `openai.base_url` 指向 Ollama 的 `http://localhost:11434/v1`，或者扩展 `MCPAgentExtractor` 增加独立的 Ollama provider。

## 使用 OpenAI

创建不提交到 Git 的 `mcp_agent.secrets.yaml`：

```yaml
openai:
  api_key: your-key
```

然后运行：

```bash
uv run credit-review review <document.docx> --provider openai
```

## 使用 DeepSeek Anthropic-compatible API

项目不会读取 `~/.claude/settings.json`。推荐在当前 shell 设置：

```bash
export DEEPSEEK_API_KEY='your-key'
uv run credit-review review <document.docx> --provider deepseek
```

也可以复制 `.env.example` 为 `.env` 并填写 Key；`.env` 已加入 `.gitignore`。项目固定使用：

```text
base_url: https://api.deepseek.com/anthropic
model: deepseek-v4-pro
```

如果服务器需要借助本机 `7890` HTTP 代理访问 DeepSeek，请从本机建立反向隧道：

```bash
ssh -NT \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -R 127.0.0.1:1234:127.0.0.1:7890 \
  -R 127.0.0.1:3456:127.0.0.1:1234 \
  <user>@<server>
```

然后在服务器的 `.env` 中配置代理（项目已提供该本地配置）：

```dotenv
HTTP_PROXY=http://127.0.0.1:1234
HTTPS_PROXY=http://127.0.0.1:1234
```

此时仍需使用 `--provider deepseek`。`1234` 是 HTTP 代理入口，不是模型 API，
所以 DeepSeek 的 `base_url` 必须继续保持 `https://api.deepseek.com/anthropic`，
不能改为 `http://localhost:1234/v1`。

DeepSeek Provider 会依次调用结构化 Extractor、Risk Analyst/Writer 和独立 Reviewer。DOCX/PDF 原文先在本地解析和脱敏，发送给 API 的是分块后的 `masked_text`。

DeepSeek V4 默认开启 Thinking，而 `mcp-agent.generate_structured()` 通过固定
`tool_choice` 强制模型返回 Pydantic 结构。项目在 `mcp_agent.config.yaml` 中将
`structured_output_thinking` 设为 `disabled`，只关闭结构化调用的 Thinking，
避免二者冲突；普通生成调用不受影响。

发送到外部模型的是 `masked_text`，不是原文。但正式使用前仍需完成数据合规、日志脱敏和供应商评估。

## 测试

```bash
uv run pytest
```

## 当前是否使用 RAG

当前版本**没有使用 RAG**。单篇长报告采用 Map-Reduce：完整解析文档后按字符预算分批抽取，再由确定性代码合并候选事实。Map-Reduce 解决的是单篇文档超出上下文窗口的问题，不是检索。

很多篇报告也不必一律先做 RAG：若每篇都要独立审批，应逐篇运行同一工作流；只有需要查询授信政策、历史批复、企业历次材料或相似案例时，才应把这些外部语料切分、建立向量/关键词混合索引，并把检索片段连同来源 ID 交给 Agent。检索结果只能作为证据候选，不能直接替代当前申请材料或确定性规则。

## 如何限制大模型幻觉

- 文档先完整解析、脱敏并生成稳定 Evidence ID。
- 结构化抽取使用 Pydantic Schema；财务快照按主体、合并/单体口径和期间分开，禁止把 2022 年 1–9 月与 2021 全年当同比。
- 所有标量、财务快照以及叙述正文中的 Evidence ID 都会校验；引用不存在时拒绝该叙述并回退到确定性文本。
- 风险指标、规则阈值、得分、风险等级、额度系数和建议额度全部由代码计算，不由 LLM 决定。
- 报告列出冲突双方、规则实际值与阈值、计算式、原文摘录及位置；没有原始证据的派生风险标为待核实并生成具体补件项。
- 独立 Reviewer 再检查数字矛盾、无依据陈述与隐私泄漏；最终必须人工审批。

## 历史数据集与评测

```bash
uv run credit-review prepare-dataset <报告目录> <项目清单及授信金额.xlsx>
uv run credit-review run-dataset evals/dataset.jsonl --split test --provider mock
uv run credit-review evaluate evals/dataset.jsonl evals/predictions.jsonl
```

数据按企业稳定划分，当前真实数据得到 development 28、validation 9、test 13。`gold` 中为 `null` 的字段是待人工标注项，不能把原报告自动抽取结果冒充金标准。

## Temporal 持久运行

先启动 Temporal Server，然后分别启动 Worker 和提交案件：

```bash
uv run credit-review temporal-worker
uv run credit-review temporal-submit CASE-001 <document.docx> --provider mock
uv run credit-review temporal-approve CASE-001 --action approve --operator reviewer
```

Temporal 保存工作流事件并在等待审批 Signal 时恢复；业务案件状态仍保存到案件仓库。

## 监控

```bash
uv run credit-review metrics --metrics-file output/metrics.jsonl
```

监控包含阶段延迟、模型失败率、Token 估算、配置化成本估算、PII 残留率和人工修改率。精确 Token/账单应在生产环境接入具体 Provider 返回的 usage，而不是依赖字符估算。



## 当前 MVP 与下一阶段

当前已具备：DOCX/PDF/TXT 解析、OCR、跨页表格标识、Map-Reduce 抽取、证据 ID、增强 PII Mask、mcp-agent 结构化抽取、Risk Analyst/Writer/Reviewer、离线基线、规则引擎、额度引擎、案件状态、报告、人工审批、Temporal 适配、历史额度 Eval、监控和单元测试。

生产化下一步：

- 完成人工字段标注；当前框架已准备好，但不能自动制造可信 gold label。
- 将文件状态库升级为 PostgreSQL，对象存储保存原件。
- 增加 FastAPI、审批 UI、RBAC、JWT、KMS、OpenTelemetry 和模型成本监控。
- 将字符 Token/成本估算替换为 Provider 原始 usage 和账单数据。

