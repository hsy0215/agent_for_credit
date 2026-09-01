from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer
from dotenv import load_dotenv

from .advisory import DeterministicAdvisoryAgents, MCPAdvisoryAgents
from .extraction import HeuristicExtractor, MapReduceExtractor
from .evals import HistoricalDatasetBuilder, evaluate_jsonl
from .llm import MCPAgentExtractor
from .monitoring import MetricsRecorder, summarize_metrics
from .providers import configure_provider
from .reporting import ReportRenderer
from .rules import AmountEngine, RiskEngine
from .storage import FileCaseRepository
from .workflow import CreditReviewWorkflow

app = typer.Typer(help="可审计的企业授信评审 Agent")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "output"


def build_workflow(provider: str, runtime: Path) -> CreditReviewWorkflow:
    load_dotenv(PROJECT_ROOT / ".env")
    configure_provider(provider)
    metrics = MetricsRecorder(runtime / "metrics.jsonl", PROJECT_ROOT / "config/model_pricing.yaml")
    if provider == "mock":
        extractor = HeuristicExtractor()
        advisory = DeterministicAdvisoryAgents()
    else:
        extractor = MCPAgentExtractor(provider, PROJECT_ROOT / "mcp_agent.config.yaml", metrics=metrics)
        advisory = MCPAdvisoryAgents(provider, PROJECT_ROOT / "mcp_agent.config.yaml", metrics=metrics)
    extractor = MapReduceExtractor(extractor)
    return CreditReviewWorkflow(
        repository=FileCaseRepository(runtime / "cases"),
        extractor=extractor,
        risk_engine=RiskEngine(PROJECT_ROOT / "config/risk_rules.yaml"),
        amount_engine=AmountEngine(PROJECT_ROOT / "config/amount_rules.yaml"),
        renderer=ReportRenderer(PROJECT_ROOT / "templates"),
        advisory_agents=advisory,
        metrics=metrics,
    )


@app.command()
def review(
    document: Path = typer.Argument(..., exists=True, dir_okay=False),
    provider: str = typer.Option("mock", help="mock、lm_studio、openai 或 deepseek"),
    case_id: str | None = typer.Option(None),
    runtime: Path = typer.Option(DEFAULT_OUTPUT, help="报告、状态、审计记录的输出根目录"),
) -> None:
    """运行完整评审流程，最终停在待人工审批状态。"""
    workflow = build_workflow(provider, runtime)
    state = asyncio.run(workflow.run(document, case_id))
    typer.echo(json.dumps({
        "case_id": state.case_id,
        "stage": state.stage,
        "recommendation": state.recommendation,
        "suggested_amount": state.amount.suggested_amount if state.amount else None,
        "report": state.report_file,
    }, ensure_ascii=False, indent=2))


@app.command()
def approve(
    case_id: str,
    action: str = typer.Option(..., help="approve、reject 或 supplement"),
    operator: str = typer.Option(...),
    comment: str = typer.Option(""),
    runtime: Path = typer.Option(DEFAULT_OUTPUT),
) -> None:
    """执行有身份记录的人工最终审批。"""
    workflow = build_workflow("mock", runtime)
    state = workflow.approve(case_id, action, operator, comment)
    typer.echo(f"{case_id}: {state.stage}")


@app.command("show")
def show_case(
    case_id: str,
    runtime: Path = typer.Option(DEFAULT_OUTPUT),
) -> None:
    state = FileCaseRepository(runtime / "cases").load(case_id)
    typer.echo(state.model_dump_json(indent=2))


@app.command("prepare-dataset")
def prepare_dataset(
    document_root: Path = typer.Argument(..., exists=True, file_okay=False),
    amount_workbook: Path = typer.Argument(..., exists=True, dir_okay=False),
    output: Path = typer.Option(PROJECT_ROOT / "evals/dataset.jsonl"),
) -> None:
    """对齐历史文档与授信清单，并按企业生成稳定的数据集划分。"""
    builder = HistoricalDatasetBuilder()
    records = builder.build(document_root, amount_workbook)
    builder.write(records, output)
    matched = sum(record.approved_amount is not None for record in records)
    splits = {name: sum(record.split == name for record in records) for name in ("development", "validation", "test")}
    typer.echo(json.dumps({"records": len(records), "matched": matched, "splits": splits, "output": str(output)}, ensure_ascii=False, indent=2))


@app.command("evaluate")
def evaluate(
    dataset: Path = typer.Argument(..., exists=True, dir_okay=False),
    predictions: Path = typer.Argument(..., exists=True, dir_okay=False),
) -> None:
    """计算字段精确率、无依据值率、额度 MAE 和档位准确率。"""
    typer.echo(json.dumps(evaluate_jsonl(dataset, predictions).to_dict(), ensure_ascii=False, indent=2))


@app.command("run-dataset")
def run_dataset(
    dataset: Path = typer.Argument(..., exists=True, dir_okay=False),
    output: Path = typer.Option(PROJECT_ROOT / "evals/predictions.jsonl"),
    provider: str = typer.Option("mock"),
    split: str = typer.Option("test"),
    limit: int | None = typer.Option(None),
    runtime: Path = typer.Option(PROJECT_ROOT / "runtime/evals"),
) -> None:
    """在指定隔离集合运行工作流并生成可评测预测文件。"""
    records = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = [record for record in records if record["split"] == split]
    if limit is not None:
        selected = selected[:limit]
    workflow = build_workflow(provider, runtime)

    async def execute() -> list[dict]:
        predictions = []
        for record in selected:
            state = await workflow.run(Path(record["document_path"]), case_id=record["record_id"])
            predictions.append({
                "record_id": record["record_id"],
                "facts": state.facts.model_dump(mode="json") if state.facts else None,
                "suggested_amount": state.amount.suggested_amount if state.amount else None,
                "recommendation": state.recommendation,
            })
        return predictions

    predictions = asyncio.run(execute())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in predictions) + "\n", encoding="utf-8")
    typer.echo(f"已生成 {len(predictions)} 条预测: {output}")


@app.command("temporal-worker")
def temporal_worker(
    address: str = typer.Option("localhost:7233"),
    namespace: str = typer.Option("default"),
) -> None:
    """启动可恢复的评审与人工审批 Worker。"""
    from .temporal_runtime import run_worker
    asyncio.run(run_worker(address, namespace))


@app.command("temporal-submit")
def temporal_submit(
    workflow_id: str,
    document: Path = typer.Argument(..., exists=True, dir_okay=False),
    provider: str = typer.Option("mock"),
    address: str = typer.Option("localhost:7233"),
    namespace: str = typer.Option("default"),
    runtime: Path = typer.Option(DEFAULT_OUTPUT / "temporal"),
) -> None:
    from .temporal_runtime import submit_case
    result = asyncio.run(submit_case(address, namespace, workflow_id, document, provider, runtime))
    typer.echo(f"已提交 Temporal workflow: {result}")


@app.command("temporal-approve")
def temporal_approve(
    workflow_id: str,
    action: str = typer.Option(...),
    operator: str = typer.Option(...),
    comment: str = typer.Option(""),
    address: str = typer.Option("localhost:7233"),
    namespace: str = typer.Option("default"),
) -> None:
    from .temporal_runtime import signal_approval
    asyncio.run(signal_approval(address, namespace, workflow_id, action, operator, comment))
    typer.echo(f"已发送审批信号: {workflow_id}")


@app.command("metrics")
def metrics_summary(
    metrics_file: Path = typer.Option(DEFAULT_OUTPUT / "metrics.jsonl", exists=True, dir_okay=False),
) -> None:
    """汇总模型、阶段、隐私与人工修改指标。"""
    typer.echo(json.dumps(summarize_metrics(metrics_file), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    app()
