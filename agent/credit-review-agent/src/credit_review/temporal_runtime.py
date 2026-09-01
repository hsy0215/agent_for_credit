from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any

from temporalio import activity, workflow
from temporalio.client import Client
from temporalio.worker import Worker

TASK_QUEUE = "credit-review"


@activity.defn
async def run_review_activity(payload: dict[str, Any]) -> dict[str, Any]:
    # Import here so Temporal's deterministic workflow sandbox never imports
    # model clients, file parsers, or mcp-agent.
    from .cli import build_workflow

    engine = build_workflow(payload["provider"], Path(payload["runtime"]))
    state = await engine.run(Path(payload["document"]), payload["case_id"])
    return {
        "case_id": state.case_id,
        "stage": state.stage,
        "recommendation": state.recommendation,
        "report_file": state.report_file,
    }


@activity.defn
async def apply_approval_activity(payload: dict[str, Any]) -> dict[str, Any]:
    from .cli import build_workflow

    engine = build_workflow("mock", Path(payload["runtime"]))
    state = engine.approve(
        payload["case_id"], payload["action"], payload["operator"], payload.get("comment", ""),
    )
    return {"case_id": state.case_id, "stage": state.stage}


@workflow.defn
class DurableCreditReviewWorkflow:
    def __init__(self) -> None:
        self.approval: dict[str, str] | None = None

    @workflow.run
    async def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        review = await workflow.execute_activity(
            run_review_activity,
            payload,
            start_to_close_timeout=timedelta(minutes=30),
        )
        await workflow.wait_condition(lambda: self.approval is not None)
        approval_payload = {**payload, **self.approval, "case_id": review["case_id"]}
        return await workflow.execute_activity(
            apply_approval_activity,
            approval_payload,
            start_to_close_timeout=timedelta(minutes=2),
        )

    @workflow.signal
    async def submit_approval(self, approval: dict[str, str]) -> None:
        self.approval = approval

    @workflow.query
    def approval_status(self) -> str:
        return "received" if self.approval else "waiting"


async def run_worker(address: str, namespace: str) -> None:
    client = await Client.connect(address, namespace=namespace)
    worker = Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[DurableCreditReviewWorkflow],
        activities=[run_review_activity, apply_approval_activity],
    )
    await worker.run()


async def submit_case(
    address: str, namespace: str, workflow_id: str, document: Path,
    provider: str, runtime: Path,
) -> str:
    client = await Client.connect(address, namespace=namespace)
    await client.start_workflow(
        DurableCreditReviewWorkflow.run,
        {
            "document": str(document.resolve()), "provider": provider,
            "runtime": str(runtime.resolve()), "case_id": workflow_id,
        },
        id=workflow_id,
        task_queue=TASK_QUEUE,
    )
    return workflow_id


async def signal_approval(
    address: str, namespace: str, workflow_id: str,
    action: str, operator: str, comment: str,
) -> None:
    client = await Client.connect(address, namespace=namespace)
    handle = client.get_workflow_handle(workflow_id)
    await handle.signal(DurableCreditReviewWorkflow.submit_approval, {
        "action": action, "operator": operator, "comment": comment,
    })
