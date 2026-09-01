from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


class MetricsRecorder:
    def __init__(self, path: Path, pricing_path: Path | None = None):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.pricing = {}
        if pricing_path and pricing_path.exists():
            self.pricing = yaml.safe_load(pricing_path.read_text(encoding="utf-8")).get("providers", {})

    def record(self, event_type: str, **data: Any) -> None:
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            **data,
        }
        with self.lock:
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")

    def model_call(
        self, provider: str, agent: str, input_chars: int,
        output_chars: int, latency_ms: float, success: bool,
    ) -> None:
        input_tokens = max(1, round(input_chars / 3))
        output_tokens = max(1, round(output_chars / 3))
        prices = self.pricing.get(provider, {"input": 0, "output": 0})
        cost = input_tokens * float(prices.get("input", 0)) / 1_000_000
        cost += output_tokens * float(prices.get("output", 0)) / 1_000_000
        self.record(
            "model_call", provider=provider, agent=agent,
            input_tokens_estimated=input_tokens, output_tokens_estimated=output_tokens,
            latency_ms=round(latency_ms, 2), estimated_cost_usd=round(cost, 8), success=success,
        )


def summarize_metrics(path: Path) -> dict[str, Any]:
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    model_calls = [event for event in events if event["event_type"] == "model_call"]
    stages = [event for event in events if event["event_type"] == "stage"]
    privacy = [event for event in events if event["event_type"] == "privacy"]
    approvals = [event for event in events if event["event_type"] == "human_approval"]
    return {
        "events": len(events),
        "model_calls": len(model_calls),
        "model_failure_rate": sum(not event["success"] for event in model_calls) / (len(model_calls) or 1),
        "estimated_tokens": sum(event["input_tokens_estimated"] + event["output_tokens_estimated"] for event in model_calls),
        "estimated_cost_usd": round(sum(event["estimated_cost_usd"] for event in model_calls), 6),
        "average_model_latency_ms": round(sum(event["latency_ms"] for event in model_calls) / (len(model_calls) or 1), 2),
        "average_stage_latency_ms": round(sum(event["latency_ms"] for event in stages) / (len(stages) or 1), 2),
        "pii_leakage_rate": sum(event["leakage_rate"] for event in privacy) / (len(privacy) or 1),
        "human_modification_rate": sum(event["modified"] for event in approvals) / (len(approvals) or 1),
    }

