from credit_review.monitoring import MetricsRecorder, summarize_metrics


def test_metrics_summary(tmp_path):
    path = tmp_path / "metrics.jsonl"
    recorder = MetricsRecorder(path)
    recorder.model_call("mock", "extractor", 300, 60, 10, True)
    recorder.record("stage", stage="extract", latency_ms=20, success=True)
    recorder.record("privacy", leakage_rate=0.0)
    recorder.record("human_approval", modified=True)
    result = summarize_metrics(path)
    assert result["estimated_tokens"] == 120
    assert result["pii_leakage_rate"] == 0
    assert result["human_modification_rate"] == 1
