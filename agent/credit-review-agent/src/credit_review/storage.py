from __future__ import annotations

import json
from pathlib import Path

from .models import CaseState


class FileCaseRepository:
    """Simple auditable repository; replace with PostgreSQL in production."""

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def case_dir(self, case_id: str) -> Path:
        path = self.root / case_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save(self, state: CaseState) -> Path:
        target = self.case_dir(state.case_id) / "state.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(state.model_dump_json(indent=2), encoding="utf-8")
        temporary.replace(target)
        return target

    def load(self, case_id: str) -> CaseState:
        target = self.case_dir(case_id) / "state.json"
        return CaseState.model_validate_json(target.read_text(encoding="utf-8"))

    def save_secret_mapping(self, case_id: str, mapping: dict[str, str]) -> Path:
        target = self.case_dir(case_id) / "pii_mapping.private.json"
        target.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
        target.chmod(0o600)
        return target

