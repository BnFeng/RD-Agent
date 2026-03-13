from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rdagent.app.prediction_market.models import MarketRecord, MonitorState


def resolve_pmm_root(pmm_root: str | Path) -> Path:
    root_path = Path(pmm_root).expanduser().resolve()
    if not root_path.exists():
        raise FileNotFoundError(f"未找到 prediction-market-monitor 目录：{root_path}")
    return root_path


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"未找到状态文件：{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_monitor_state(pmm_root: str | Path) -> MonitorState:
    root_path = resolve_pmm_root(pmm_root)
    state_dir = root_path / "data" / "state"

    markets_payload = _read_json(state_dir / "markets.json")
    insights_payload = _read_json(state_dir / "insights.json")
    summary_payload = _read_json(state_dir / "summary.json")

    markets = [MarketRecord.model_validate(item) for item in markets_payload.get("items", [])]
    markets_by_uid = {item.market_uid: item for item in markets}

    return MonitorState(
        generated_at=markets_payload.get("generated_at") or insights_payload.get("generated_at"),
        markets=markets,
        markets_by_uid=markets_by_uid,
        insights=insights_payload,
        summary=summary_payload,
    )
