from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import typer

from rdagent.app.prediction_market.loader import load_monitor_state
from rdagent.app.prediction_market.models import MonitorState, RDAgentSignalRecord, SignalCandidate, SignalDecision
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend

SYSTEM_PROMPT = """
你是 RD-Agent 的预测市场执行信号路由器。

你的任务不是重新发现所有机会，而是对监控系统已经筛出来的候选信号进行二次审核，决定是否发出可以进入模拟下注的结构化信号。

核心要求：
1. 优先保证实时性，若信息陈旧、盘口不足、条件不完整，应降低动作级别或拒绝发信号。
2. 只允许输出 JSON。
3. 你必须显式判断是否 emit。
4. actionability 只能是 tradable、research、monitor。
5. tradable 只在你认为该信号足够明确、足够及时、且执行条件没有明显缺口时给出。
6. 如果是长期信息差，可以保留 fair_probability；如果无法判断则填 null。
7. max_age_sec 必须是一个正整数，短期时间差信号应更短，长期信息差信号可以更长。

输出 JSON 字段：
{
  "emit": true,
  "actionability": "tradable",
  "confidence": 0.82,
  "priority_score": 81.5,
  "max_age_sec": 45,
  "fair_probability": 0.63,
  "target_entry_min": 0.41,
  "target_entry_max": 0.46,
  "reason": "一句中文原因",
  "risk_flags": ["可选风险1", "可选风险2"]
}
""".strip()


def _parse_iso(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp_path.replace(path)


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _signal_key(signal_bucket: str, item: dict[str, Any]) -> str:
    entry_legs = [
        {
            "source": leg.get("source"),
            "market_uid": leg.get("market_uid"),
            "action": leg.get("action"),
            "price": leg.get("price"),
        }
        for leg in (item.get("entry_legs") or item.get("legs") or [])
        if isinstance(leg, dict)
    ]
    identity = {
        "bucket": signal_bucket,
        "family": item.get("signal_family") or signal_bucket,
        "display_name": item.get("display_name"),
        "market_uid": item.get("market_uid"),
        "group_key": item.get("group_key"),
        "direction_label": item.get("direction_label"),
        "entry_legs": entry_legs,
    }
    raw = json.dumps(identity, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _infer_signal_horizon(signal_family: str, item: dict[str, Any]) -> str:
    explicit = item.get("signal_horizon") or item.get("horizon_label")
    if explicit:
        return str(explicit)
    if signal_family == "information_edge":
        return "long"
    if signal_family in {"cross_source_spread", "basket_arbitrage", "microstructure_pressure", "repricing_lag"}:
        return "short"
    return "medium"


def _infer_actionability(signal_bucket: str, item: dict[str, Any]) -> str:
    explicit = item.get("actionability")
    if explicit:
        return str(explicit)
    if signal_bucket == "basket_arbitrage":
        return "tradable"
    if signal_bucket == "cross_source_spread" and item.get("execution_type") == "tradable":
        return "tradable"
    if signal_bucket == "execution_ready" and item.get("signal_label") == "ready":
        return "tradable"
    if signal_bucket in {"repricing_lag", "information_edge"}:
        return "research"
    return "monitor"


def _related_markets(state: MonitorState, candidate: SignalCandidate) -> list[dict[str, Any]]:
    market_ids: list[str] = []
    if candidate.market_uid:
        market_ids.append(candidate.market_uid)
    for leg in candidate.entry_legs:
        market_uid = leg.get("market_uid")
        if market_uid:
            market_ids.append(str(market_uid))
    related: list[dict[str, Any]] = []
    seen: set[str] = set()
    for market_uid in market_ids:
        if market_uid in seen:
            continue
        market = state.markets_by_uid.get(market_uid)
        if market is None:
            continue
        seen.add(market_uid)
        related.append(market.to_prompt_dict())
    return related


def _iter_signal_candidates(
    state: MonitorState,
    *,
    candidate_buckets: list[str],
    candidate_limit: int,
) -> list[SignalCandidate]:
    strategy_watch = state.insights.get("strategy_watch") if isinstance(state.insights, dict) else {}
    if not isinstance(strategy_watch, dict):
        return []
    candidates: list[SignalCandidate] = []
    for bucket in candidate_buckets:
        raw_items = strategy_watch.get(bucket) or []
        if not isinstance(raw_items, list):
            continue
        for raw in raw_items:
            if not isinstance(raw, dict):
                continue
            entry_legs = raw.get("entry_legs") or raw.get("legs") or []
            if not isinstance(entry_legs, list):
                entry_legs = []
            market_uid = raw.get("market_uid")
            if not market_uid and entry_legs:
                market_uid = entry_legs[0].get("market_uid")
            candidate = SignalCandidate(
                candidate_id=f"{bucket}:{_signal_key(bucket, raw)}",
                source_signal_bucket=bucket,
                source_signal_key=_signal_key(bucket, raw),
                source_signal_family=str(raw.get("signal_family") or bucket),
                display_name=str(raw.get("display_name") or raw.get("market_title") or market_uid or bucket),
                direction_label=raw.get("direction_label") or raw.get("signal_reason") or raw.get("reason"),
                actionability=_infer_actionability(bucket, raw),
                priority_score=_safe_float(raw.get("priority_score") or raw.get("signal_score")),
                expected_edge_bps=_safe_float(
                    raw.get("expected_edge_bps")
                    or raw.get("edge_bps")
                    or raw.get("reference_gap_bps")
                    or raw.get("gap_bps")
                ),
                source=raw.get("source") or raw.get("leader_source") or raw.get("buy_source"),
                market_uid=str(market_uid) if market_uid else None,
                group_key=raw.get("group_key"),
                topic_domain=raw.get("topic_domain"),
                strategy_label=raw.get("strategy_label"),
                signal_horizon=_infer_signal_horizon(str(raw.get("signal_family") or bucket), raw),
                horizon_days=_safe_float(raw.get("horizon_days")),
                thesis_title=raw.get("thesis_title"),
                evidence_count=_safe_int(raw.get("evidence_count")),
                alpha_source=raw.get("alpha_source"),
                holding_window=raw.get("holding_window"),
                fair_probability=_safe_float(raw.get("fair_probability")),
                market_price=_safe_float(raw.get("market_price") or raw.get("mid_price") or raw.get("catalog_price")),
                thesis_id=raw.get("thesis_id"),
                entry_legs=[leg for leg in entry_legs if isinstance(leg, dict)],
                raw=raw,
            )
            candidates.append(candidate)
    candidates.sort(
        key=lambda item: (
            0 if item.actionability == "tradable" else 1 if item.actionability == "research" else 2,
            -(item.priority_score or 0.0),
            -(item.expected_edge_bps or 0.0),
        )
    )
    limit = max(1, candidate_limit)
    family_cap = max(1, min(3, limit // 2 or 1))
    family_counts: dict[str, int] = {}
    selected: list[SignalCandidate] = []
    overflow: list[SignalCandidate] = []
    for candidate in candidates:
        family = candidate.source_signal_family or candidate.source_signal_bucket
        if family_counts.get(family, 0) < family_cap:
            selected.append(candidate)
            family_counts[family] = family_counts.get(family, 0) + 1
        else:
            overflow.append(candidate)
        if len(selected) >= limit:
            return selected[:limit]
    for candidate in overflow:
        selected.append(candidate)
        if len(selected) >= limit:
            break
    return selected[:limit]


def _fallback_decision(candidate: SignalCandidate) -> SignalDecision:
    expected_edge_bps = float(candidate.expected_edge_bps or 0.0)
    priority_score = float(candidate.priority_score or 0.0)
    if candidate.source_signal_family in {"cross_source_spread", "basket_arbitrage"} and candidate.actionability == "tradable":
        return SignalDecision(
            emit=True,
            actionability="tradable",
            confidence=min(0.88, 0.58 + min(expected_edge_bps, 400.0) / 1200 + min(priority_score, 100.0) / 500),
            priority_score=max(60.0, priority_score or 60.0),
            max_age_sec=45 if candidate.source_signal_family == "cross_source_spread" else 90,
            fair_probability=None,
            target_entry_min=None,
            target_entry_max=None,
            reason="启发式通过：短期时间差信号边际和优先级满足执行条件",
            risk_flags=[],
        )
    if candidate.source_signal_family == "information_edge" and candidate.actionability in {"tradable", "research"}:
        actionability = "tradable" if candidate.actionability == "tradable" and expected_edge_bps >= 120 else "research"
        return SignalDecision(
            emit=True,
            actionability=actionability,
            confidence=min(0.84, 0.55 + min(expected_edge_bps, 500.0) / 1800 + min(priority_score, 100.0) / 600),
            priority_score=max(55.0, priority_score or 55.0),
            max_age_sec=900,
            fair_probability=candidate.fair_probability,
            target_entry_min=None,
            target_entry_max=None,
            reason="启发式通过：长期信息差具备正边际，进入 RD-Agent 信号层",
            risk_flags=[],
        )
    if candidate.source_signal_family in {"microstructure_pressure", "repricing_lag"} and candidate.actionability in {"tradable", "research"}:
        should_emit = expected_edge_bps >= 80 or priority_score >= 70
        return SignalDecision(
            emit=should_emit,
            actionability="research" if should_emit else "monitor",
            confidence=min(0.8, 0.46 + min(expected_edge_bps, 240.0) / 2200 + min(priority_score, 100.0) / 650),
            priority_score=max(50.0, priority_score or 50.0),
            max_age_sec=120 if candidate.source_signal_family == "microstructure_pressure" else 180,
            fair_probability=candidate.fair_probability,
            target_entry_min=None,
            target_entry_max=None,
            reason="启发式通过：盘口 / 重定价信号进入研究队列，等待后续验证" if should_emit else "启发式拒绝：当前候选未达到 RD-Agent 执行阈值",
            risk_flags=["仅研究跟踪，不进入实时模拟下注"] if should_emit else ["需要更强边际或更明确执行条件"],
        )
    return SignalDecision(
        emit=False,
        actionability="monitor",
        confidence=min(0.7, 0.35 + min(priority_score, 100.0) / 500),
        priority_score=priority_score or 0.0,
        max_age_sec=60,
        fair_probability=candidate.fair_probability,
        target_entry_min=None,
        target_entry_max=None,
        reason="启发式拒绝：当前候选未达到 RD-Agent 执行阈值",
        risk_flags=["需要更强边际或更明确执行条件"],
    )


def _build_prompt(state: MonitorState, candidate: SignalCandidate) -> str:
    payload = {
        "monitor_generated_at": state.generated_at,
        "monitor_metrics": (state.insights or {}).get("metrics", {}),
        "source_health": (state.summary or {}).get("sources", {}),
        "candidate": candidate.to_prompt_dict(),
        "related_markets": _related_markets(state, candidate),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _normalize_decision(raw_response: str) -> SignalDecision:
    payload = json.loads(raw_response)
    if not isinstance(payload, dict):
        raise ValueError("LLM 返回不是 JSON 对象")
    actionability = str(payload.get("actionability") or "monitor").strip().lower()
    if actionability not in {"tradable", "research", "monitor"}:
        actionability = "monitor"
    confidence = _safe_float(payload.get("confidence"))
    priority_score = _safe_float(payload.get("priority_score"))
    max_age_sec = payload.get("max_age_sec")
    try:
        normalized_max_age_sec = max(1, int(max_age_sec))
    except (TypeError, ValueError):
        normalized_max_age_sec = 60
    risk_flags = payload.get("risk_flags")
    return SignalDecision(
        emit=bool(payload.get("emit", False)),
        actionability=actionability,  # type: ignore[arg-type]
        confidence=max(0.0, min(1.0, float(confidence or 0.0))),
        priority_score=priority_score,
        max_age_sec=normalized_max_age_sec,
        fair_probability=_safe_float(payload.get("fair_probability")),
        target_entry_min=_safe_float(payload.get("target_entry_min")),
        target_entry_max=_safe_float(payload.get("target_entry_max")),
        reason=str(payload.get("reason") or "LLM 未提供原因"),
        risk_flags=risk_flags if isinstance(risk_flags, list) else [],
    )


def _evaluate_candidate(candidate: SignalCandidate, state: MonitorState, *, use_llm: bool) -> tuple[SignalDecision, str, str | None]:
    if use_llm:
        try:
            api_backend = APIBackend()
            raw_response = api_backend.build_messages_and_create_chat_completion(
                user_prompt=_build_prompt(state, candidate),
                system_prompt=SYSTEM_PROMPT,
                chat_cache_prefix="prediction_market_signal:",
                shrink_multiple_break=True,
                json_mode=True,
            )
            return _normalize_decision(raw_response), "rdagent_llm", raw_response
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"RD-Agent 信号判定失败，回退启发式 candidate={candidate.candidate_id}: {exc}")
            decision = _fallback_decision(candidate)
            decision.risk_flags.append(f"LLM 失败回退：{exc}")
            return decision, "rdagent_heuristic", None
    return _fallback_decision(candidate), "rdagent_heuristic", None


def _materialize_signal(
    candidate: SignalCandidate,
    decision: SignalDecision,
    *,
    decision_source: str,
    generated_at: datetime,
    snapshot_generated_at: str | None,
) -> RDAgentSignalRecord:
    signal_id = hashlib.sha1(
        json.dumps(
            {
                "source_signal_key": candidate.source_signal_key,
                "decision_source": decision_source,
                "snapshot_generated_at": snapshot_generated_at,
                "actionability": decision.actionability,
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    expires_at = generated_at + timedelta(seconds=max(1, int(decision.max_age_sec)))
    raw = {
        **candidate.raw,
        "source_signal_bucket": candidate.source_signal_bucket,
        "source_signal_key": candidate.source_signal_key,
        "decision_source": decision_source,
        "rdagent_reason": decision.reason,
        "rdagent_risk_flags": decision.risk_flags,
        "rdagent_confidence": decision.confidence,
        "source_snapshot_generated_at": snapshot_generated_at,
        "target_entry_min": decision.target_entry_min,
        "target_entry_max": decision.target_entry_max,
    }
    return RDAgentSignalRecord(
        signal_id=signal_id,
        generated_at=generated_at.isoformat(),
        expires_at=expires_at.isoformat(),
        signal_family=candidate.source_signal_family,
        actionability=decision.actionability,
        priority_score=decision.priority_score if decision.priority_score is not None else candidate.priority_score,
        expected_edge_bps=candidate.expected_edge_bps,
        display_name=candidate.display_name,
        direction_label=candidate.direction_label,
        source=candidate.source,
        market_uid=candidate.market_uid,
        group_key=candidate.group_key,
        entry_legs=candidate.entry_legs,
        source_signal_bucket=candidate.source_signal_bucket,
        source_signal_key=candidate.source_signal_key,
        topic_domain=candidate.topic_domain,
        strategy_label=candidate.strategy_label,
        signal_horizon=candidate.signal_horizon,
        horizon_days=candidate.horizon_days,
        thesis_title=candidate.thesis_title,
        evidence_count=candidate.evidence_count,
        alpha_source=candidate.alpha_source,
        holding_window=candidate.holding_window,
        thesis_id=candidate.thesis_id,
        fair_probability=decision.fair_probability if decision.fair_probability is not None else candidate.fair_probability,
        rdagent_confidence=decision.confidence,
        decision_source=decision_source,
        rdagent_reason=decision.reason,
        risk_flags=decision.risk_flags,
        last_updated_at=snapshot_generated_at,
        raw=raw,
    )


def _run_prediction_market_signal_once(
    *,
    pmm_root: str,
    output_dir: str,
    candidate_buckets: list[str],
    candidate_limit: int,
    max_signals: int,
    min_confidence: float,
    use_llm: bool,
) -> dict[str, Any]:
    state = load_monitor_state(pmm_root)
    generated_at = datetime.now(timezone.utc)
    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    candidates = _iter_signal_candidates(
        state,
        candidate_buckets=candidate_buckets,
        candidate_limit=max(1, candidate_limit),
    )
    analyses: list[dict[str, Any]] = []
    signals: list[dict[str, Any]] = []
    llm_success_count = 0
    heuristic_count = 0
    for candidate in candidates:
        decision, decision_source, raw_response = _evaluate_candidate(candidate, state, use_llm=use_llm)
        if decision_source == "rdagent_llm":
            llm_success_count += 1
        else:
            heuristic_count += 1
        analysis_row = {
            "generated_at": generated_at.isoformat(),
            "source_snapshot_generated_at": state.generated_at,
            "candidate": candidate.model_dump(),
            "decision": decision.model_dump(),
            "decision_source": decision_source,
            "raw_response": raw_response,
        }
        analyses.append(analysis_row)
        if not decision.emit or decision.confidence < min_confidence:
            continue
        signal = _materialize_signal(
            candidate,
            decision,
            decision_source=decision_source,
            generated_at=generated_at,
            snapshot_generated_at=state.generated_at,
        ).model_dump()
        signals.append(signal)

    signals.sort(
        key=lambda item: (
            0 if item.get("actionability") == "tradable" else 1 if item.get("actionability") == "research" else 2,
            -(float(item.get("priority_score") or 0.0)),
            -(float(item.get("expected_edge_bps") or 0.0)),
            -(float(item.get("rdagent_confidence") or 0.0)),
        )
    )
    signals = signals[: max(1, max_signals)]
    actionability_counts: dict[str, int] = {}
    family_counts: dict[str, int] = {}
    for signal in signals:
        actionability = str(signal.get("actionability") or "unknown")
        family = str(signal.get("signal_family") or "unknown")
        actionability_counts[actionability] = actionability_counts.get(actionability, 0) + 1
        family_counts[family] = family_counts.get(family, 0) + 1

    current_payload = {
        "generated_at": generated_at.isoformat(),
        "updated_at": generated_at.isoformat(),
        "source_snapshot_generated_at": state.generated_at,
        "mode": "llm" if llm_success_count > 0 else "heuristic",
        "items": signals,
        "signals": signals,
        "stats": {
            "candidate_count": len(candidates),
            "emitted_signal_count": len(signals),
            "tradable_signal_count": actionability_counts.get("tradable", 0),
            "research_signal_count": actionability_counts.get("research", 0),
            "monitor_signal_count": actionability_counts.get("monitor", 0),
            "llm_success_count": llm_success_count,
            "heuristic_count": heuristic_count,
            "candidate_buckets": candidate_buckets,
            "actionability_counts": actionability_counts,
            "family_counts": family_counts,
        },
    }
    _write_json(output_root / "current.json", current_payload)
    _append_jsonl(output_root / "history.jsonl", [{**item, "batch_generated_at": generated_at.isoformat()} for item in signals])
    _append_jsonl(output_root / "decision_trace.jsonl", analyses)

    summary = {
        "generated_at": generated_at.isoformat(),
        "source_snapshot_generated_at": state.generated_at,
        "candidate_count": len(candidates),
        "signal_count": len(signals),
        "mode": current_payload["mode"],
        "candidate_buckets": candidate_buckets,
        "tradable_signal_count": actionability_counts.get("tradable", 0),
        "research_signal_count": actionability_counts.get("research", 0),
        "output_dir": str(output_root),
    }
    logger.log_object(summary, tag="prediction_market_signal_summary")
    return summary


def prediction_market_signal_once(
    pmm_root: str = typer.Option("../prediction-market-monitor", help="prediction-market-monitor 根目录"),
    output_dir: str = typer.Option("", help="输出目录，默认写入 PMM 的 data/rdagent"),
    candidate_buckets: str = typer.Option("cross_source_spread,basket_arbitrage,information_edge,formal_signals", help="候选信号桶，逗号分隔"),
    candidate_limit: int = typer.Option(10, min=1, help="最多分析多少个候选信号"),
    max_signals: int = typer.Option(8, min=1, help="最多保留多少个 RD-Agent 信号"),
    min_confidence: float = typer.Option(0.66, min=0.0, max=1.0, help="最小置信度"),
    use_llm: bool = typer.Option(True, "--llm/--no-llm", help="是否调用 RD-Agent LLM 后端"),
) -> None:
    root_path = Path(pmm_root).expanduser().resolve()
    final_output_dir = output_dir or str(root_path / "data" / "rdagent")
    buckets = [item.strip() for item in candidate_buckets.split(",") if item.strip()]
    summary = _run_prediction_market_signal_once(
        pmm_root=str(root_path),
        output_dir=final_output_dir,
        candidate_buckets=buckets,
        candidate_limit=candidate_limit,
        max_signals=max_signals,
        min_confidence=min_confidence,
        use_llm=use_llm,
    )
    typer.echo(json.dumps(summary, ensure_ascii=False, indent=2))


def prediction_market_signal_worker(
    pmm_root: str = typer.Option("../prediction-market-monitor", help="prediction-market-monitor 根目录"),
    output_dir: str = typer.Option("", help="输出目录，默认写入 PMM 的 data/rdagent"),
    candidate_buckets: str = typer.Option("cross_source_spread,basket_arbitrage,information_edge,formal_signals", help="候选信号桶，逗号分隔"),
    candidate_limit: int = typer.Option(10, min=1, help="最多分析多少个候选信号"),
    max_signals: int = typer.Option(8, min=1, help="最多保留多少个 RD-Agent 信号"),
    min_confidence: float = typer.Option(0.66, min=0.0, max=1.0, help="最小置信度"),
    interval_sec: int = typer.Option(15, min=3, help="轮询刷新间隔秒数"),
    use_llm: bool = typer.Option(True, "--llm/--no-llm", help="是否调用 RD-Agent LLM 后端"),
) -> None:
    root_path = Path(pmm_root).expanduser().resolve()
    final_output_dir = output_dir or str(root_path / "data" / "rdagent")
    buckets = [item.strip() for item in candidate_buckets.split(",") if item.strip()]
    last_snapshot_ts: str | None = None
    while True:
        try:
            state = load_monitor_state(root_path)
            if state.generated_at and state.generated_at == last_snapshot_ts:
                time.sleep(interval_sec)
                continue
            summary = _run_prediction_market_signal_once(
                pmm_root=str(root_path),
                output_dir=final_output_dir,
                candidate_buckets=buckets,
                candidate_limit=candidate_limit,
                max_signals=max_signals,
                min_confidence=min_confidence,
                use_llm=use_llm,
            )
            last_snapshot_ts = summary.get("source_snapshot_generated_at")
            typer.echo(json.dumps(summary, ensure_ascii=False))
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"prediction_market_signal_worker 运行失败：{exc}")
        time.sleep(interval_sec)
