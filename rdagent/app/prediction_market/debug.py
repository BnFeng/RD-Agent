from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from rdagent.app.prediction_market.loader import load_monitor_state
from rdagent.app.prediction_market.models import CandidatePair, MatchDecision, MonitorState, PairAnalysis
from rdagent.app.prediction_market.normalizer import (
    build_recall_candidates,
    compute_candidate_similarity,
    normalize_text,
)
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend

SYSTEM_PROMPT = """
你是预测市场跨平台语义匹配引擎。

你的任务是判断两个盘口是否真的可以视为同一个可比较命题，而不是只看标题像不像。

判定原则：
1. 必须先判断是否属于同一事件或同一主题对象。
2. 必须再判断是否属于同一命题，特别注意时间窗、阈值、结算条件、是否为“冠军/提名/是否发生”这类不同命题。
3. 若只是同一主题，但阈值不同、时间不同、条件不同、对象范围不同，不可直接判为 match。
4. 若信息不充分，但你认为高度可能是同题且需要查看规则页、市场说明页、结算条款页，输出 needs_mcp。
5. 不要依据价格判断语义是否一致，价格只能当辅助上下文。

输出要求：
- 只输出 JSON。
- confidence 取值 0 到 1。
- 只有在 same_event=true 且 same_proposition=true 且 confidence>=0.75 时，action 才能为 match。
- 若 outcome 是互补关系，例如 Yes vs No，可输出 outcome_relation=complement。
- 若命题存在明显条件差异，例如 >4B 与 >6B，通常应判定 same_proposition=false。
""".strip()

OUTCOME_RELATION_ALIASES = {
    "equivalent": "same",
    "identical": "same",
    "same": "same",
    "inverse": "complement",
    "opposite": "complement",
    "complementary": "complement",
    "complement": "complement",
    "conditional": "conditional_related",
    "related": "conditional_related",
    "conditional_related": "conditional_related",
    "different": "different",
    "distinct": "different",
    "unknown": "unknown",
}

ACTION_ALIASES = {
    "match": "match",
    "needs_mcp": "needs_mcp",
    "need_mcp": "needs_mcp",
    "needs review": "needs_mcp",
    "review": "needs_mcp",
    "reject": "reject",
    "no_match": "reject",
}


def _filter_markets_by_source(
    state: MonitorState,
    source_name: str,
    *,
    max_stale_seconds: int,
) -> list:
    filtered_markets = [market for market in state.markets if market.source == source_name]
    if max_stale_seconds <= 0:
        return filtered_markets
    fresh_markets = [
        market
        for market in filtered_markets
        if market.stale_seconds is None or float(market.stale_seconds) <= max_stale_seconds
    ]
    return fresh_markets or filtered_markets


def _build_insight_candidates(
    state: MonitorState,
    *,
    source_a: str,
    source_b: str,
    limit: int,
) -> list[CandidatePair]:
    candidates: list[CandidatePair] = []
    seen_pairs: set[tuple[str, str]] = set()
    cross_source_watch = state.insights.get("strategy_watch", {}).get("cross_source_spread", [])

    for item in cross_source_watch:
        left_source = item.get("left_source")
        right_source = item.get("right_source")
        pair_matches = (left_source == source_a and right_source == source_b) or (
            left_source == source_b and right_source == source_a
        )
        if not pair_matches:
            continue

        left_market_uid = item.get("left_market_uid")
        right_market_uid = item.get("right_market_uid")
        if not left_market_uid or not right_market_uid:
            continue

        left_market = state.markets_by_uid.get(left_market_uid)
        right_market = state.markets_by_uid.get(right_market_uid)
        if left_market is None or right_market is None:
            continue

        if left_market.source != source_a:
            left_market, right_market = right_market, left_market

        pair_key = (left_market.market_uid, right_market.market_uid)
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)

        heuristic_score, topic_similarity, proposition_similarity, shared_tokens = compute_candidate_similarity(
            left_market,
            right_market,
        )
        outcome_similarity = 1.0 if normalize_text(left_market.outcome) == normalize_text(right_market.outcome) else 0.0

        candidates.append(
            CandidatePair(
                candidate_id=f"insight:{left_market.market_uid}:{right_market.market_uid}",
                seed_type="insight",
                left=left_market,
                right=right_market,
                source_pair=f"{left_market.source}->{right_market.source}",
                heuristic_score=heuristic_score,
                topic_similarity=topic_similarity,
                proposition_similarity=proposition_similarity,
                outcome_similarity=outcome_similarity,
                shared_tokens=shared_tokens[:12],
                execution_type=item.get("execution_type"),
                signal_label=item.get("signal_label"),
                reference_gap_bps=item.get("reference_gap_bps"),
                metadata={
                    "display_name": item.get("display_name"),
                    "similarity": item.get("similarity"),
                    "gap_bps": item.get("gap_bps"),
                    "buy_source": item.get("buy_source"),
                    "sell_source": item.get("sell_source"),
                },
            )
        )
        if len(candidates) >= limit:
            break

    return candidates


def _deduplicate_candidates(candidates: list[CandidatePair], limit: int) -> list[CandidatePair]:
    deduplicated: list[CandidatePair] = []
    seen_pairs: set[tuple[str, str]] = set()

    for candidate in candidates:
        pair_key = (candidate.left.market_uid, candidate.right.market_uid)
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)
        deduplicated.append(candidate)
        if len(deduplicated) >= limit:
            break

    return deduplicated


def _fallback_decision(candidate: CandidatePair, error_message: str | None = None) -> MatchDecision:
    left_outcome = normalize_text(candidate.left.outcome)
    right_outcome = normalize_text(candidate.right.outcome)
    outcome_relation = "unknown"
    outcome_mapping: dict[str, str] = {}

    if left_outcome and right_outcome:
        if left_outcome == right_outcome:
            outcome_relation = "same"
            outcome_mapping = {"left_outcome": candidate.left.outcome or "", "right_outcome": candidate.right.outcome or ""}
        elif {left_outcome, right_outcome} == {"yes", "no"}:
            outcome_relation = "complement"
            outcome_mapping = {"left_outcome": candidate.left.outcome or "", "right_outcome": candidate.right.outcome or ""}
        else:
            outcome_relation = "different"

    same_event = candidate.topic_similarity >= 0.55
    same_proposition = candidate.proposition_similarity >= 0.62
    confidence = min(0.74, round(candidate.heuristic_score, 3))
    action = "reject"
    if same_event and same_proposition and confidence >= 0.55:
        action = "needs_mcp"
    elif candidate.heuristic_score >= 0.45:
        action = "needs_mcp"

    risk_notes = []
    if candidate.left.outcome and candidate.right.outcome and outcome_relation == "different":
        risk_notes.append("结果项名称不一致，可能不是同一命题")
    if error_message:
        risk_notes.append(f"LLM 调用失败，已回退到启发式判定：{error_message}")

    return MatchDecision(
        same_event=same_event,
        same_market_family=same_event,
        same_proposition=same_proposition,
        outcome_relation=outcome_relation,
        outcome_mapping=outcome_mapping,
        confidence=confidence,
        action=action,
        reason=f"启发式分数={candidate.heuristic_score:.3f}，topic={candidate.topic_similarity:.3f}，prop={candidate.proposition_similarity:.3f}",
        risk_notes=risk_notes,
    )


def _build_user_prompt(candidate: CandidatePair) -> str:
    return json.dumps(candidate.to_prompt_dict(), ensure_ascii=False, indent=2)


def _normalize_match_decision(raw_response: str) -> MatchDecision:
    payload = json.loads(raw_response)
    if not isinstance(payload, dict):
        raise ValueError("LLM 返回不是 JSON 对象")

    outcome_relation = str(payload.get("outcome_relation", "unknown")).strip().lower()
    payload["outcome_relation"] = OUTCOME_RELATION_ALIASES.get(outcome_relation, "unknown")

    payload["same_event"] = bool(payload.get("same_event", False))
    payload["same_proposition"] = bool(payload.get("same_proposition", False))
    payload["same_market_family"] = bool(
        payload.get("same_market_family", payload["same_event"] or payload["same_proposition"])
    )

    confidence = payload.get("confidence", 0.0)
    try:
        payload["confidence"] = max(0.0, min(1.0, float(confidence)))
    except (TypeError, ValueError):
        payload["confidence"] = 0.0

    action_value = str(payload.get("action", "")).strip().lower()
    payload["action"] = ACTION_ALIASES.get(action_value, "")
    if payload["action"] not in {"match", "needs_mcp", "reject"}:
        if payload["same_event"] and payload["same_proposition"] and payload["confidence"] >= 0.75:
            payload["action"] = "match"
        elif payload["same_event"] or payload["same_proposition"]:
            payload["action"] = "needs_mcp"
        else:
            payload["action"] = "reject"

    outcome_mapping = payload.get("outcome_mapping")
    payload["outcome_mapping"] = outcome_mapping if isinstance(outcome_mapping, dict) else {}

    risk_notes = payload.get("risk_notes")
    payload["risk_notes"] = risk_notes if isinstance(risk_notes, list) else []

    reason = payload.get("reason")
    payload["reason"] = str(reason).strip() if reason else "LLM 未提供原因"

    return MatchDecision.model_validate(payload)


def _evaluate_candidate_with_llm(candidate: CandidatePair) -> PairAnalysis:
    api_backend = APIBackend()
    user_prompt = _build_user_prompt(candidate)
    response_kwargs: dict[str, Any] = {
        "user_prompt": user_prompt,
        "system_prompt": SYSTEM_PROMPT,
        "chat_cache_prefix": "prediction_market_match:",
        "shrink_multiple_break": True,
        "json_mode": True,
    }

    raw_response = api_backend.build_messages_and_create_chat_completion(**response_kwargs)
    decision = _normalize_match_decision(raw_response)
    return PairAnalysis(candidate=candidate, decision=decision, raw_response=raw_response)


def _evaluate_candidates(candidates: list[CandidatePair], use_llm: bool) -> list[PairAnalysis]:
    analyses: list[PairAnalysis] = []
    for index, candidate in enumerate(candidates, start=1):
        with logger.tag(f"candidate_{index:03d}"):
            logger.log_object(candidate.model_dump(), tag="candidate_input")
            if use_llm:
                try:
                    analysis = _evaluate_candidate_with_llm(candidate)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"候选对 {candidate.candidate_id} LLM 判定失败：{exc}")
                    analysis = PairAnalysis(
                        candidate=candidate,
                        decision=_fallback_decision(candidate, str(exc)),
                        raw_response=None,
                    )
            else:
                analysis = PairAnalysis(candidate=candidate, decision=_fallback_decision(candidate), raw_response=None)
            logger.log_object(analysis.model_dump(), tag="candidate_result")
            analyses.append(analysis)
    return analyses


def _build_summary(
    state: MonitorState,
    analyses: list[PairAnalysis],
    *,
    source_a: str,
    source_b: str,
    mode: str,
) -> dict[str, Any]:
    action_counter = {"match": 0, "needs_mcp": 0, "reject": 0}
    for analysis in analyses:
        action_counter[analysis.decision.action] += 1

    return {
        "generated_at": state.generated_at,
        "mode": mode,
        "source_pair": f"{source_a}->{source_b}",
        "market_snapshot_count": len(state.markets),
        "analysis_count": len(analyses),
        "actions": action_counter,
        "results": [
            {
                "candidate_id": analysis.candidate.candidate_id,
                "seed_type": analysis.candidate.seed_type,
                "left_market_uid": analysis.candidate.left.market_uid,
                "right_market_uid": analysis.candidate.right.market_uid,
                "left_display_name": analysis.candidate.left.display_name,
                "right_display_name": analysis.candidate.right.display_name,
                "heuristic_score": analysis.candidate.heuristic_score,
                "topic_similarity": analysis.candidate.topic_similarity,
                "proposition_similarity": analysis.candidate.proposition_similarity,
                "action": analysis.decision.action,
                "confidence": analysis.decision.confidence,
                "same_event": analysis.decision.same_event,
                "same_proposition": analysis.decision.same_proposition,
                "outcome_relation": analysis.decision.outcome_relation,
                "reason": analysis.decision.reason,
                "risk_notes": analysis.decision.risk_notes,
            }
            for analysis in analyses
        ],
        "monitor_metrics": state.insights.get("metrics", {}),
    }


def _run_prediction_market_debug(
    *,
    pmm_root: str,
    source_a: str,
    source_b: str,
    mode: str,
    limit: int,
    max_candidates_per_left: int,
    min_score: float,
    max_stale_seconds: int,
    use_llm: bool,
    output_path: str,
) -> dict[str, Any]:
    state = load_monitor_state(pmm_root)
    left_markets = _filter_markets_by_source(state, source_a, max_stale_seconds=max_stale_seconds)
    right_markets = _filter_markets_by_source(state, source_b, max_stale_seconds=max_stale_seconds)

    insight_candidates: list[CandidatePair] = []
    recall_candidates: list[CandidatePair] = []

    if mode in {"insight", "mixed"}:
        insight_candidates = _build_insight_candidates(state, source_a=source_a, source_b=source_b, limit=limit)

    if mode in {"recall", "mixed"}:
        recall_candidates = build_recall_candidates(
            left_markets,
            right_markets,
            max_pairs=limit * max_candidates_per_left,
            max_candidates_per_left=max_candidates_per_left,
            min_score=min_score,
        )

    candidates = _deduplicate_candidates(insight_candidates + recall_candidates, limit)

    with logger.tag(f"prediction_market.{source_a}_vs_{source_b}"):
        logger.log_object(
            {
                "pmm_root": str(Path(pmm_root).expanduser().resolve()),
                "source_a": source_a,
                "source_b": source_b,
                "mode": mode,
                "limit": limit,
                "max_candidates_per_left": max_candidates_per_left,
                "min_score": min_score,
                "max_stale_seconds": max_stale_seconds,
                "use_llm": use_llm,
                "market_count": len(state.markets),
                "left_market_count": len(left_markets),
                "right_market_count": len(right_markets),
            },
            tag="run_config",
        )
        logger.log_object(state.summary, tag="monitor_summary")
        logger.log_object([candidate.model_dump() for candidate in candidates], tag="candidate_pairs")
        analyses = _evaluate_candidates(candidates, use_llm=use_llm)
        summary = _build_summary(state, analyses, source_a=source_a, source_b=source_b, mode=mode)
        logger.log_object(summary, tag="summary")

    if output_path:
        output_file = Path(output_path).expanduser().resolve()
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return summary


def prediction_market_debug(
    pmm_root: str = typer.Option(
        "../prediction-market-monitor",
        help="prediction-market-monitor 根目录",
    ),
    source_a: str = typer.Option("opinion", help="左侧来源"),
    source_b: str = typer.Option("polymarket", help="右侧来源"),
    mode: str = typer.Option("mixed", help="候选来源：insight、recall、mixed"),
    limit: int = typer.Option(8, min=1, help="最大分析候选数"),
    max_candidates_per_left: int = typer.Option(3, min=1, help="每个左侧市场最多召回候选数"),
    min_score: float = typer.Option(0.38, min=0.0, max=1.0, help="启发式召回最小分数"),
    max_stale_seconds: int = typer.Option(900, min=0, help="候选召回的最大陈旧秒数，0 表示不过滤"),
    use_llm: bool = typer.Option(True, "--llm/--no-llm", help="是否调用 LLM 做结构化判定"),
    output_path: str = typer.Option("", help="可选：结果 JSON 输出路径"),
) -> None:
    summary = _run_prediction_market_debug(
        pmm_root=pmm_root,
        source_a=source_a,
        source_b=source_b,
        mode=mode,
        limit=limit,
        max_candidates_per_left=max_candidates_per_left,
        min_score=min_score,
        max_stale_seconds=max_stale_seconds,
        use_llm=use_llm,
        output_path=output_path,
    )
    typer.echo(json.dumps(summary, ensure_ascii=False, indent=2))


def prediction_market_match(
    pmm_root: str = typer.Option(
        "../prediction-market-monitor",
        help="prediction-market-monitor 根目录",
    ),
    source_a: str = typer.Option("opinion", help="左侧来源"),
    source_b: str = typer.Option("polymarket", help="右侧来源"),
    limit: int = typer.Option(12, min=1, help="最大分析候选数"),
    max_candidates_per_left: int = typer.Option(4, min=1, help="每个左侧市场最多召回候选数"),
    min_score: float = typer.Option(0.42, min=0.0, max=1.0, help="启发式召回最小分数"),
    max_stale_seconds: int = typer.Option(900, min=0, help="候选召回的最大陈旧秒数，0 表示不过滤"),
    use_llm: bool = typer.Option(True, "--llm/--no-llm", help="是否调用 LLM 做结构化判定"),
    output_path: str = typer.Option("", help="可选：结果 JSON 输出路径"),
) -> None:
    summary = _run_prediction_market_debug(
        pmm_root=pmm_root,
        source_a=source_a,
        source_b=source_b,
        mode="recall",
        limit=limit,
        max_candidates_per_left=max_candidates_per_left,
        min_score=min_score,
        max_stale_seconds=max_stale_seconds,
        use_llm=use_llm,
        output_path=output_path,
    )
    typer.echo(json.dumps(summary, ensure_ascii=False, indent=2))
