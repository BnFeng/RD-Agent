from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class MarketRecord(BaseModel):
    market_uid: str
    source: str
    market_key: str | None = None
    group_key: str | int | None = None
    display_name: str
    market_title: str | None = None
    outcome: str | None = None
    best_bid: float | None = None
    best_ask: float | None = None
    spread: float | None = None
    mid_price: float | None = None
    last_trade_price: float | None = None
    last_trade_size: float | None = None
    catalog_volume: float | None = None
    orderbook_updated_at: str | None = None
    trade_updated_at: str | None = None
    last_updated_at: str | None = None
    anomaly_tags: list[str] = Field(default_factory=list)
    status: str | None = None
    url: str | None = None
    slug: str | None = None
    sport_label: str | None = None
    league_label: str | None = None
    stale_seconds: float | int | None = None

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "market_uid": self.market_uid,
            "source": self.source,
            "display_name": self.display_name,
            "market_title": self.market_title,
            "outcome": self.outcome,
            "best_bid": self.best_bid,
            "best_ask": self.best_ask,
            "mid_price": self.mid_price,
            "catalog_volume": self.catalog_volume,
            "status": self.status,
            "sport_label": self.sport_label,
            "league_label": self.league_label,
            "stale_seconds": self.stale_seconds,
            "url": self.url,
            "slug": self.slug,
        }


class CandidatePair(BaseModel):
    candidate_id: str
    seed_type: Literal["insight", "recall"]
    left: MarketRecord
    right: MarketRecord
    source_pair: str
    heuristic_score: float
    topic_similarity: float
    proposition_similarity: float
    outcome_similarity: float
    shared_tokens: list[str] = Field(default_factory=list)
    execution_type: str | None = None
    signal_label: str | None = None
    reference_gap_bps: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "seed_type": self.seed_type,
            "source_pair": self.source_pair,
            "heuristic_score": round(self.heuristic_score, 4),
            "topic_similarity": round(self.topic_similarity, 4),
            "proposition_similarity": round(self.proposition_similarity, 4),
            "outcome_similarity": round(self.outcome_similarity, 4),
            "shared_tokens": self.shared_tokens,
            "execution_type": self.execution_type,
            "signal_label": self.signal_label,
            "reference_gap_bps": self.reference_gap_bps,
            "left": self.left.to_prompt_dict(),
            "right": self.right.to_prompt_dict(),
            "metadata": self.metadata,
        }


class MatchDecision(BaseModel):
    same_event: bool
    same_market_family: bool
    same_proposition: bool
    outcome_relation: Literal["same", "complement", "conditional_related", "different", "unknown"]
    outcome_mapping: dict[str, str] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0)
    action: Literal["match", "needs_mcp", "reject"]
    reason: str
    risk_notes: list[str] = Field(default_factory=list)


class PairAnalysis(BaseModel):
    candidate: CandidatePair
    decision: MatchDecision
    raw_response: str | None = None


class MonitorState(BaseModel):
    generated_at: str | None = None
    markets: list[MarketRecord]
    markets_by_uid: dict[str, MarketRecord]
    insights: dict[str, Any]
    summary: dict[str, Any]


class SignalCandidate(BaseModel):
    candidate_id: str
    source_signal_bucket: str
    source_signal_key: str
    source_signal_family: str
    display_name: str
    direction_label: str | None = None
    actionability: str | None = None
    priority_score: float | None = None
    expected_edge_bps: float | None = None
    source: str | None = None
    market_uid: str | None = None
    group_key: str | int | None = None
    topic_domain: str | None = None
    strategy_label: str | None = None
    signal_horizon: str | None = None
    horizon_days: float | None = None
    thesis_title: str | None = None
    evidence_count: int | None = None
    alpha_source: str | None = None
    holding_window: str | None = None
    fair_probability: float | None = None
    market_price: float | None = None
    thesis_id: str | None = None
    entry_legs: list[dict[str, Any]] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "source_signal_bucket": self.source_signal_bucket,
            "source_signal_key": self.source_signal_key,
            "source_signal_family": self.source_signal_family,
            "display_name": self.display_name,
            "direction_label": self.direction_label,
            "actionability": self.actionability,
            "priority_score": self.priority_score,
            "expected_edge_bps": self.expected_edge_bps,
            "source": self.source,
            "market_uid": self.market_uid,
            "group_key": self.group_key,
            "topic_domain": self.topic_domain,
            "strategy_label": self.strategy_label,
            "signal_horizon": self.signal_horizon,
            "horizon_days": self.horizon_days,
            "thesis_title": self.thesis_title,
            "evidence_count": self.evidence_count,
            "alpha_source": self.alpha_source,
            "holding_window": self.holding_window,
            "fair_probability": self.fair_probability,
            "market_price": self.market_price,
            "thesis_id": self.thesis_id,
            "entry_legs": self.entry_legs,
            "raw": self.raw,
        }


class SignalDecision(BaseModel):
    emit: bool
    actionability: Literal["tradable", "research", "monitor"]
    confidence: float = Field(ge=0.0, le=1.0)
    priority_score: float | None = None
    max_age_sec: int = Field(ge=1, default=60)
    fair_probability: float | None = None
    target_entry_min: float | None = None
    target_entry_max: float | None = None
    reason: str
    risk_flags: list[str] = Field(default_factory=list)


class RDAgentSignalRecord(BaseModel):
    signal_id: str
    generated_at: str
    expires_at: str | None = None
    signal_label: str = "RD-Agent 信号"
    signal_family: str
    actionability: Literal["tradable", "research", "monitor"]
    priority_score: float | None = None
    expected_edge_bps: float | None = None
    display_name: str
    direction_label: str | None = None
    source: str | None = None
    market_uid: str | None = None
    group_key: str | int | None = None
    entry_legs: list[dict[str, Any]] = Field(default_factory=list)
    source_signal_bucket: str
    source_signal_key: str
    topic_domain: str | None = None
    strategy_label: str | None = None
    signal_horizon: str | None = None
    horizon_days: float | None = None
    thesis_title: str | None = None
    evidence_count: int | None = None
    alpha_source: str | None = None
    holding_window: str | None = None
    thesis_id: str | None = None
    fair_probability: float | None = None
    rdagent_confidence: float = Field(ge=0.0, le=1.0)
    decision_source: str
    rdagent_reason: str
    risk_flags: list[str] = Field(default_factory=list)
    last_updated_at: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)
