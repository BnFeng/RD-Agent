from __future__ import annotations

import math
import re
import unicodedata
from collections import defaultdict

from rdagent.app.prediction_market.models import CandidatePair, MarketRecord

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "vs",
    "will",
    "win",
    "with",
    "yes",
    "no",
}


def normalize_text(value: str | None) -> str:
    if not value:
        return ""
    normalized_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    normalized_value = normalized_value.lower()
    normalized_value = re.sub(r"[^a-z0-9]+", " ", normalized_value)
    return re.sub(r"\s+", " ", normalized_value).strip()


def tokenize_text(value: str | None) -> list[str]:
    normalized_value = normalize_text(value)
    if not normalized_value:
        return []
    return [token for token in normalized_value.split(" ") if token and token not in STOPWORDS]


def build_topic_text(market: MarketRecord) -> str:
    parts = [
        market.market_title or "",
        market.display_name or "",
        market.sport_label or "",
        market.league_label or "",
    ]
    return " ".join(part for part in parts if part)


def build_proposition_text(market: MarketRecord) -> str:
    parts = [market.market_title or "", market.outcome or "", market.display_name or ""]
    return " ".join(part for part in parts if part)


def token_set(value: str | None) -> set[str]:
    return set(tokenize_text(value))


def jaccard_similarity(left_tokens: set[str], right_tokens: set[str]) -> float:
    if not left_tokens or not right_tokens:
        return 0.0
    intersection_size = len(left_tokens & right_tokens)
    union_size = len(left_tokens | right_tokens)
    if union_size == 0:
        return 0.0
    return intersection_size / union_size


def numeric_overlap_bonus(left_text: str, right_text: str) -> float:
    left_numbers = set(re.findall(r"\d+(?:\.\d+)?", left_text))
    right_numbers = set(re.findall(r"\d+(?:\.\d+)?", right_text))
    if not left_numbers or not right_numbers:
        return 0.0
    shared_numbers = len(left_numbers & right_numbers)
    return min(shared_numbers * 0.08, 0.16)


def phrase_containment_bonus(left_text: str, right_text: str) -> float:
    if not left_text or not right_text:
        return 0.0
    if left_text in right_text or right_text in left_text:
        return 0.12
    return 0.0


def label_bonus(left_market: MarketRecord, right_market: MarketRecord) -> float:
    score = 0.0
    if left_market.sport_label and left_market.sport_label == right_market.sport_label:
        score += 0.08
    if left_market.league_label and left_market.league_label == right_market.league_label:
        score += 0.08
    return score


def compute_candidate_similarity(left_market: MarketRecord, right_market: MarketRecord) -> tuple[float, float, float, list[str]]:
    left_topic_text = build_topic_text(left_market)
    right_topic_text = build_topic_text(right_market)
    left_prop_text = build_proposition_text(left_market)
    right_prop_text = build_proposition_text(right_market)

    left_topic_tokens = token_set(left_topic_text)
    right_topic_tokens = token_set(right_topic_text)
    left_prop_tokens = token_set(left_prop_text)
    right_prop_tokens = token_set(right_prop_text)
    left_outcome_tokens = token_set(left_market.outcome)
    right_outcome_tokens = token_set(right_market.outcome)

    topic_similarity = jaccard_similarity(left_topic_tokens, right_topic_tokens)
    proposition_similarity = jaccard_similarity(left_prop_tokens, right_prop_tokens)
    outcome_similarity = jaccard_similarity(left_outcome_tokens, right_outcome_tokens)

    heuristic_score = (
        topic_similarity * 0.5
        + proposition_similarity * 0.3
        + outcome_similarity * 0.1
        + numeric_overlap_bonus(left_prop_text, right_prop_text)
        + phrase_containment_bonus(normalize_text(left_prop_text), normalize_text(right_prop_text))
        + label_bonus(left_market, right_market)
    )
    heuristic_score = min(1.0, heuristic_score)
    shared_tokens = sorted(left_topic_tokens & right_topic_tokens)
    return heuristic_score, topic_similarity, proposition_similarity, shared_tokens


def build_recall_candidates(
    left_markets: list[MarketRecord],
    right_markets: list[MarketRecord],
    *,
    max_pairs: int,
    max_candidates_per_left: int,
    min_score: float,
) -> list[CandidatePair]:
    grouped_pairs: dict[str, list[CandidatePair]] = defaultdict(list)

    for left_market in left_markets:
        for right_market in right_markets:
            heuristic_score, topic_similarity, proposition_similarity, shared_tokens = compute_candidate_similarity(
                left_market,
                right_market,
            )
            if heuristic_score < min_score:
                continue
            outcome_similarity = jaccard_similarity(token_set(left_market.outcome), token_set(right_market.outcome))
            grouped_pairs[left_market.market_uid].append(
                CandidatePair(
                    candidate_id=f"recall:{left_market.market_uid}:{right_market.market_uid}",
                    seed_type="recall",
                    left=left_market,
                    right=right_market,
                    source_pair=f"{left_market.source}->{right_market.source}",
                    heuristic_score=heuristic_score,
                    topic_similarity=topic_similarity,
                    proposition_similarity=proposition_similarity,
                    outcome_similarity=outcome_similarity,
                    shared_tokens=shared_tokens[:12],
                )
            )

    selected_pairs: list[CandidatePair] = []
    for left_market_uid, candidates in grouped_pairs.items():
        candidates.sort(
            key=lambda candidate: (
                candidate.heuristic_score,
                candidate.proposition_similarity,
                candidate.topic_similarity,
                -abs((candidate.left.stale_seconds or math.inf) - (candidate.right.stale_seconds or math.inf)),
            ),
            reverse=True,
        )
        selected_pairs.extend(candidates[:max_candidates_per_left])

    selected_pairs.sort(
        key=lambda candidate: (
            candidate.heuristic_score,
            candidate.proposition_similarity,
            candidate.topic_similarity,
        ),
        reverse=True,
    )
    return selected_pairs[:max_pairs]
