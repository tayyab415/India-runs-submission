#!/usr/bin/env python3
"""Shared, offline, dependency-free helpers for the URSI-FL ranking pipeline.

This module deliberately contains NO career-evidence keyword lexicons and NO
manual template grades. The only "word lists" here are:

  * a factual technology -> public-release-date table used purely for temporal
    consistency (anachronism) detection, and
  * JD-derived natural-language anchor paragraphs (positive/negative concepts)
    that are *encoded as text* by an embedding model, never matched literally.

Both the precompute step and the official offline ranker import from here so the
honeypot / consistency logic is identical across the pipeline (fixes the
"semantic-fast lacks hard honeypot exclusion" divergence).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Any

# Single reference "today" used everywhere (was split across 06-22 / 06-26).
REF_DATE = date(2026, 6, 26)


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------
def parse_date(s: Any) -> date | None:
    if not s:
        return None
    try:
        return date.fromisoformat(str(s)[:10])
    except (ValueError, TypeError):
        return None


def months_between(d1: date | None, d2: date | None) -> int | None:
    if not d1 or not d2:
        return None
    return (d2.year - d1.year) * 12 + (d2.month - d1.month)


def add_months(d: date, months: int) -> date:
    m = d.month - 1 + months
    y = d.year + m // 12
    m = m % 12 + 1
    return date(y, m, 1)


def role_end_date(job: dict[str, Any]) -> date | None:
    """Best estimate of when a role stopped being active.

    Priority: explicit end_date -> today if current -> start_date + duration.
    This lets the temporal check cover *current* roles too (V2 fix: the old code
    `if end is None: continue` skipped every current role).
    """
    end = parse_date(job.get("end_date"))
    if end is not None:
        return end
    if job.get("is_current"):
        return REF_DATE
    start = parse_date(job.get("start_date"))
    dur = int(job.get("duration_months") or 0)
    if start is not None and dur > 0:
        return add_months(start, dur)
    return None


# ---------------------------------------------------------------------------
# Temporal consistency: factual tech-release dates (anachronism detection only)
# ---------------------------------------------------------------------------
# Broadened from the original 3-term table. These are public, verifiable release
# dates; matched on word boundaries so substrings (e.g. "llama" in "llamas")
# do not trigger. This is NOT career-evidence matching -- a role mentioning any
# of these terms but ending before the term existed is internally impossible.
TECH_RELEASE_DATES: dict[str, date] = {
    "chatgpt": date(2022, 11, 30),
    "gpt-3.5": date(2022, 11, 30),
    "gpt-4": date(2023, 3, 14),
    "gpt-4o": date(2024, 5, 13),
    "gpt-4 turbo": date(2023, 11, 6),
    "text-embedding-3": date(2024, 1, 25),
    "text-embedding-ada-002": date(2022, 12, 15),
    "llama-2": date(2023, 7, 18),
    "llama 2": date(2023, 7, 18),
    "llama-3": date(2024, 4, 18),
    "llama 3": date(2024, 4, 18),
    "mistral-7b": date(2023, 9, 27),
    "mistral 7b": date(2023, 9, 27),
    "mixtral": date(2023, 12, 11),
    "claude 2": date(2023, 7, 11),
    "claude-2": date(2023, 7, 11),
    "claude 3": date(2024, 3, 4),
    "claude-3": date(2024, 3, 4),
    "gemini": date(2023, 12, 6),
    "qlora": date(2023, 5, 23),
    "dall-e 3": date(2023, 10, 1),
    "stable diffusion": date(2022, 8, 22),
    "whisper": date(2022, 9, 21),
    "langchain": date(2022, 10, 17),
    "llamaindex": date(2022, 11, 9),
    "bge embeddings": date(2023, 8, 5),
    "bge-m3": date(2024, 1, 30),
}

_TECH_PATTERNS = {
    term: re.compile(r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])")
    for term in TECH_RELEASE_DATES
}


def temporal_inconsistencies(candidate: dict[str, Any]) -> list[str]:
    """Return human-readable anachronisms (impossible tech-before-release claims).

    Covers current roles as well (via role_end_date), and a broadened tech table.
    """
    issues: list[str] = []
    for job in candidate.get("career_history", []):
        desc = (job.get("description") or "").lower()
        if not desc:
            continue
        end = role_end_date(job)
        if end is None:
            continue
        for term, release in TECH_RELEASE_DATES.items():
            if end < release and _TECH_PATTERNS[term].search(desc):
                issues.append(f"{term} referenced in a role active until {end.isoformat()} (released {release.isoformat()})")
                break
    return issues


# ---------------------------------------------------------------------------
# Structural honeypot detection (consistency only -- no semantics, no keywords)
# ---------------------------------------------------------------------------
def honeypot_reasons(candidate: dict[str, Any]) -> list[str]:
    """All checks are clean structural impossibilities; any hit => honeypot.

    NOTE: skill duration_months exceeding total experience is NORMAL in this
    dataset (thousands of candidates) and is deliberately NOT used.
    """
    reasons: list[str] = []
    profile = candidate.get("profile", {})
    years = float(profile.get("years_of_experience") or 0)
    cap_m = years * 12
    career = candidate.get("career_history", [])

    if any((int(c.get("duration_months") or 0)) > cap_m + 12 for c in career):
        reasons.append("a single role lasts longer than the entire stated career")

    starts = [parse_date(c.get("start_date")) for c in career]
    starts = [s for s in starts if s]
    if starts:
        span = months_between(min(starts), REF_DATE)
        if span is not None and cap_m > span + 36:
            reasons.append("claimed experience exceeds time since earliest role")

    if sum(int(c.get("duration_months") or 0) for c in career) > cap_m + 36:
        reasons.append("role durations sum implausibly beyond stated experience")

    zero_expert = sum(
        1 for s in candidate.get("skills", [])
        if s.get("proficiency") == "expert" and int(s.get("duration_months") or 0) == 0
    )
    if zero_expert >= 3:
        reasons.append("multiple expert skills with zero months of use")

    reasons.extend(temporal_inconsistencies(candidate))
    return reasons


def is_honeypot(candidate: dict[str, Any]) -> bool:
    return bool(honeypot_reasons(candidate))


# ---------------------------------------------------------------------------
# JD-derived anchor pack (encoded as text; never literally matched)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AnchorFamily:
    name: str
    weight: float
    positive: str
    negative: str


# Seven JD-derived evidence facets from URSI V3.2. URSI-FL keeps the math and
# changes only anchor text variants, so the baseline pack remains available for
# evaluation diffs and rollback.
BASELINE_ROLE_ANCHORS: list[AnchorFamily] = [
    AnchorFamily(
        "retrieval_ops", 0.20,
        "Engineer who built and operated production embeddings-based retrieval deployed to real users: vector databases, hybrid search, index refresh, embedding drift handling, retrieval-quality regression, latency, monitoring, and search infrastructure ownership.",
        "Framework demo, LangChain or OpenAI wrapper, AI-assisted content workflow, or prototype without operational search or retrieval infrastructure, real traffic, drift monitoring, or production ownership.",
    ),
    AnchorFamily(
        "ranking_eval", 0.18,
        "Hands-on ranking, recommendation, or search engineer who built learning-to-rank, feed ranking, recommendation, personalization, relevance evaluation with NDCG, MRR, MAP, offline-online correlation, A/B testing, and feedback loops.",
        "Generic data analyst, BI dashboard, business KPI reporting, SEO search traffic, support knowledge base, or process analytics without ranking models, recommender systems, or relevance evaluation.",
    ),
    AnchorFamily(
        "candidate_matching", 0.16,
        "Engineer who built candidate-job matching, recruiter-facing semantic search, talent-marketplace ranking, or large-scale relevance and matching infrastructure for HR or recruiting products.",
        "HR, recruiting, or talent-acquisition operations, sourcing, or coordination work that uses tools but does not build matching, search, or ranking systems.",
    ),
    AnchorFamily(
        "semantic_search_quality", 0.15,
        "Semantic search engineer who built dense retrieval over a large document corpus: sentence-transformers, BGE or E5 embeddings, FAISS or vector indexes, query expansion, BM25-to-hybrid migration, human relevance judgments, and measurable search-relevance improvements in production.",
        "Search engine optimization, content search traffic, support knowledge-base maintenance, or keyword search usage without building dense retrieval, vector indexes, relevance evaluation, or search systems.",
    ),
    AnchorFamily(
        "production_ml", 0.13,
        "Engineer who shipped production machine learning: model serving, training and inference pipelines, evaluation harnesses, monitoring, and Python or backend ownership of deployed models used by real users.",
        "Academic, research-only, or prototype machine learning, computer vision, speech, or robotics without deployed systems, real users, or production ownership.",
    ),
    AnchorFamily(
        "matching_finetune_eval", 0.10,
        "Engineer who fine-tuned encoder or LLM models on recruiter or relevance labels, generated preference pairs, built ranking evaluation harnesses, and deployed low-latency model serving to improve matching quality.",
        "LLM tutorial, prompt engineering, or chatbot wrapper work without labels, ranking metrics, production serving, or matching infrastructure.",
    ),
    AnchorFamily(
        "product_shipper", 0.08,
        "Product-minded senior individual contributor who ships useful ranking or matching systems quickly, writes production code recently, iterates with PMs, recruiters, and users, and makes pragmatic ML tradeoffs in a startup.",
        "Management-only lead, title-chasing architect, or non-coding strategist who has not recently written production code and has no product-facing ML delivery.",
    ),
]


FL_RETRIEVAL_OPS_NEGATIVE = (
    "Prompt-engineering tutorial, demo chatbot, thin API-integration side project, "
    "or AI-assisted content workflow centered on canned responses, productivity "
    "automation, or content generation rather than an owned production search service."
)

FL_MATCHING_FINETUNE_POSITIVE = (
    "Engineer who fine-tuned encoder or LLM models for recruiter relevance or "
    "candidate-JD matching, using LoRA, QLoRA, PEFT adapters, LLM-based re-ranking, "
    "distillation to low-latency rankers, preference-pair curation, relevance-label "
    "pipelines, ranking evaluation harnesses, and deployed serving to improve "
    "matching quality."
)

FL_PRODUCTION_ML_NEGATIVE = (
    "Computer vision, image or video moderation, object detection, speech, audio, "
    "robotics, or perception engineering as the primary domain. Mechanical, "
    "hardware, civil, or other non-software engineering design work."
)

FL_PRODUCTION_ML_NEGATIVE_STRICT = (
    "Computer-vision image moderation or object-detection model work using PyTorch, "
    "ResNet, YOLO, OpenCV, labeled image datasets, precision/recall monitoring, or "
    "image and video safety pipelines. Speech recognition, TTS, audio modeling, "
    "robotics, perception, CAD, SolidWorks, FEA, DFMA, production tooling, hardware "
    "scale-up, mechanical engineering, or civil engineering design work."
)


def replace_anchor(
    anchors: list[AnchorFamily],
    name: str,
    *,
    positive: str | None = None,
    negative: str | None = None,
) -> list[AnchorFamily]:
    out: list[AnchorFamily] = []
    for anchor in anchors:
        if anchor.name == name:
            out.append(AnchorFamily(
                anchor.name,
                anchor.weight,
                positive if positive is not None else anchor.positive,
                negative if negative is not None else anchor.negative,
            ))
        else:
            out.append(anchor)
    return out


def build_fl_variants() -> dict[str, list[AnchorFamily]]:
    """Return the four fixed-leak anchor variants from HANDOFF_URSI_FL.md."""
    fl_a = replace_anchor(
        BASELINE_ROLE_ANCHORS,
        "retrieval_ops",
        negative=FL_RETRIEVAL_OPS_NEGATIVE,
    )
    fl_b = replace_anchor(
        BASELINE_ROLE_ANCHORS,
        "matching_finetune_eval",
        positive=FL_MATCHING_FINETUNE_POSITIVE,
    )
    fl_c = replace_anchor(
        fl_a,
        "matching_finetune_eval",
        positive=FL_MATCHING_FINETUNE_POSITIVE,
    )
    fl_d = replace_anchor(
        fl_c,
        "production_ml",
        negative=FL_PRODUCTION_ML_NEGATIVE,
    )
    fl_d2 = replace_anchor(
        fl_c,
        "production_ml",
        negative=FL_PRODUCTION_ML_NEGATIVE_STRICT,
    )
    return {
        "baseline": BASELINE_ROLE_ANCHORS,
        "fl_a": fl_a,
        "fl_b": fl_b,
        "fl_c": fl_c,
        "fl_d": fl_d,
        "fl_d2": fl_d2,
        "fl_d3": fl_d2,
        "fl_e": fl_d2,
    }


FL_VARIANT_ANCHORS = build_fl_variants()

# Default import target for scripts that do not pass an explicit anchor pack.
ROLE_ANCHORS = FL_VARIANT_ANCHORS["fl_c"]

ROLE_NEG_LAMBDA = 0.70

# Title coherence: a contradiction guard on the *title* only. It can dampen a
# strong-looking role document stapled onto an unrelated title; it never turns a
# weak role into a strong one.
TITLE_POSITIVE_ANCHOR = (
    "machine learning engineer, applied scientist, search engineer, ranking engineer, "
    "retrieval engineer, recommendation systems engineer, NLP engineer, data scientist, "
    "production ML or backend engineer who builds search, ranking, retrieval, or matching systems."
)
TITLE_NEGATIVE_ANCHOR = (
    "marketing, sales, customer support, accounting, finance, human resources, recruiting, "
    "content writing, graphic design, operations, project management, business analysis, "
    "computer vision, speech recognition, robotics, perception, civil engineering, "
    "or mechanical engineering role."
)
TITLE_NEG_LAMBDA = 0.70
