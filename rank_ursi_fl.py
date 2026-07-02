#!/usr/bin/env python3
"""URSI-FL official ranker — unified, reproducible, offline, CPU-only.

ONE semantic source of truth:
  * Role-history evidence (career_evidence_semantic / evidence_gate) precomputed
    by precompute_role_semantic_index_fl.py from JD-derived contrastive embeddings
    over deduplicated role documents. No manual template grades. No career-evidence
    keyword lexicons.
  * The final best-role term uses coverage-gated raw role_semantic_evidence from
    unique_role_scores.csv so duration/relevant-month credit is not counted twice,
    while short relevant spikes do not receive full raw peak credit.

Structured differentiators (no keyword matching of career text):
  * availability/behavioral: last-active, open-to-work (market-activity
    calibrated), recruiter response rate + response time, notice period,
    recent applications, offer acceptance, interview completion;
  * logistics/geography: India/visa, JD hubs, remote-preference outside hubs;
  * total-experience prior, GitHub signal, Redrob-verified JD-skill
    assessments (bounded [0.99, 1.02] confirmation factor);
  * documented JD-mirror structured gates: consulting-firm list,
    services-industry career gate, off-domain current-title guard.

Hard gates (shared with the rest of the pipeline via ursi_common):
  * structural honeypot impossibilities and tech anachronisms  -> score 0.

Final ranking: score desc, candidate_id asc. Emits top 100 with grounded reasoning.

Official command (offline, no network, no model load; FL-E is the promoted variant):
  python3 rank_ursi_fl.py \
    --candidates candidates.jsonl \
    --role-projection artifacts/role_semantic_index_fl/fl_e/candidate_role_projection.csv \
    --out submission_ursi_fl_fl_e.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

from ursi_fl_common import REF_DATE, honeypot_reasons, parse_date

TOP_N = 100
RAW_BEST_FULL_RELEVANT_MONTHS = 54.0
RAW_BEST_FULL_RELEVANCE_SHARE = 0.60

PRIORITY_LOCATIONS = (
    "pune", "noida", "delhi", "gurgaon", "gurugram", "ncr", "mumbai",
    "hyderabad",
)

# JD "explicitly do NOT want": careers spent entirely at IT-services / consulting
# firms. This is a STRUCTURED business gate on the company field (not career-text
# keyword matching), mirroring the JD's own named list. Matched as a lowercased
# substring against the company name.
CONSULTING_FIRMS = (
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "tech mahindra", "hcl", "mindtree", "ltimindtree", "deloitte",
    "kpmg", "pwc", "ernst", " ey ", "ibm global", "dxc", "mphasis", "hexaware",
)
SERVICE_INDUSTRIES = {"it services", "consulting"}

OFF_DOMAIN_TECH_TITLE_RE = re.compile(
    r"\b(computer vision|vision engineer|speech|audio|robotics|perception)\b",
    re.I,
)

JD_ASSESSMENT_SKILLS = {
    "Python", "Machine Learning", "NLP", "Semantic Search", "Embeddings",
    "Vector Search", "Information Retrieval", "Learning to Rank",
    "Recommendation Systems", "LLMs", "Fine-tuning LLMs", "PEFT", "LoRA",
    "QLoRA", "Sentence Transformers", "Hugging Face Transformers", "FAISS",
    "BM25", "OpenSearch", "Elasticsearch", "Qdrant", "Weaviate", "Milvus",
    "Pinecone", "pgvector", "RAG", "PyTorch", "scikit-learn", "Haystack",
}

ANCHOR_JD_LANGUAGE = {
    "retrieval_ops": "production retrieval and search infrastructure",
    "ranking_eval": "ranking systems with evaluation rigor (NDCG/MRR/A/B)",
    "candidate_matching": "candidate-job matching and semantic scoring",
    "semantic_search_quality": "semantic search and embedding-based retrieval",
    "production_ml": "production ML systems deployed at scale",
    "matching_finetune_eval": "matching-system evaluation and ML fine-tuning",
    "product_shipper": "shipping ML products end-to-end to real users",
}


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


# ---------------------------------------------------------------------------
# Structured signals (no career-text keyword matching)
# ---------------------------------------------------------------------------
def total_experience_alignment(c: dict[str, Any]) -> float:
    years = float(c.get("profile", {}).get("years_of_experience") or 0)
    if 6.0 <= years <= 8.0:
        return 1.0
    if 5.0 <= years < 6.0 or 8.0 < years <= 9.0:
        return 0.88
    if 4.0 <= years < 5.0 or 9.0 < years <= 11.0:
        return 0.75
    if 3.0 <= years < 4.0 or 11.0 < years <= 13.0:
        return 0.45
    return 0.18


def location_gate(c: dict[str, Any]) -> float:
    """Hard business constraint: JD offers no visa sponsorship (India only)."""
    p = c.get("profile", {})
    s = c.get("redrob_signals", {})
    country = (p.get("country") or "").strip().lower()
    relocate = bool(s.get("willing_to_relocate"))
    if country == "india":
        loc = (p.get("location") or "").lower()
        if any(x in loc for x in PRIORITY_LOCATIONS):
            return 1.0
        return 1.0 if relocate else 0.95
    return 0.60 if relocate else 0.30


def months_since_last_active(c: dict[str, Any]) -> int | None:
    la = parse_date(c.get("redrob_signals", {}).get("last_active_date"))
    if not la:
        return None
    return (REF_DATE.year - la.year) * 12 + (REF_DATE.month - la.month)


def recent_application_factor(c: dict[str, Any]) -> float:
    """Recent applications are a structured job-market signal from Redrob."""
    s = c.get("redrob_signals", {})
    apps = int(s.get("applications_submitted_30d") or 0)
    if apps == 0:
        return 0.97 if s.get("open_to_work_flag") else 0.95
    if apps <= 2:
        return 0.99
    if apps >= 15:
        return 1.015
    if apps >= 8:
        return 1.005
    return 1.0


def open_to_work_factor(c: dict[str, Any]) -> float:
    """Open-to-work is not binary when other market-activity signals disagree."""
    s = c.get("redrob_signals", {})
    if s.get("open_to_work_flag"):
        return 1.0
    apps = int(s.get("applications_submitted_30d") or 0)
    mi = months_since_last_active(c)
    rr = float(s.get("recruiter_response_rate") or 0)
    if apps >= 10 and mi is not None and mi <= 4 and rr >= 0.50:
        return 0.85
    if apps >= 1 and mi is not None and mi <= 2:
        return 0.83
    return 0.80


def response_time_factor(c: dict[str, Any]) -> float:
    hours = c.get("redrob_signals", {}).get("avg_response_time_hours")
    if hours is None:
        return 1.0
    h = float(hours)
    if h <= 12:
        return 1.010
    if h <= 36:
        return 1.005
    if h <= 72:
        return 1.000
    if h <= 120:
        return 0.995
    if h <= 168:
        return 0.990
    return 0.985


def offer_acceptance_factor(c: dict[str, Any]) -> float:
    rate = c.get("redrob_signals", {}).get("offer_acceptance_rate")
    if rate is None:
        return 1.0
    r = float(rate)
    if r < 0:
        return 1.0
    if r >= 0.75:
        return 1.010
    if r >= 0.50:
        return 1.000
    if r >= 0.35:
        return 0.985
    return 0.965


def work_mode_factor(c: dict[str, Any]) -> float:
    """Hybrid/onsite/flexible are neutral; remote-only is only a mild logistics hit."""
    p = c.get("profile", {})
    s = c.get("redrob_signals", {})
    mode = (s.get("preferred_work_mode") or "").strip().lower()
    loc = (p.get("location") or "").lower()
    if mode == "remote" and not s.get("willing_to_relocate"):
        if not any(x in loc for x in PRIORITY_LOCATIONS):
            return 0.985
    return 1.0


def skill_assessment_factor(c: dict[str, Any]) -> float:
    """Tiny confirmation term from Redrob-verified assessments, not self-claims."""
    scores = c.get("redrob_signals", {}).get("skill_assessment_scores") or {}
    jd_scores = [
        float(score) for skill, score in scores.items()
        if skill in JD_ASSESSMENT_SKILLS and isinstance(score, (int, float))
    ]
    if len(jd_scores) >= 2:
        avg = sum(jd_scores) / len(jd_scores)
        if avg >= 75:
            return 1.020
        if avg >= 68:
            return 1.010
        if avg < 45:
            return 0.990
    elif len(jd_scores) == 1 and jd_scores[0] >= 80:
        return 1.005
    return 1.0


def behavioral_gate(c: dict[str, Any]) -> float:
    """Hiring-availability gate using structured Redrob market signals."""
    s = c.get("redrob_signals", {})
    f = 1.0
    mi = months_since_last_active(c)
    if mi is not None:
        if mi <= 2:
            pass
        elif mi <= 4:
            f *= 0.97
        elif mi <= 6:
            f *= 0.90
        elif mi <= 9:
            f *= 0.80
        else:
            f *= 0.70
    rr = float(s.get("recruiter_response_rate") or 0)
    if rr >= 0.5:
        pass
    elif rr >= 0.3:
        f *= 0.96
    elif rr >= 0.1:
        f *= 0.90
    else:
        f *= 0.80
    f *= open_to_work_factor(c)
    np_ = float(s.get("notice_period_days") or 0)
    if np_ <= 30:
        pass
    elif np_ <= 60:
        f *= 0.98
    elif np_ <= 90:
        f *= 0.93
    elif np_ <= 120:
        f *= 0.85
    else:
        f *= 0.80
    icr = s.get("interview_completion_rate")
    if icr is not None and float(icr) < 0.5:
        f *= 0.95
    f *= recent_application_factor(c)
    f *= response_time_factor(c)
    f *= offer_acceptance_factor(c)
    return max(f, 0.45)


def experience_gate(c: dict[str, Any]) -> float:
    """Mild seniority prior (never a hard cut)."""
    return 0.85 + 0.15 * total_experience_alignment(c)


def github_factor(c: dict[str, Any]) -> float:
    """JD: long fully closed-source with no external validation is a weak negative."""
    return 0.97 if c.get("redrob_signals", {}).get("github_activity_score", 0) == -1 else 1.0


def _is_consulting(company: str | None) -> bool:
    c = (company or "").lower()
    return any(f in c for f in CONSULTING_FIRMS)


def _is_services_role(role: dict[str, Any]) -> bool:
    industry = (role.get("industry") or "").strip().lower()
    return _is_consulting(role.get("company")) or industry in SERVICE_INDUSTRIES


def consulting_gate(c: dict[str, Any]) -> float:
    """JD explicit do-not-want: an ENTIRE career at IT-services/consulting firms.

    Negative-only. The JD also says "currently at one of these but with prior
    product-company experience is fine" -> we only penalize when *every* role is
    at a services firm, so any single product-company role spares the candidate.
    """
    roles = [
        r for r in c.get("career_history", [])
        if r.get("company") or r.get("industry")
    ]
    if not roles:
        return 1.0
    return 0.40 if all(_is_services_role(role) for role in roles) else 1.0


def tenure_gate(c: dict[str, Any]) -> float:
    """JD explicit do-not-want: title-chasers who switch every ~1.5 years.

    Negative-only and deliberately mild: only fires for clear serial hoppers
    (4+ roles with low mean *completed* tenure) so a candidate with one or two
    legitimately short stints is not punished.
    """
    career = c.get("career_history", [])
    completed = [r for r in career if not r.get("is_current")]
    if len(career) >= 4 and completed:
        mean_ten = sum(int(r.get("duration_months") or 0) for r in completed) / len(completed)
        if mean_ten < 12:
            return 0.82
        if mean_ten < 16:
            return 0.90
    return 1.0


def off_domain_title_gate(c: dict[str, Any], title_coherence: float, relevant_months: float) -> float:
    """Negative-only guard for JD-named adjacent technical domains.

    The JD explicitly calls out CV/speech/robotics/perception profiles as risky
    unless there is enough NLP/IR/search evidence. Role history still supplies
    the positive signal; this gate only dampens current-title contradictions
    that the title embedding sees as marginal rather than impossible.
    """
    profile = c.get("profile", {})
    title = (profile.get("current_title") or "")
    if not OFF_DOMAIN_TECH_TITLE_RE.search(title):
        return 1.0
    if title_coherence >= 0.55:
        return 1.0

    years = float(profile.get("years_of_experience") or 0)
    if years < 5.0 or relevant_months < RAW_BEST_FULL_RELEVANT_MONTHS:
        return 0.92
    return 0.97


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------
def load_role_projection(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"candidate_id", "career_evidence_semantic"}
        if not required <= set(reader.fieldnames or []):
            raise SystemExit(f"role-projection missing columns; need {required}")
        for r in reader:
            rows[r["candidate_id"]] = {
                "career_evidence_semantic": float(r["career_evidence_semantic"]),
                "semantic_relevant_experience": float(r.get("semantic_relevant_experience") or 0),
                "semantic_relevant_months": float(r.get("semantic_relevant_months") or 0),
                "best_role_evidence": float(r.get("best_role_evidence") or 0),
                "current_title_coherence": float(r.get("current_title_coherence") or 0.5),
                "best_role_doc_id": r.get("best_role_doc_id", ""),
            }
    if not rows:
        raise SystemExit(f"No role projections loaded from {path}")
    return rows


def load_role_previews(path: Path) -> dict[str, str]:
    """Map role_doc_id -> short human-readable evidence snippet (for reasoning)."""
    out: dict[str, str] = {}
    if not path.exists():
        return out
    with path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[r.get("role_doc_id", "")] = (r.get("preview", "") or "").strip()
    return out


def load_role_metadata(path: Path) -> dict[str, dict[str, str]]:
    """Map role_doc_id -> {top_anchor_family, preview} for reasoning enrichment."""
    out: dict[str, dict[str, str]] = {}
    if not path.exists():
        return out
    with path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rid = r.get("role_doc_id", "")
            if rid:
                out[rid] = {
                    "top_anchor_family": r.get("top_anchor_family", ""),
                    "preview": (r.get("preview", "") or "").strip(),
                }
    return out


def load_role_doc_evidence(path: Path) -> dict[str, float]:
    """Map role_doc_id -> raw role_semantic_evidence from unique_role_scores.csv."""
    out: dict[str, float] = {}
    if not path.exists():
        return out
    with path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rid = r.get("role_doc_id", "")
            if not rid:
                continue
            try:
                out[rid] = float(r.get("role_semantic_evidence") or 0)
            except ValueError:
                continue
    return out


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def iter_candidates(path: Path, limit: int = 0):
    with path.open(encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            if limit and idx > limit:
                break
            if line.strip():
                yield json.loads(line)


def score_candidate(
    c: dict[str, Any],
    proj: dict[str, dict[str, Any]],
    role_doc_evidence: dict[str, float] | None = None,
) -> dict[str, Any]:
    cid = c["candidate_id"]

    hp = honeypot_reasons(c)
    if hp:
        return {"candidate_id": cid, "score": 0.0, "honeypot": True,
                "reasons": hp, "candidate": c}

    pr = proj.get(cid)
    if pr is None:
        return {"candidate_id": cid, "score": 0.0, "honeypot": False,
                "reasons": ["missing role projection"], "candidate": c}

    ces = pr["career_evidence_semantic"]
    rel = pr["semantic_relevant_experience"]
    rel_months = pr["semantic_relevant_months"]
    tcoh = pr["current_title_coherence"]
    projected_best_role = pr["best_role_evidence"]
    best_role_doc_id = pr["best_role_doc_id"]
    raw_best_role = (
        role_doc_evidence.get(best_role_doc_id, projected_best_role)
        if role_doc_evidence else projected_best_role
    )
    total_role_months = sum(
        int(role.get("duration_months") or 0)
        for role in c.get("career_history", [])
    )
    relevance_share = rel_months / max(total_role_months, 1)
    raw_best_coverage = min(
        1.0,
        rel_months / RAW_BEST_FULL_RELEVANT_MONTHS,
        relevance_share / RAW_BEST_FULL_RELEVANCE_SHARE,
    )
    effective_best_role = projected_best_role + raw_best_coverage * (
        raw_best_role - projected_best_role
    )
    effective_best_role = clamp(
        effective_best_role,
        min(projected_best_role, raw_best_role),
        max(projected_best_role, raw_best_role),
    )

    # Single semantic source of truth: career-history role evidence. Skills and
    # summaries are deliberately NOT scored here, so keyword-stuffed skill sections
    # have zero effect (the JD's anti-keyword-stuffing intent, by construction).
    # career_evidence_semantic stays continuous at the top (de-saturated ordering).
    #
    # URSI V2: title coherence is NO LONGER a continuous term in base_fit. In this
    # corpus titles are sampled from an in-domain pool per role description, so the
    # title percentile is ~pure noise once the description is fixed (measured: the
    # single strongest template spans title_coh 0.59-0.97 across 7 equally-relevant
    # titles). Title now only acts as the one-sided f_coh contradiction veto below.
    # The freed 0.10 weight is returned to the description-evidence signals.
    #
    # URSI V3.2: the best-role slot uses coverage-gated raw role-document
    # evidence. CES already carries chronology and relevant-month signal, so
    # projected best alone double-counted long stints. But full raw peak credit
    # is only warranted when the candidate has enough relevant career coverage;
    # otherwise interpolate back toward the projected value.
    base_fit = (
        0.60 * ces
        + 0.25 * effective_best_role
        + 0.15 * rel
    )

    reasons: list[str] = []

    # Title-coherence gate (semantic, no keyword list): a clearly non-technical
    # current title under strong role evidence is a planted contradiction. Uses the
    # precomputed title coherence percentile; never boosts above 1.0.
    f_coh = 1.0
    if tcoh < 0.30 and ces >= 0.55:
        f_coh = 0.60
        reasons.append("strong role evidence under a non-technical current title")
    elif tcoh < 0.30 and ces >= 0.40:
        f_coh = 0.80

    f_title_domain = off_domain_title_gate(c, tcoh, rel_months)
    if f_title_domain < 1.0:
        reasons.append("current title is an adjacent CV/speech/robotics domain for this JD")

    f_loc = location_gate(c)
    f_beh = behavioral_gate(c)
    f_work = work_mode_factor(c)
    f_exp = experience_gate(c)
    f_git = github_factor(c)
    f_assess = skill_assessment_factor(c)
    f_cons = consulting_gate(c)
    f_ten = tenure_gate(c)

    score = (
        base_fit * f_loc * f_beh * f_work * f_exp * f_git * f_assess
        * f_coh * f_title_domain * f_cons * f_ten
    )
    return {
        "candidate_id": cid, "score": clamp(score), "honeypot": False,
        "career_evidence_semantic": ces,
        "relevant_experience": rel, "relevant_months": rel_months,
        "total_role_months": total_role_months,
        "relevance_share": relevance_share,
        "raw_best_coverage": raw_best_coverage,
        "title_coherence": tcoh, "best_role_evidence": effective_best_role,
        "effective_best_role_evidence": effective_best_role,
        "projected_best_role_evidence": projected_best_role,
        "raw_best_role_evidence": raw_best_role, "base_fit": base_fit,
        "best_role_doc_id": best_role_doc_id,
        "f_loc": f_loc, "f_beh": f_beh, "f_exp": f_exp, "f_coh": f_coh,
        "f_work": f_work, "f_git": f_git, "f_assess": f_assess,
        "f_title_domain": f_title_domain, "f_cons": f_cons, "f_ten": f_ten,
        "reasons": reasons, "candidate": c,
    }


# ---------------------------------------------------------------------------
# Reasoning
# ---------------------------------------------------------------------------
def _snippet(text: str, n: int = 90) -> str:
    text = (text or "").strip().rstrip(".")
    if len(text) > n:
        text = text[:n].rsplit(" ", 1)[0] + "\u2026"
    return text


def build_reasoning(row: dict[str, Any], rank: int, previews: dict[str, str],
                    role_meta: dict[str, dict[str, str]]) -> str:
    c = row["candidate"]
    p = c.get("profile", {})
    s = c.get("redrob_signals", {})
    title = (p.get("current_title") or "Candidate").strip()
    years = float(p.get("years_of_experience") or 0)
    loc = (p.get("location") or "unknown location").strip()
    company = (p.get("current_company") or "?").strip()
    ces = row["career_evidence_semantic"]
    rel_y = row["relevant_months"] / 12.0
    tcoh = row["title_coherence"]

    doc_id = row.get("best_role_doc_id", "")
    meta = role_meta.get(doc_id, {})
    anchor = meta.get("top_anchor_family", "")
    anchor_desc = ANCHOR_JD_LANGUAGE.get(anchor, "search, ranking, and retrieval")

    variant = int(row["candidate_id"].replace("CAND_", "")) % 3

    if ces >= 0.80:
        fit = (
            f"career history strongly aligns with the JD\u2019s {anchor_desc} requirements",
            f"deep career match on {anchor_desc} \u2014 the JD\u2019s core technical need",
            f"strong fit: {rel_y:.1f}y of relevant experience in {anchor_desc}",
        )[variant]
    elif ces >= 0.62:
        fit = (
            f"career history demonstrates {anchor_desc} experience matching the JD profile",
            f"solid JD alignment on {anchor_desc} with {rel_y:.1f}y relevant",
            f"clear evidence of {anchor_desc} work relevant to the JD",
        )[variant]
    elif ces >= 0.45:
        fit = (
            f"career history shows adjacent experience touching {anchor_desc}",
            f"partial JD alignment: some {anchor_desc} signals but not deep",
            f"relevant adjacent work near {anchor_desc}, lighter than core-fit candidates",
        )[variant]
    else:
        fit = (
            "adjacent profile with limited direct career evidence for this JD",
            "limited career evidence for the JD\u2019s core search/ranking/retrieval profile",
            "role history is peripheral to the JD\u2019s technical requirements",
        )[variant]

    # Spec \u00a72: reasoning must be a 1-2 sentence justification. Sentence 1 packs
    # facts + fit tier + grounded quote with semicolons (the spec's own example
    # style); sentence 2 is the honest concern / behavioral strengths.
    evidence = previews.get(doc_id, "")
    ev_label = ("best evidence", "strongest career signal", "key role evidence")[variant]
    ev_txt = f' \u2014 {ev_label}: \u201c{_snippet(evidence)}\u201d' if evidence else ""

    if variant == 0:
        lead = (f"{years:.1f}y {title} at {company} ({loc}); {fit} "
                f"(role-evidence {ces:.2f}, {rel_y:.1f}y relevant){ev_txt}.")
    elif variant == 1:
        lead = (f"{title} at {company} ({loc}, {years:.1f}y total); {fit} "
                f"(role-evidence {ces:.2f}){ev_txt}.")
    else:
        lead = (f"{fit} (role-evidence {ces:.2f}); "
                f"{years:.1f}y as {title} at {company}, {loc}{ev_txt}.")
    lead = lead[0].upper() + lead[1:]

    concern = None
    rr = float(s.get("recruiter_response_rate") or 0)
    la = parse_date(s.get("last_active_date"))
    mi = (REF_DATE - la).days // 30 if la else None
    in_hub = any(x in loc.lower() for x in PRIORITY_LOCATIONS)
    if row["reasons"]:
        concern = row["reasons"][0]
    elif row.get("f_cons", 1.0) < 1.0:
        concern = "entire career at IT-services/consulting firms (JD prefers product companies)"
    elif row.get("f_ten", 1.0) < 1.0:
        concern = "short average tenure across many roles (job-hopping signal)"
    elif tcoh < 0.40 and ces >= 0.62:
        concern = f"current title ({title}) is off-profile for the strong role evidence"
    elif (p.get("country") or "").strip().lower() != "india":
        concern = f"based outside India ({p.get('country')}); no visa sponsorship per JD"
    elif int(s.get("applications_submitted_30d") or 0) == 0 and s.get("open_to_work_flag"):
        concern = "marked open-to-work but has no recent Redrob applications"
    elif not s.get("open_to_work_flag"):
        concern = "not currently marked open-to-work"
    elif row.get("f_work", 1.0) < 1.0:
        concern = "remote preference outside JD hubs and not open to relocate"
    elif 0 <= float(s.get("offer_acceptance_rate") or -1) < 0.35:
        concern = f"low historical offer acceptance rate ({float(s.get('offer_acceptance_rate')):.2f})"
    elif float(s.get("avg_response_time_hours") or 0) > 120:
        concern = f"slow average response time ({float(s.get('avg_response_time_hours')):.0f}h)"
    elif rr < 0.20:
        concern = f"low recruiter response rate ({rr:.2f})"
    elif mi is not None and mi > 5:
        concern = f"inactive ~{mi} months"
    elif float(s.get("notice_period_days") or 0) > 90:
        concern = f"long notice period ({s.get('notice_period_days')}d)"
    elif not in_hub and not s.get("willing_to_relocate"):
        concern = f"in {loc} (not a JD-preferred hub) and not open to relocate"
    elif s.get("github_activity_score", 0) == -1:
        concern = "no linked GitHub / limited external validation"

    if concern:
        return f"{lead} Concern: {concern}."

    positives = []
    if rr >= 0.65:
        positives.append(f"recruiter response rate {rr:.0%}")
    np_ = int(s.get("notice_period_days") or 0)
    if np_ <= 30:
        positives.append(f"{np_}d notice period")
    apps = int(s.get("applications_submitted_30d") or 0)
    if apps >= 10:
        positives.append("actively applying on Redrob")
    gh = float(s.get("github_activity_score") or -1)
    if gh >= 50:
        positives.append(f"GitHub score {gh:.0f}/100")
    if positives:
        return f"{lead} Behavioral strengths: {'; '.join(positives[:2])}."
    return lead


def write_submission(rows: list[dict[str, Any]], out: Path, previews: dict[str, str],
                     role_meta: dict[str, dict[str, str]]) -> None:
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        prev = float("inf")
        for rank, row in enumerate(rows, start=1):
            score = clamp(min(float(row["score"]), prev - 1e-9))
            prev = score
            w.writerow([row["candidate_id"], rank, f"{score:.9f}",
                        build_reasoning(row, rank, previews, role_meta)])


def main() -> None:
    ap = argparse.ArgumentParser(description="URSI offline ranker")
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--role-projection", type=Path, required=True)
    ap.add_argument("--role-scores", type=Path, default=None,
                    help="Optional unique_role_scores.csv for raw best-role evidence and snippets")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--top-n", type=int, default=TOP_N,
                    help="Rows to emit. Keep the default 100 for the official submission; use <=100 for sandbox smoke tests.")
    args = ap.parse_args()
    if args.top_n <= 0:
        raise SystemExit("--top-n must be positive")

    proj = load_role_projection(args.role_projection)
    role_scores_path = args.role_scores or (args.role_projection.parent / "unique_role_scores.csv")
    previews = load_role_previews(role_scores_path)
    role_meta = load_role_metadata(role_scores_path)
    role_doc_evidence = load_role_doc_evidence(role_scores_path)

    scored = []
    honeypots = 0
    for c in iter_candidates(args.candidates, args.limit):
        r = score_candidate(c, proj, role_doc_evidence)
        if r.get("honeypot"):
            honeypots += 1
        scored.append((round(r["score"], 9), r["candidate_id"], r))
    if len(scored) < args.top_n:
        raise SystemExit(f"Need >= {args.top_n} candidates; got {len(scored)}")

    scored.sort(key=lambda x: (-x[0], x[1]))
    top = [r for _, _, r in scored[:args.top_n]]
    write_submission(top, args.out, previews, role_meta)

    print(f"Ranked {len(scored)} candidates; honeypots excluded: {honeypots}")
    print(f"Wrote top {args.top_n} -> {args.out}")
    print(f"Top {min(10, len(top))}:")
    for i, row in enumerate(top[:10], 1):
        p = row["candidate"].get("profile", {})
        print(f"  {i:02d} {row['candidate_id']} score={row['score']:.4f} "
              f"ces={row['career_evidence_semantic']:.3f} "
              f"title={p.get('current_title')} yrs={p.get('years_of_experience')} loc={p.get('location')}")


if __name__ == "__main__":
    main()
