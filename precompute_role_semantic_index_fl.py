#!/usr/bin/env python3
"""URSI-FL precompute: fixed-leak role semantic index.

This step MAY use the network (Azure text-embedding-3-large). It produces frozen
CSV artifacts that the official offline ranker consumes. It does NOT read manual
template grades and does NOT assume the corpus has 44 templates -- it deduplicates
whatever unique role descriptions / titles exist.

Pipeline:
  1. Stream candidates.jsonl; discover unique role descriptions and titles by hash.
  2. Embed unique descriptions, unique titles, and JD-derived anchor paragraphs
     with Azure text-embedding-3-large (cached to .npz for reproducibility).
  3. Per unique description: contrastive multi-anchor score, rank-percentile
     normalized within the discovered corpus -> role_semantic_evidence.
  4. Per unique title: semantic title coherence (contradiction guard).
  5. Project role-level evidence to candidates by recency, duration, currentness,
     breadth, and title coherence -> candidate_role_projection.csv.

Outputs (artifacts/role_semantic_index_fl/<variant>/):
  manifest.json, jd_anchor_pack.json, unique_role_scores.csv,
  candidate_role_projection.csv, validation_report.md, plus summary/skill
  diagnostic projections. The offline ranker consumes only the role projection
  and role scores; summary/skill artifacts are for review and false-positive
  auditing unless a separate validated methodology promotes them.

Usage (fl_e is the promoted/official variant):
  export AZURE_OPENAI_API_KEY="your-azure-openai-key"   # see README.md
  python3 precompute_role_semantic_index_fl.py \
      --candidates candidates.jsonl --variant fl_e \
      --out-dir artifacts/role_semantic_index_fl/fl_e \
      --cache artifacts/role_semantic_index_fl/embed_cache_fl.npz
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ursi_fl_common import (
    FL_VARIANT_ANCHORS,
    REF_DATE,
    ROLE_NEG_LAMBDA,
    AnchorFamily,
    TITLE_NEGATIVE_ANCHOR,
    TITLE_NEG_LAMBDA,
    TITLE_POSITIVE_ANCHOR,
    months_between,
    parse_date,
)

DEPLOYMENT = "text-embedding-3-large"
API_VERSION = "2024-10-21"
# Set to your own Azure OpenAI resource endpoint before running precompute.
# Not needed to reproduce the submission CSV -- rank_ursi_fl.py only reads the
# already-frozen artifacts under artifacts/role_semantic_index_fl/fl_e/.
AZURE_RESOURCE_URL = os.environ.get("AZURE_OPENAI_RESOURCE_URL", "")
DEFAULT_OUT_ROOT = Path("artifacts/role_semantic_index_fl")
DEFAULT_CACHE = DEFAULT_OUT_ROOT / "embed_cache_fl.npz"
BASELINE_CACHE = Path("artifacts/role_semantic_index/embed_cache.npz")
NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\+?\b")


# ---------------------------------------------------------------------------
# Text normalization / hashing
# ---------------------------------------------------------------------------
def norm_text(s: Any) -> str:
    return " ".join(str(s or "").split())


def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def normalize_numbers(text: str) -> str:
    return NUMBER_RE.sub("<NUM>", norm_text(text))


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def summary_flags(template: str) -> dict[str, bool]:
    lower = template.lower()
    return {
        "summary_ai_curiosity_flag": any(
            phrase in lower
            for phrase in (
                "curious about how ai tools",
                "excited about how ai and genai tools",
                "self-learner level",
                "online courses",
                "side project",
                "transitioning toward",
                "want to grow into",
                "start contributing to ml-adjacent systems",
            )
        ),
        "summary_no_professional_ml_flag": any(
            phrase in lower
            for phrase in (
                "haven't done it in a professional capacity",
                "not the core of my day",
                "still building depth",
                "lighter on the deep-learning side",
            )
        ),
        "summary_search_retrieval_claim_flag": any(
            phrase in lower
            for phrase in (
                "search, retrieval, and ranking",
                "semantic search",
                "hybrid retrieval",
                "keyword-based ranking",
                "recommendation system",
                "learning-to-rank",
                "candidate-jd matching",
                "ndcg",
                "mrr",
            )
        ),
        "summary_transition_claim_flag": any(
            phrase in lower
            for phrase in (
                "transitioning toward",
                "want to grow into",
                "looking to grow into",
                "start contributing",
            )
        ),
    }


def anchor_pack_payload(name: str, anchors: list[AnchorFamily]) -> dict[str, Any]:
    return {
        "name": name,
        "source": "HANDOFF_URSI_FL.md fixed-leak anchor variant",
        "role_neg_lambda": ROLE_NEG_LAMBDA,
        "families": [
            {
                "name": a.name,
                "weight": a.weight,
                "positive": a.positive,
                "negative": a.negative,
            }
            for a in anchors
        ],
        "title_positive": TITLE_POSITIVE_ANCHOR,
        "title_negative": TITLE_NEGATIVE_ANCHOR,
    }


def write_builtin_anchor_packs(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, anchors in FL_VARIANT_ANCHORS.items():
        if name == "baseline":
            continue
        path = out_dir / f"ursi_{name}.json"
        path.write_text(json.dumps(anchor_pack_payload(name, anchors), indent=2), encoding="utf-8")


def load_anchor_pack(path: Path) -> tuple[str, list[AnchorFamily]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    anchors = [
        AnchorFamily(
            row["name"],
            float(row["weight"]),
            row["positive"],
            row["negative"],
        )
        for row in payload["families"]
    ]
    return payload.get("name") or path.stem, anchors


# ---------------------------------------------------------------------------
# Azure embeddings (precompute only) with local cache
# ---------------------------------------------------------------------------
def azure_embed(texts: list[str], api_key: str) -> np.ndarray:
    url = f"{AZURE_RESOURCE_URL}/openai/deployments/{DEPLOYMENT}/embeddings?api-version={API_VERSION}"
    out: list[list[float]] = []
    B = 64
    for i in range(0, len(texts), B):
        chunk = texts[i:i + B]
        body = json.dumps({"input": chunk}).encode("utf-8")
        for attempt in range(6):
            try:
                req = urllib.request.Request(
                    url, data=body, method="POST",
                    headers={"api-key": api_key, "Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = json.loads(r.read())
                rows = [d["embedding"] for d in sorted(data["data"], key=lambda d: d["index"])]
                out.extend(rows)
                break
            except urllib.error.HTTPError as e:
                if e.code == 429 and attempt < 5:
                    wait = 2 ** attempt
                    print(f"  429 rate-limited, sleeping {wait}s")
                    time.sleep(wait)
                    continue
                raise
    return np.asarray(out, dtype=np.float32)


def l2norm(a: np.ndarray) -> np.ndarray:
    return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)


def embed_cached(texts: list[str], cache_path: Path, api_key: str) -> dict[str, np.ndarray]:
    """Embed a list of texts, caching vectors by content hash. Returns {hash: vec}."""
    cache: dict[str, np.ndarray] = {}
    if BASELINE_CACHE.exists():
        z = np.load(BASELINE_CACHE, allow_pickle=True)
        for k in z.files:
            cache[k] = z[k]
    if cache_path.exists():
        z = np.load(cache_path, allow_pickle=True)
        for k in z.files:
            cache[k] = z[k]
    wanted = {sha(t): t for t in texts}
    missing = [t for h, t in wanted.items() if h not in cache]
    if missing:
        print(f"  embedding {len(missing)} new texts via Azure {DEPLOYMENT} ...")
        vecs = l2norm(azure_embed(missing, api_key))
        for t, v in zip(missing, vecs):
            cache[sha(t)] = v.astype(np.float32)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, **cache)
    else:
        print("  all texts already cached")
    return {h: cache[h] for h in wanted}


# ---------------------------------------------------------------------------
# Rank-percentile normalization within the discovered corpus
# ---------------------------------------------------------------------------
def rank_percentile(values: list[float]) -> list[float]:
    n = len(values)
    if n <= 1:
        return [0.5] * n
    order = sorted(values)
    pct = []
    for v in values:
        less = sum(1 for x in order if x < v)
        eq = sum(1 for x in order if x == v)
        pct.append((less + 0.5 * eq) / n)
    return pct


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


# ---------------------------------------------------------------------------
# Projection helpers
# ---------------------------------------------------------------------------
def role_recency(job: dict[str, Any]) -> float:
    if job.get("is_current"):
        return 1.00
    mi = months_between(parse_date(job.get("end_date")), REF_DATE)
    if mi is None:
        return 0.84
    if mi <= 18:
        return 0.96
    if mi <= 36:
        return 0.88
    if mi <= 60:
        return 0.78
    return 0.66


def duration_factor(months: int) -> float:
    return min(1.0, (max(months, 1) / 48.0) ** 0.5)


# URSI V2 title-coherence veto threshold. Title coherence is a percentile in
# [0,1]; anything at or above this is treated as a coherent technical title (no
# effect), and only the genuinely non-technical tail below it is dampened.
TITLE_VETO = 0.30


def evidence_gate(career_evidence_semantic: float) -> float:
    if career_evidence_semantic >= 0.72:
        return 1.00
    if career_evidence_semantic >= 0.55:
        return 0.82
    if career_evidence_semantic >= 0.40:
        return 0.58
    return 0.28


def compute_role_evidence(desc_vecs: np.ndarray, pos_vecs: np.ndarray,
                          neg_vecs: np.ndarray, weights: np.ndarray,
                          neg_lambda: float = ROLE_NEG_LAMBDA) -> tuple[np.ndarray, np.ndarray]:
    """Contrastive multi-anchor role evidence, percentile-normalized in-corpus.

    Returns (role_evidence[D], contrast[D,F]). Reused by the validation suite for
    leave-one-anchor-out stability (pass a subset of families).
    """
    pos_cos = desc_vecs @ pos_vecs.T
    neg_cos = desc_vecs @ neg_vecs.T
    contrast = pos_cos - neg_lambda * neg_cos
    D, F = contrast.shape
    w = weights / weights.sum()
    contrast_pct = np.zeros((D, F), dtype=np.float32)
    pos_pct = np.zeros((D, F), dtype=np.float32)
    for j in range(F):
        contrast_pct[:, j] = rank_percentile(list(contrast[:, j]))
        pos_pct[:, j] = rank_percentile(list(pos_cos[:, j]))
    blend = 0.88 * (contrast_pct @ w) + 0.12 * (pos_pct @ w)
    role_evidence = np.array(rank_percentile(list(blend)), dtype=np.float32)
    return role_evidence, contrast


def project_candidate(career: list[dict[str, Any]], current_title_h: str,
                      desc_evidence: dict[str, float], title_coh: dict[str, float]) -> dict[str, Any]:
    """Project role-level evidence to one candidate (chronology + title coherence)."""
    role_values: list[float] = []
    recent_values: list[float] = []
    evid_list: list[float] = []
    dur_weights: list[int] = []
    strong_docs: set[str] = set()
    sem_rel_months = 0
    best_doc, best_rv = "", -1.0
    for r in career:
        h = sha(norm_text(r.get("description")))
        ev = desc_evidence.get(h, 0.0)
        tc = title_coh.get(sha(norm_text(r.get("title")).lower()), 0.5)
        rec = role_recency(r)
        dur = int(r.get("duration_months") or 0)
        # URSI V2: title is a one-sided *contradiction veto*, not a continuous
        # contributor. In this corpus role descriptions are sampled with an
        # in-domain title pool, so the title percentile carries ~no signal once
        # the description is fixed (measured: identical strongest template spans
        # title_coh 0.59-0.97 -> a 13% per-role swing on byte-identical work).
        # We therefore only dampen a role whose title is a genuine non-technical
        # contradiction (bottom of the coherence distribution); all technical
        # titles are treated as equal.
        title_guard = 1.0 if tc >= TITLE_VETO else 0.70
        rv = ev * rec * (0.70 + 0.30 * duration_factor(dur)) * title_guard
        role_values.append(rv)
        evid_list.append(ev)
        dur_weights.append(min(dur, 60))
        if rv > best_rv:
            best_rv, best_doc = rv, h
        if r.get("is_current") or (months_between(parse_date(r.get("end_date")), REF_DATE) or 999) <= 18:
            recent_values.append(rv)
        if ev >= 0.62 and tc >= TITLE_VETO:
            sem_rel_months += max(0, dur)
        if ev >= 0.70:
            strong_docs.add(h)  # distinct docs only -> repeated identical roles don't inflate breadth

    if role_values:
        best_role = max(role_values)
        cur_recent = max(recent_values) if recent_values else 0.85 * best_role
        tw = sum(dur_weights) or 1
        dwa = sum(e * w_ for e, w_ in zip(evid_list, dur_weights)) / tw
        breadth = min(0.08, 0.02 * len(strong_docs))
        ces = clamp(0.58 * best_role + 0.25 * cur_recent + 0.17 * dwa + breadth)
    else:
        best_role = cur_recent = dwa = breadth = 0.0
        ces = 0.0
    return {
        "career_evidence_semantic": ces,
        "evidence_gate": evidence_gate(ces),
        "semantic_relevant_months": sem_rel_months,
        "semantic_relevant_experience": clamp(sem_rel_months / 54.0),
        "best_role_evidence": best_role,
        "current_or_recent_best": cur_recent,
        "duration_weighted_evidence": dwa,
        "role_evidence_breadth": breadth,
        "current_title_coherence": title_coh.get(current_title_h, 0.5),
        "best_role_doc_id": best_doc,
    }


def project_candidate_skills(
    skills: list[dict[str, Any]],
    skill_evidence: dict[str, float],
) -> dict[str, Any]:
    scored: list[tuple[float, float, str]] = []
    for skill in skills:
        name = norm_text(skill.get("name"))
        if not name:
            continue
        h = sha(name.lower())
        ev = skill_evidence.get(h)
        if ev is None:
            continue
        prof = norm_text(skill.get("proficiency")).lower()
        months = int(skill.get("duration_months") or 0)
        endorsements = int(skill.get("endorsements") or 0)
        weight = 0.10
        if prof == "expert":
            weight += 0.18
        elif prof == "advanced":
            weight += 0.12
        elif prof == "intermediate":
            weight += 0.06
        weight += min(0.12, months / 72.0 * 0.12)
        weight += min(0.06, endorsements / 60.0 * 0.06)
        scored.append((ev, weight, name))
    if not scored:
        return {
            "skill_semantic_evidence": 0.0,
            "strongest_skill": "",
            "strongest_skill_evidence": 0.0,
            "skill_count": 0,
        }
    scored.sort(reverse=True)
    top = scored[:8]
    den = sum(weight for _ev, weight, _name in top) or 1.0
    weighted = sum(ev * weight for ev, weight, _name in top) / den
    best = top[0]
    return {
        "skill_semantic_evidence": weighted,
        "strongest_skill": best[2],
        "strongest_skill_evidence": best[0],
        "skill_count": len(scored),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--variant", choices=sorted(k for k in FL_VARIANT_ANCHORS if k != "baseline"), default="fl_e")
    ap.add_argument("--anchor-pack", type=Path, default=None)
    ap.add_argument("--write-anchor-packs", type=Path, default=DEFAULT_OUT_ROOT / "anchor_packs")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    args = ap.parse_args()

    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "AZURE_OPENAI_API_KEY not set. Export it first, e.g.:\n"
            "  export AZURE_OPENAI_API_KEY=\"your-azure-openai-key\"\n"
            "This precompute step is network-allowed (see submission_spec.md); "
            "the official ranker (rank_ursi_fl.py) needs no key and no network."
        )
    if not AZURE_RESOURCE_URL:
        raise SystemExit(
            "AZURE_OPENAI_RESOURCE_URL not set. Export it first, e.g.:\n"
            "  export AZURE_OPENAI_RESOURCE_URL=\"https://<your-resource>.openai.azure.com\"\n"
            "Only needed to rebuild artifacts from scratch -- the committed "
            "artifacts/role_semantic_index_fl/fl_e/ directory already has everything "
            "rank_ursi_fl.py needs, so this step can be skipped entirely."
        )

    if args.write_anchor_packs:
        write_builtin_anchor_packs(args.write_anchor_packs)
    if args.anchor_pack:
        variant_name, role_anchors = load_anchor_pack(args.anchor_pack)
    else:
        variant_name = args.variant
        role_anchors = FL_VARIANT_ANCHORS[variant_name]
    if args.out_dir is None:
        args.out_dir = DEFAULT_OUT_ROOT / variant_name
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Discover unique role descriptions & titles -------------------
    print("Discovering unique role documents, summaries, titles, and skills ...")
    desc_by_hash: dict[str, str] = {}
    desc_count: dict[str, int] = {}
    title_by_hash: dict[str, str] = {}
    summary_by_hash: dict[str, str] = {}
    summary_count: dict[str, int] = {}
    skill_by_hash: dict[str, str] = {}
    skill_count: dict[str, int] = {}
    n_cand = 0
    n_roles = 0
    with args.candidates.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            n_cand += 1
            c = json.loads(line)
            ct = norm_text(c.get("profile", {}).get("current_title"))
            if ct:
                title_by_hash.setdefault(sha(ct.lower()), ct)
            summary = normalize_numbers(c.get("profile", {}).get("summary"))
            if summary:
                sh = sha(summary)
                summary_by_hash.setdefault(sh, summary)
                summary_count[sh] = summary_count.get(sh, 0) + 1
            seen_skills: set[str] = set()
            for skill in c.get("skills", []):
                skill_name = norm_text(skill.get("name"))
                if not skill_name:
                    continue
                skill_hash = sha(skill_name.lower())
                skill_by_hash.setdefault(skill_hash, skill_name)
                seen_skills.add(skill_hash)
            for skill_hash in seen_skills:
                skill_count[skill_hash] = skill_count.get(skill_hash, 0) + 1
            for r in c.get("career_history", []):
                n_roles += 1
                d = norm_text(r.get("description"))
                h = sha(d)
                desc_by_hash.setdefault(h, d)
                desc_count[h] = desc_count.get(h, 0) + 1
                t = norm_text(r.get("title"))
                if t:
                    title_by_hash.setdefault(sha(t.lower()), t)
    print(
        f"  candidates={n_cand} roles={n_roles} unique_desc={len(desc_by_hash)} "
        f"unique_titles={len(title_by_hash)} unique_summaries={len(summary_by_hash)} "
        f"unique_skills={len(skill_by_hash)}"
    )

    # ---- 2. Embed anchors + unique docs ----------------------------------
    anchor_pos = [a.positive for a in role_anchors]
    anchor_neg = [a.negative for a in role_anchors]
    desc_hashes = list(desc_by_hash)
    desc_texts = [desc_by_hash[h] for h in desc_hashes]
    title_hashes = list(title_by_hash)
    title_texts = [title_by_hash[h] for h in title_hashes]
    summary_hashes = list(summary_by_hash)
    summary_texts = [summary_by_hash[h] for h in summary_hashes]
    skill_hashes = list(skill_by_hash)
    skill_texts = [f"Skill evidence: {skill_by_hash[h]}" for h in skill_hashes]

    all_texts = (
        anchor_pos + anchor_neg
        + [TITLE_POSITIVE_ANCHOR, TITLE_NEGATIVE_ANCHOR]
        + desc_texts + title_texts + summary_texts + skill_texts
    )
    emb = embed_cached(all_texts, args.cache, api_key)

    pos_vecs = np.stack([emb[sha(t)] for t in anchor_pos])           # [F, d]
    neg_vecs = np.stack([emb[sha(t)] for t in anchor_neg])           # [F, d]
    title_pos = emb[sha(TITLE_POSITIVE_ANCHOR)]
    title_neg = emb[sha(TITLE_NEGATIVE_ANCHOR)]
    desc_vecs = np.stack([emb[sha(t)] for t in desc_texts])          # [D, d]
    title_vecs = np.stack([emb[sha(t)] for t in title_texts])        # [T, d]
    summary_vecs = np.stack([emb[sha(t)] for t in summary_texts])    # [S, d]
    skill_vecs = np.stack([emb[sha(t)] for t in skill_texts])        # [K, d]

    weights = np.array([a.weight for a in role_anchors], dtype=np.float32)

    # ---- 3. Per-description role_semantic_evidence -----------------------
    role_evidence, contrast = compute_role_evidence(desc_vecs, pos_vecs, neg_vecs, weights)
    F = contrast.shape[1]
    desc_evidence = {h: float(role_evidence[i]) for i, h in enumerate(desc_hashes)}
    desc_top_family = {
        h: role_anchors[int(np.argmax(contrast[i]))].name for i, h in enumerate(desc_hashes)
    }

    # Summary and skill scores are diagnostic-only. They use the same Azure model
    # and FL anchors, but the official FL ranker does not let them boost candidates.
    summary_evidence, summary_contrast = compute_role_evidence(
        summary_vecs, pos_vecs, neg_vecs, weights
    )
    summary_evidence_by_hash = {
        h: float(summary_evidence[i]) for i, h in enumerate(summary_hashes)
    }
    summary_top_family = {
        h: role_anchors[int(np.argmax(summary_contrast[i]))].name
        for i, h in enumerate(summary_hashes)
    }
    skill_evidence, skill_contrast = compute_role_evidence(
        skill_vecs, pos_vecs, neg_vecs, weights
    )
    skill_evidence_by_hash = {
        h: float(skill_evidence[i]) for i, h in enumerate(skill_hashes)
    }
    skill_top_family = {
        h: role_anchors[int(np.argmax(skill_contrast[i]))].name
        for i, h in enumerate(skill_hashes)
    }

    # ---- 4. Per-title coherence ------------------------------------------
    t_pos = title_vecs @ title_pos
    t_neg = title_vecs @ title_neg
    t_contrast = t_pos - TITLE_NEG_LAMBDA * t_neg
    t_pct = rank_percentile(list(t_contrast))
    title_coh = {h: float(t_pct[i]) for i, h in enumerate(title_hashes)}

    # ---- 5. Write unique role catalog ------------------------------------
    role_csv = args.out_dir / "unique_role_scores.csv"
    with role_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["role_doc_id", "count_roles", "role_semantic_evidence",
                    "top_anchor_family"] + [f"contrast_{a.name}" for a in role_anchors]
                   + ["preview"])
        for i, h in enumerate(desc_hashes):
            w.writerow([h, desc_count[h], f"{desc_evidence[h]:.6f}", desc_top_family[h]]
                       + [f"{contrast[i, j]:.4f}" for j in range(F)]
                       + [desc_by_hash[h][:160]])

    summary_csv = args.out_dir / "summary_template_scores.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "summary_doc_id", "candidate_count", "summary_semantic_evidence",
            "top_anchor_family", "preview",
            "summary_ai_curiosity_flag", "summary_no_professional_ml_flag",
            "summary_search_retrieval_claim_flag", "summary_transition_claim_flag",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for h in summary_hashes:
            flags = summary_flags(summary_by_hash[h])
            w.writerow({
                "summary_doc_id": h,
                "candidate_count": summary_count[h],
                "summary_semantic_evidence": f"{summary_evidence_by_hash[h]:.6f}",
                "top_anchor_family": summary_top_family[h],
                "preview": summary_by_hash[h][:240],
                **{key: bool_text(value) for key, value in flags.items()},
            })

    skill_csv = args.out_dir / "skill_scores.csv"
    with skill_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "skill_doc_id", "skill", "count_candidates", "skill_semantic_evidence",
            "top_anchor_family",
        ])
        for h in skill_hashes:
            w.writerow([
                h,
                skill_by_hash[h],
                skill_count.get(h, 0),
                f"{skill_evidence_by_hash[h]:.6f}",
                skill_top_family[h],
            ])

    # ---- 6. Project to candidates ----------------------------------------
    print("Projecting role evidence to candidates ...")
    proj_csv = args.out_dir / "candidate_role_projection.csv"
    fields = [
        "candidate_id", "career_evidence_semantic", "evidence_gate",
        "semantic_relevant_months", "semantic_relevant_experience",
        "best_role_evidence", "current_or_recent_best", "duration_weighted_evidence",
        "role_evidence_breadth", "current_title_coherence", "best_role_doc_id",
    ]
    summary_fields = [
        "candidate_id", "summary_doc_id", "summary_semantic_evidence",
        "summary_ai_curiosity_flag", "summary_no_professional_ml_flag",
        "summary_search_retrieval_claim_flag", "summary_transition_claim_flag",
    ]
    skill_fields = [
        "candidate_id", "skill_semantic_evidence", "strongest_skill",
        "strongest_skill_evidence", "skill_count",
    ]
    n_written = 0
    summary_proj_csv = args.out_dir / "candidate_summary_projection.csv"
    skill_proj_csv = args.out_dir / "candidate_skill_projection.csv"
    with (
        args.candidates.open(encoding="utf-8") as f,
        proj_csv.open("w", newline="", encoding="utf-8") as out,
        summary_proj_csv.open("w", newline="", encoding="utf-8") as summary_out,
        skill_proj_csv.open("w", newline="", encoding="utf-8") as skill_out,
    ):
        w = csv.DictWriter(out, fieldnames=fields)
        w.writeheader()
        summary_writer = csv.DictWriter(summary_out, fieldnames=summary_fields)
        summary_writer.writeheader()
        skill_writer = csv.DictWriter(skill_out, fieldnames=skill_fields)
        skill_writer.writeheader()
        for line in f:
            if not line.strip():
                continue
            c = json.loads(line)
            ct_h = sha(norm_text(c.get("profile", {}).get("current_title")).lower())
            pj = project_candidate(c.get("career_history", []), ct_h, desc_evidence, title_coh)
            summary = normalize_numbers(c.get("profile", {}).get("summary"))
            sh = sha(summary) if summary else ""
            flags = summary_flags(summary)
            skill_projection = project_candidate_skills(
                c.get("skills", []),
                skill_evidence_by_hash,
            )
            w.writerow({
                "candidate_id": c["candidate_id"],
                "career_evidence_semantic": f"{pj['career_evidence_semantic']:.6f}",
                "evidence_gate": f"{pj['evidence_gate']:.4f}",
                "semantic_relevant_months": pj["semantic_relevant_months"],
                "semantic_relevant_experience": f"{pj['semantic_relevant_experience']:.6f}",
                "best_role_evidence": f"{pj['best_role_evidence']:.6f}",
                "current_or_recent_best": f"{pj['current_or_recent_best']:.6f}",
                "duration_weighted_evidence": f"{pj['duration_weighted_evidence']:.6f}",
                "role_evidence_breadth": f"{pj['role_evidence_breadth']:.6f}",
                "current_title_coherence": f"{pj['current_title_coherence']:.6f}",
                "best_role_doc_id": pj["best_role_doc_id"],
            })
            summary_writer.writerow({
                "candidate_id": c["candidate_id"],
                "summary_doc_id": sh,
                "summary_semantic_evidence": f"{summary_evidence_by_hash.get(sh, 0.0):.6f}",
                **{key: bool_text(value) for key, value in flags.items()},
            })
            skill_writer.writerow({
                "candidate_id": c["candidate_id"],
                "skill_semantic_evidence": f"{skill_projection['skill_semantic_evidence']:.6f}",
                "strongest_skill": skill_projection["strongest_skill"],
                "strongest_skill_evidence": f"{skill_projection['strongest_skill_evidence']:.6f}",
                "skill_count": skill_projection["skill_count"],
            })
            n_written += 1
    print(f"  wrote {n_written} candidate projections -> {proj_csv}")
    print(f"  wrote summary projection -> {summary_proj_csv}")
    print(f"  wrote skill projection -> {skill_proj_csv}")

    # ---- 7. Manifest + anchor pack + validation report -------------------
    (args.out_dir / "jd_anchor_pack.json").write_text(
        json.dumps(anchor_pack_payload(variant_name, role_anchors), indent=2),
        encoding="utf-8",
    )

    manifest = {
        "artifact_name": "role_semantic_index_fl",
        "variant": variant_name,
        "created_at": datetime.now().date().isoformat(),
        "embedding_model": f"azure:{DEPLOYMENT}",
        "embedding_used_only_in_precompute": True,
        "summary_and_skill_embeddings_used_for_diagnostics": True,
        "summary_and_skill_embeddings_used_for_official_ranking": False,
        "manual_template_grades_used_for_scoring": False,
        "template_foreknowledge_used": False,
        "official_ranker_requires_network": False,
        "candidates": n_cand,
        "roles": n_roles,
        "unique_role_docs": len(desc_by_hash),
        "unique_titles": len(title_by_hash),
        "unique_summary_templates": len(summary_by_hash),
        "unique_skills": len(skill_by_hash),
        "normalization": "rank_percentile_within_unique_role_docs",
        "role_neg_lambda": ROLE_NEG_LAMBDA,
        "ref_date": REF_DATE.isoformat(),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # face-validity: top/bottom descriptions by evidence
    ranked = sorted(desc_hashes, key=lambda h: -desc_evidence[h])
    lines = [f"# URSI-FL {variant_name} role-semantic index — validation report", "",
             f"candidates={n_cand} roles={n_roles} unique_role_docs={len(desc_by_hash)} "
             f"unique_titles={len(title_by_hash)} unique_summary_templates={len(summary_by_hash)} "
             f"unique_skills={len(skill_by_hash)}",
             f"embedding_model=azure:{DEPLOYMENT}  role_neg_lambda={ROLE_NEG_LAMBDA}", "",
             "## Top 8 role documents by semantic evidence (face validity)"]
    for h in ranked[:8]:
        lines.append(f"- {desc_evidence[h]:.3f} [{desc_top_family[h]}] (n={desc_count[h]}) {desc_by_hash[h][:120]}")
    lines += ["", "## Bottom 8 role documents (should be non-technical)"]
    for h in ranked[-8:]:
        lines.append(f"- {desc_evidence[h]:.3f} [{desc_top_family[h]}] (n={desc_count[h]}) {desc_by_hash[h][:120]}")
    ranked_summaries = sorted(summary_hashes, key=lambda h: -summary_evidence_by_hash[h])
    lines += ["", "## Top 6 summary templates by diagnostic semantic evidence"]
    for h in ranked_summaries[:6]:
        lines.append(
            f"- {summary_evidence_by_hash[h]:.3f} [{summary_top_family[h]}] "
            f"(n={summary_count[h]}) {summary_by_hash[h][:140]}"
        )
    ranked_skills = sorted(skill_hashes, key=lambda h: -skill_evidence_by_hash[h])
    lines += ["", "## Top 12 skills by diagnostic semantic evidence"]
    for h in ranked_skills[:12]:
        lines.append(
            f"- {skill_evidence_by_hash[h]:.3f} [{skill_top_family[h]}] "
            f"(n={skill_count.get(h, 0)}) {skill_by_hash[h]}"
        )
    (args.out_dir / "validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Top role docs:")
    for h in ranked[:6]:
        print(f"  {desc_evidence[h]:.3f} [{desc_top_family[h]:>22}] {desc_by_hash[h][:80]}")
    print("Bottom role docs:")
    for h in ranked[-6:]:
        print(f"  {desc_evidence[h]:.3f} [{desc_top_family[h]:>22}] {desc_by_hash[h][:80]}")


if __name__ == "__main__":
    main()
