# URSI V3.2 — Changelog & Rationale

> **Historical iteration record.** This document is kept for provenance —
> URSI-FL (variant FL-E, see `METHODOLOGY_URSI_FL.md`) is the submitted system.
> The `rank_ursi.py` / `precompute_role_semantic_index.py` scripts referenced
> below are the pre-FL baseline and are **not included in this repository**;
> URSI-FL's official path is `rank_ursi_fl.py` / `precompute_role_semantic_index_fl.py`.

This document records how the official URSI ranker evolved **V1 → V2 → V3 →
V3.1 → V3.2**, why each change was made, and the evidence that justified it. It complements
`METHODOLOGY_URSI.md` (the authoritative end-to-end methodology); this file is the
*iteration history* — what we changed, what we measured, and what we deliberately
chose **not** to do.

> One-line summary of V3.2: keep URSI's semantic evidence spine, stop letting a
> noisy in-domain **job title** reorder candidates (V2), and restore the two
> **JD-named, negative-only** business gates the URSI path had dropped — an
> entirely-consulting career and serial job-hopping (V3), then stop double-counting
> duration in the final best-role slot by using raw role-document evidence (V3.1),
> then gate that raw uplift by relevant-month coverage so a thin relevant spike
> does not get full raw peak credit (V3.2).

---

## 0. The objective (anchor for every decision)

Rank the top 100 candidates for **Redrob's Senior AI Engineer, Founding Team**
role. The JD is explicit about both what it wants and what it rejects:

- **Wants:** production search / ranking / retrieval / recommendation evidence,
  shipped end-to-end to real users, hands-on and recent, in/near Noida–Pune (or
  relocatable), reachable.
- **Explicitly does NOT want:** keyword-stuffed skill sections, title-without-work,
  research-only, framework/wrapper-only, **entirely-consulting careers**,
  **title-chasers who switch every ~1.5 years**, honeypots.

The hidden score is top-heavy (P@5 / P@10 are tie-breakers), so correctness at
ranks 1–10 and 1–50 matters most, and a regression there is expensive.

---

## 1. Two data facts that drove everything

Both were verified directly against `candidates.jsonl` (100,000 candidates).

### 1a. Role descriptions are templated; titles are sampled from an in-domain pool
There are only **44 distinct role-description templates** in the whole corpus.
Each template is paired with a *pool* of titles from its own domain. The single
strongest, on-target template (the LTR e-commerce-search paragraph,
`role_semantic_evidence = 0.92`) appears 78 times across **7 distinct titles, all
relevant**:

```
AI Engineer · Applied ML Engineer · Machine Learning Engineer · NLP Engineer
Recommendation Systems Engineer · Search Engineer · Senior Data Scientist
```

**Implication:** once the description is fixed, the title carries ~no signal —
which label the generator stamped on identical work is a coin-flip.

### 1b. The URSI official path had silently dropped JD-named business gates
The cruder `rank.py` baseline modeled `consulting_factor` and `jobhop_factor`.
The URSI rewrite (which fixed real semantic bugs) did **not** carry these over —
its score was `base_fit · f_loc · f_beh · f_exp · f_git · f_coh`, with no
consulting and no tenure gate, even though the JD names both as disqualifiers.

---

## 2. V1 → V2 — Title becomes a veto, not a continuous score term

### Problem
V1 used job title as a **continuous contributor** in two places:
- per-role multiplier `(0.70 + 0.30 · title_coherence)` in the projection, and
- a direct `0.10 · current_title_coherence` term in `base_fit`.

Given fact **1a**, that injected pure label noise into the ordering.

### Evidence (measured)
For the **identical** strongest LTR template, the 7 in-domain titles span
`title_coherence` **0.594 → 0.969**, producing a per-role multiplier swing of
**0.878 → 0.991 (≈12.8%) on byte-identical work** — and the squarely-relevant
"Recommendation Systems Engineer" (0.594) and "Senior Data Scientist" (0.635)
were docked hardest.

| Title (same LTR description) | title_coh | per-role multiplier |
|---|---|---|
| Machine Learning Engineer | 0.969 | 0.991 |
| Applied ML Engineer | 0.844 | 0.953 |
| NLP Engineer | 0.823 | 0.947 |
| Search Engineer | 0.781 | 0.934 |
| AI Engineer | 0.656 | 0.897 |
| Senior Data Scientist | 0.635 | 0.891 |
| Recommendation Systems Engineer | 0.594 | 0.878 |

### Change
- **Projection** (`precompute_role_semantic_index.py`): per-role title factor →
  veto-only `title_guard = 1.0 if title_coh ≥ 0.30 else 0.70`; relevant-months
  title gate `≥ 0.45` → `≥ 0.30`.
- **Ranker** (`rank_ursi.py`): removed `0.10 · tcoh` from `base_fit`; reweighted to
  `0.60·ces + 0.25·best_role + 0.15·rel`. V3.1 later makes that `best_role`
  term the raw role-document evidence rather than the projected duration value.
- **Kept:** the `f_coh < 0.30` contradiction veto (strong evidence under a
  genuinely non-technical *current* title = stale/planted profile). Title can
  still veto; it can no longer reorder coherent technical titles.

### Why not the alternative (a pairwise title↔description matrix)
A hand-authored compatibility matrix (e.g. "NLP Engineer + LTR = 0.70") would
assign different scores to the 7 *equally-valid in-domain* titles above — a
0.65→1.00 spread on identical work — i.e. it amplifies the noise of fact **1a**
and would demote genuinely strong candidates for a generation coin-flip. Rejected
on evidence.

### Verified effect (V1 → V2)
- **Entered top-100:** mean `title_coh ≈ 0.611` (strong evidence V1 was
  suppressing). **Dropped:** mean `title_coh ≈ 0.906` (riding the old title bonus
  on comparable/weaker evidence).
- Flagship: **CAND_0086151** (`ces 0.877`, `title_coh 0.594`) rose **69 → 36**;
  **CAND_0001610** (`ces 0.706`, `title_coh 0.969`) fell out of the top-100.

---

## 3. V2 → V3 — Restore the JD's negative-only business gates

### Principle: negative weighting only
Both new gates are multiplicative **≤ 1.0** — they can only demote candidates the
JD explicitly rejects, never lift a weak profile. The semantic evidence spine
still decides who is *good*; the gates only remove the disqualified. We add no
positive "product-company boost" — the JD gives a clean negative list, not a
trustworthy positive label, and a whitelist would drift toward keyword-stuffing.

```
score = base_fit · f_loc · f_beh · f_exp · f_git · f_coh · f_cons · f_ten
```

### `f_cons` — entirely-consulting career
- `0.40` if **every** role's company is on the JD's IT-services/consulting list
  (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini, Tech Mahindra, HCL, …).
- `1.0` otherwise — including the JD's explicit carve-out: *currently* at a
  services firm **but with any prior product-company role is fine**.
- Structured gate on the **company field** (not career-text keyword matching),
  consistent with `f_loc`'s city list.

### `f_ten` — serial job-hopping
- `0.90` for **4+ roles** with mean *completed* tenure `< 16` months; `0.82` if
  `< 12`. Deliberately mild and gated at 4+ roles so one or two legitimately
  short stints are not punished.

### Verified effect (V2 → V3) — validator: "Submission is valid."
- **3 consulting-only candidates removed** from the top-100:
  CAND_0045141 (HCL/HCL, was 46), CAND_0091712 (HCL, was 56),
  CAND_0089012 (TCS, was 58). **0 consulting-only remain** in the top-100.
- **Carve-out confirmed working** — currently-at-services-but-prior-product
  candidates were spared and even rose: Tech Mahindra→Freshworks→Paytm (61→58),
  Wipro→CRED (67→64), HCL→Yellow.ai (81→78).
- **Top-20 unchanged (0/20 moved).** Gates only touched lower ranks; 3 next-best
  candidates backfilled ranks 98–100.
- **Gates are live, not dead code:** `f_cons < 1` fires on **9,745** pool
  candidates, `f_ten < 1` on **596** — and **0 of each** survive into the
  V3 top-100.

---

## 3.5. V3 → V3.1 — Use raw best-role evidence in the final base fit

### Problem
`candidate_role_projection.csv`'s `best_role_evidence` is a projected role value:
raw role evidence multiplied by recency, duration, and title guard. That is right
inside `career_evidence_semantic`, because chronology matters. But V3 then reused
that projected value as the independent `0.25·best_role` term in `base_fit`,
which double-counted duration for long stints.

This surfaced at rank 1: **CAND_0093193** had two long roles with the same
plain-language matching-layer template and edged out **CAND_0081846** by only
0.0015, even though the Razorpay candidate's strongest role document is the most
JD-isomorphic one in the corpus: recruiter-facing BM25+dense retrieval, BGE,
FAISS, LLM re-ranking, NDCG/MRR, A/B metrics, and recruiter feedback.

### Change
The official ranker now loads `unique_role_scores.csv` and maps
`best_role_doc_id -> role_semantic_evidence`. The base fit is:

```
base_fit = 0.60·career_evidence_semantic
         + 0.25·raw_best_role_evidence
         + 0.15·relevant_experience
```

This keeps duration, recency, and repeated relevant months inside
`career_evidence_semantic`, while the best-role slot measures the raw semantic
quality of the candidate's strongest role document. No manual grades or keyword
lists are used; the new term comes from the existing URSI semantic artifact and
falls back to the projected value if the role-score file is unavailable.

### Verified effect (V3 → V3.1)
- **CAND_0081846** moves **2 → 1**.
- **CAND_0093193** moves **1 → 2**; it remains a very strong fit, not a reject.
- **CAND_0086022** remains rank 3.
- **CAND_0066999** moves **4 → 5**; **CAND_0061257** remains rank 8.
- Top overlap: **9/10**, **19/20**, **47/50**, **96/100**.
- The output validates with `validate_submission.py`.

Rejected alternative: lowering the duration cap in projection also fixed the
top-two ordering, but it changes precompute semantics and saturates some top
scores. The raw-best-role correction is smaller, ranker-only, and directly fixes
the double-counting mechanism.

---

## 3.6. V3.1 → V3.2 — Gate raw best-role uplift by career coverage

### Problem
V3.1 correctly separated raw role-document quality from projected chronology, but
the raw best-role term was still a fixed 25% peak feature. That left a calibration
risk: one excellent role document inside a mostly irrelevant career could receive
full raw peak credit even though `career_evidence_semantic` and relevant months
were already saying the career was thinner.

### Change
V3.2 keeps the V3.1 raw-best correction but interpolates between projected best
and raw best using relevant-month coverage:

```
relevance_share = semantic_relevant_months / max(total_role_months, 1)
raw_coverage = min(1.0, semantic_relevant_months / 54.0, relevance_share / 0.60)
effective_best = projected_best + raw_coverage · (raw_best − projected_best)

base_fit = 0.60·career_evidence_semantic
         + 0.25·effective_best
         + 0.15·relevant_experience
```

Full raw peak credit now requires roughly 4.5 years of relevant role history and
relevant work making up most of the candidate's role months. This is still
ranker-only: no precompute change, no manual template grades, no career-text
keyword gates, no skills/summary scoring, and no title boost.

### Verified effect (V3.1 → V3.2)
- **CAND_0081846** remains rank 1; **CAND_0093193** remains rank 2; **CAND_0086022** remains rank 3.
- Top overlap: **10/10**, **20/20**, **50/50**, **100/100** by candidate membership.
- Only 13 top-100 candidates change rank, and the largest movement is 3 places.
- **CAND_0041669** remains rank 14 but its score is damped because only 43 / 95 role months are relevant (coverage 0.754, relevance share 0.453).
- **CAND_0084819**, the clearer thin-spike pattern, remains outside the top 100 and is strongly damped (20 / 53 relevant months, coverage 0.370).

---

## 4. JD → URSI V3.2 coverage map

| JD signal | In JD? | URSI V3.2 mechanism | Status |
|---|---|---|---|
| Production ranking/search/recsys/retrieval evidence | core | `career_evidence_semantic` (7 JD anchors) | ✅ |
| Availability (active / response / notice / open-to-work) | yes | `f_beh` | ✅ |
| Location / no visa sponsorship | yes | `f_loc` | ✅ |
| Seniority 5–9y (soft) | yes | `f_exp` | ✅ |
| Honeypots / tech anachronism | spec | `honeypot_reasons` → 0 | ✅ |
| Non-technical title contradiction | yes | `f_coh` veto (right-sized in V2) | ✅ |
| Entirely-consulting career | **yes** | **`f_cons` (V3)** | ✅ restored |
| Title-chasing / job-hopping | **yes** | **`f_ten` (V3)** | ✅ restored |
| Keyword-stuffed skills/summary | trap | never scored (by construction) | ✅ |

---

## 5. Caveats & reversibility

- **Unmeasurable against the hidden score.** The leaderboard is hidden; V3.2 is more
  *JD-faithful* and *internally consistent with the data*, but we cannot claim a
  proven NDCG gain. Every change above is justified by the JD text + verified data
  behavior, not by leaderboard fitting.
- **Tenure gate is conservative (4+ roles).** Borderline 3-role short-tenure
  candidates (e.g. at ranks 15, 18) are intentionally **not** penalized. Tightening
  to 3+ roles is a one-line change if desired.
- **Fully reversible.** Each version's CSV is preserved:
  `submission_ursi_v1.csv`, `submission_ursi_v2.csv`, `submission_ursi_v3.csv`,
  `submission_ursi_v31.csv`, and `submission_ursi_v32.csv` (official = `submission.csv` = V3.2). V1 projection saved at
  `artifacts/role_semantic_index/candidate_role_projection_v1.csv`.

---

## 6. Reproduction

```bash
# 1) Precompute (network allowed; cached -> offline on re-run; writes frozen CSVs)
export AZURE_OPENAI_API_KEY="your-azure-openai-key"
python3 precompute_role_semantic_index.py \
  --candidates candidates.jsonl --out-dir artifacts/role_semantic_index

# 2) Official rank (offline, CPU-only, deterministic) -> V3.2 submission
python3 rank_ursi.py \
  --candidates candidates.jsonl \
  --role-projection artifacts/role_semantic_index/candidate_role_projection.csv \
  --out submission.csv

# 3) Validate
python3 validate_submission.py submission.csv
```

Only the **projection** step touches the network (and only on a cold cache); the
official ranking step is fully offline. V2 changed the projection math + ranker
weights (precompute re-run required); V3 changed only ranker-side structured gates
(no precompute needed); V3.1 changed only the ranker-side best-role term
(no precompute needed); V3.2 gates that term by coverage in the ranker only
(no precompute needed).

---

## 7. V3.2 top-10 snapshot

Source: `submission_ursi_v32.csv` (official = `submission.csv`).

| Rank | Candidate | Score | Title | Company | Location | YoE | Role evidence | Relevant | Best evidence (template) |
|---:|---|---:|---|---|---|---:|---:|---:|---|
| 1 | CAND_0081846 | 0.971 | Lead AI Engineer | Razorpay | Jaipur | 6.7 | 0.96 | 6.6y | RAG ranking pipeline 50M+ queries/mo |
| 2 | CAND_0093193 | 0.957 | Senior ML Engineer | Niramai | Bangalore | 7.9 | 0.98 | 7.8y | Plain-language matching layer |
| 3 | CAND_0086022 | 0.953 | Senior Applied Scientist | Sarvam AI | Kolkata | 5.3 | 0.96 | 5.2y | RAG ranking pipeline 50M+ queries/mo |
| 4 | CAND_0006567 | 0.925 | Senior AI Engineer | Meta | Noida | 7.9 | 0.92 | 7.8y | Plain-language matching layer |
| 5 | CAND_0066999 | 0.919 | Rec Sys Engineer | Microsoft | Delhi | 5.9 | 0.93 | 5.8y | E-commerce LTR search ranking layer |
| 6 | CAND_0018549 | 0.913 | Rec Sys Engineer | Uber | Coimbatore | 6.8 | 0.92 | 6.8y | E-commerce LTR search ranking layer |
| 7 | CAND_0078492 | 0.909 | Rec Sys Engineer | Verloop.io | Kochi | 5.1 | 0.91 | 5.1y | E-commerce LTR search ranking layer |
| 8 | CAND_0061257 | 0.890 | Staff ML Engineer | LinkedIn | Noida | 8.0 | 0.88 | 7.9y | Flagship product ranking layer design |
| 9 | CAND_0077337 | 0.889 | Staff ML Engineer | Paytm | Kochi | 7.0 | 0.87 | 6.9y | Embedding search migration, candidate corpus |
| 10 | CAND_0076163 | 0.877 | NLP Engineer | Ola | Chandigarh | 6.9 | 0.94 | 6.8y | E-commerce LTR search ranking layer |

### Face-validity read

**What looks right**

- Every top-10 profile has **strong role-history evidence** (ces ≥ 0.88) and
  **5+ years of relevant months** except rank 10 (4.0y — still strong evidence,
  slightly below the JD's preferred 5–9 band).
- All are at **product companies** (Niramai, Razorpay, Sarvam, Microsoft, Meta,
  Uber, LinkedIn, Ola, Apple) — none hit `f_cons`.
- Ranks 1–3 are the clearest JD fits: direct production RAG/retrieval/ranking
  evidence at ranks 1 and 3, with the strong plain-language matching profile
  still retained at rank 2.
- Rank 8 (LinkedIn, Noida) is a strong logistics + seniority signal for the role.

**What to watch**

- **Rank 10 (CAND_0076163):** highest career-evidence score in the lower top 10
  (0.94) but
  flagged in reasoning — Chandigarh, not open to relocate. Good evidence, weaker
  hiring logistics.
- **CAND_0008239** moves 10 → 11: only 4.0y experience and a single-company career;
  strong template evidence but thin seniority signal vs the revised top 10.
- **Ranks 5–7 and 10** share the same LTR e-commerce template with different
  in-domain titles — V2 correctly stopped penalizing that title noise; the
  ordering among them is driven by evidence depth, recency, and behavioral gates.

### V3.2 gates on top 10

All ten: `f_cons = 1.0`, `f_ten = 1.0` (no consulting-only careers, no serial
hoppers in this slice).
