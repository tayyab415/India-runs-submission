# URSI-FL — Final Methodology (Submitted System)

**Unified Role Semantic Index, Fixed-Leak anchors, variant FL-E.**
This is the definitive description of the system that produces
`submission_ursi_fl_fl_e.csv`. Iteration history lives in `URSI_FL.md` and
`URSI_V3.md`; the baseline spine write-up is `METHODOLOGY_URSI.md`. This
document is self-contained — a reviewer can evaluate the submitted system from
this file alone.

---

## 1. Problem framing

Rank 100,000 synthetic candidate profiles against the Redrob Senior AI Engineer
(Founding Team) JD and submit the top 100. The hidden ground truth is scored
top-heavy: `0.50·NDCG@10 + 0.30·NDCG@50 + 0.15·MAP + 0.05·P@10` — 80% of the
composite is decided inside the top 50. The design therefore optimizes for
precision at the head, gates hard against the dataset's planted traps
(keyword-stuffed skills, plain-language tier-5s, honeypots, behavioral twins),
and treats the JD's own "read between the lines" section as the specification.

## 2. Data-engineering findings that shaped the design

1. **The corpus is closed.** The 300,171 career-history rows deduplicate to
   **44 unique role-description templates**; titles deduplicate to **48**.
   Discovered by content hashing, not assumed. This collapses the semantic
   problem from "embed 100K resumes" to "score 44 documents once, then project."
2. **Skills and summaries are traps.** The JD says so explicitly ("a candidate
   who has all the AI keywords listed as skills but whose title is Marketing
   Manager is not a fit"). Skills/summaries are therefore **never scored** —
   they are embedded only for false-positive diagnostics (76 summary templates,
   133 skill names).
3. **Titles carry almost no signal beyond contradiction.** Within this corpus,
   titles are sampled from an in-domain pool per role description — the single
   strongest template spans title-coherence 0.59–0.97 across seven
   equally-relevant titles. Title is therefore a one-sided **contradiction
   veto**, never a booster.
4. **Honeypots are structural, not semantic.** ~80 planted impossible profiles
   (spec §7). We detect **95** via pure consistency checks (§6); all catches
   are verifiable impossibilities; **0 appear in our top 100**.

## 3. Architecture — two cleanly separated stages

```
STAGE A (one-time, network allowed)          STAGE B (official, offline)
precompute_role_semantic_index_fl.py         rank_ursi_fl.py
  Azure text-embedding-3-large                 stdlib-only Python (csv/json/re)
  44 role docs + 48 titles + 16 anchors        reads candidates.jsonl +
  → frozen CSV artifacts                       frozen CSV artifacts → top-100 CSV
```

Both stages import `ursi_fl_common.py`, so honeypot/date/anchor logic cannot
diverge. The official ranking step loads **no model, makes no network calls,
uses no third-party packages** — measured **34.9 s wall-clock, 1.78 GB peak
RSS** on the full 100K pool (limits: 5 min, 16 GB). Output is byte-identical
across runs; ties are broken by ascending `candidate_id` (the spec's suggested
deterministic tiebreak).

## 4. Stage A — the semantic spine (precompute)

1. **JD → contrastive multi-anchor query.** Seven JD-derived facets, each with
   a positive concept (what the JD wants built) and a negative concept (the
   look-alike non-fit the JD's trap section warns about), weighted by JD
   emphasis: `retrieval_ops .20, ranking_eval .18, candidate_matching .16,
   semantic_search_quality .15, production_ml .13, matching_finetune_eval .10,
   product_shipper .08`. FL-E uses the fixed-leak anchor pack: a sharpened
   `retrieval_ops` negative (prompt-engineering/demo-chatbot wrappers), an
   expanded `matching_finetune_eval` positive (LoRA/QLoRA/re-ranking/
   distillation), and a strict `production_ml` negative (CV/speech/robotics/
   mechanical as primary domain).
2. **Score each of the 44 role documents once:**
   `contrast = cos(doc, pos) − 0.70·cos(doc, neg)` per facet, rank-percentile
   normalized **within the discovered corpus** per facet, blended
   `0.88·contrast_pct + 0.12·pos_pct` across facets, then percentile-normalized
   again → `role_semantic_evidence ∈ [0,1]`. Percentile normalization within a
   closed corpus is what makes plain-language fits ("owned the ranking layer…")
   score high and non-technical roles collapse toward 0 with **no keyword
   lists**.
3. **Title coherence** is computed the same contrastive way over the 48 unique
   titles (technical-IC positive vs non-technical/adjacent-domain negative).
4. **Project role evidence to each candidate** from their actual career rows:
   - per-role value `rv = evidence × recency × (0.70 + 0.30·√(months/48)) ×
     title_guard`, where recency steps 1.00 (current) → 0.66 (>5y old) and
     `title_guard = 0.70` only when the role's title is a bottom-30%-coherence
     contradiction;
   - `career_evidence_semantic (CES) = 0.58·best_role + 0.25·current_or_recent_best
     + 0.17·duration_weighted_avg + breadth` (breadth = 0.02 per distinct strong
     template, capped 0.08 — repeated identical roles cannot inflate it);
   - `semantic_relevant_months` counts only months in roles with evidence ≥ 0.62
     and a non-contradictory title.

Artifacts frozen in `artifacts/role_semantic_index_fl/fl_e/`: candidate
projection, unique role scores, anchor pack, manifest, diagnostics-only
summary/skill projections, and a determinism repeat-check copy of the
submission.

## 5. Stage B — the official offline ranker

```
base_fit = 0.60·CES + 0.25·effective_best_role + 0.15·relevant_experience
score    = base_fit × f_loc × f_beh × f_work × f_exp × f_git × f_assess
                    × f_coh × f_title_domain × f_cons × f_ten
honeypot ⇒ score = 0
```

`effective_best_role` interpolates from the candidate's *projected* best-role
value toward the *raw* template evidence, gated by coverage
(`min(relevant_months/54, relevance_share/0.60)` capped at 1) — full raw peak
credit requires sustained relevant history, so a short relevant spike cannot
borrow a top template's full score, and duration is not double-counted.

Every multiplicative gate maps to an explicit JD statement:

| Gate | JD basis | Range |
|---|---|---|
| `f_loc` location | India-only (no visa sponsorship); Pune/Noida/NCR/Mumbai/Hyderabad hubs preferred; relocation honored | 0.30–1.00 |
| `f_beh` behavioral | "a perfect-on-paper candidate who hasn't logged in for 6 months and has a 5% response rate is not actually available" — last-active, response rate/time, open-to-work (market-activity calibrated), notice period (JD: sub-30d loved, 30+ raises the bar), interview completion, recent applications, offer acceptance | floor 0.45 |
| `f_work` work mode | remote-preference outside hubs w/o relocation is a mild logistics hit | 0.985–1.00 |
| `f_exp` experience prior | "5–9 years… a range, not a requirement" — mild prior peaking at 6–8y, never a hard cut | 0.85–1.00 |
| `f_git` external validation | "entirely closed-source for 5+ years without external validation" | 0.97 / 1.00 |
| `f_assess` verified skills | Redrob-**verified** assessments on JD skills only (self-claims never scored) | 0.99–1.02 |
| `f_coh` title contradiction | non-technical current title under strong role evidence (planted contradiction) | 0.60–1.00 |
| `f_title_domain` adjacent domain | JD: "primary expertise CV/speech/robotics without NLP/IR" — dampens only marginal-coherence cases | 0.92–1.00 |
| `f_cons` consulting | JD's named list (TCS/Infosys/Wipro/…) — fires only when **every** role is services; one product role spares the candidate, exactly as the JD states | 0.40 / 1.00 |
| `f_ten` tenure | "title-chasers switching every ~1.5 years" — fires only for 4+ roles with low mean completed tenure | 0.82–1.00 |

Nothing in Stage B keyword-matches career text. The only text matching in the
whole official path is the honeypot anachronism table (§6) and the JD's own
named consulting-firm list — both structural, both quoted from public facts or
the JD itself.

## 6. Honeypot and impossibility gates (shared `ursi_fl_common.py`)

A candidate scores 0 if any of these hard structural checks fire:

- a single role longer than the entire stated career (+12mo tolerance);
- claimed experience exceeding time since earliest role start (+36mo tolerance);
- role durations summing implausibly beyond stated experience (+36mo tolerance);
- ≥3 "expert" skills with 0 months of use (the spec's own example);
- **tech anachronisms**: a role whose best-estimate end date precedes the public
  release of a technology named in its description (27-entry factual table:
  GPT-4, ChatGPT, Llama-2/3, Mistral, QLoRA, LangChain, …), matched on word
  boundaries, covering current roles via `role_end_date()`.

Result on the full pool: **95 zeroed, 0 in the top 100**. Audited: every catch
is a verifiable impossibility (e.g., GPT-4 in a role ending 2019); the
"skill duration > total experience" pattern common in thousands of legitimate
profiles is deliberately **not** used.

## 7. Reasoning column (Stage 4 alignment)

Reasoning is generated deterministically *after* scoring and cannot affect
ranks. Each row is exactly the spec's required **1–2 sentences** (spec §2:
"a 1-2 sentence justification"): sentence 1 packs the facts, fit tier, and
grounded quote with semicolons — the spec's own example style — and sentence 2
is the honest concern or behavioral strengths. Each row is assembled from the
candidate's actual data:

- **facts**: years, current title, company, location, role-evidence score,
  relevant years — all read from the profile/artifacts;
- **JD connection**: the candidate's dominant anchor family is rendered in JD
  language ("ranking systems with evaluation rigor (NDCG/MRR/A/B)",
  "production retrieval and search infrastructure", …);
- **grounded evidence**: a direct quote from the candidate's strongest actual
  career-history description (no generation, no paraphrase → no hallucination);
- **honest concern**: the dominant penalty that actually fired for this
  candidate (location/relocation, notice period, open-to-work, response rate,
  offer acceptance, GitHub absence, …), or **behavioral strengths** (response
  rate, notice period, active applications, GitHub) when no concern fired, or
  a neutral middling-availability sentence built from actual signal values —
  every row carries a second sentence;
- **variation**: three clause structures rotated by `candidate_id % 3`
  (deterministic, no RNG), three evidence labels, five anchor phrasings, and
  rank-consistent fit language ("aligns strongly" → "solid" →
  "lighter-weight" → "adjacent") tiered on the same CES that drives the score.

All 100 strings are unique; tone tracks rank by construction because both are
driven by the same underlying score components.

## 8. Validation evidence (all reproducible offline)

| Check | Result |
|---|---|
| Format validator | `Submission is valid.` |
| Reproduction | byte-identical CSV across runs; 34.9 s; 1.78 GB peak |
| Fixed-leak invariant | negative-anchor effect vs manual grade spearman **−0.001** (the leak URSI-FL exists to fix; was +0.43 pre-fix) |
| Template audit (44 docs) | spearman vs manual audit grades **+0.875** (adoption bar +0.869); all tier-5 templates in top ranks [1,3,4,5,14,15] |
| Honeypots | 95 zeroed / 0 in top-100 (independent re-check in `verify_fl_variant.py`) |
| Anchor stability | leave-one-anchor-out Jaccard@100 **0.87–1.00** (weakest: `production_ml`) |
| Sentinels | RD34 holders hold/rise; CV-engineer false-positive sentinel stays excluded |
| Delta vs URSI baseline | top10 9/10, top50 45/50, top100 92/100 overlap — changes concentrated in the low-weight tail |
| Signal accounting | 12/23 Redrob signals used; the 11 unused are individually justified in `URSI_FL.md` (popularity/vanity signals like profile views, connections, endorsements are recruiter-demand echoes, not candidate quality) |

Manual template grades exist **only** as an after-the-fact audit; they are not
an input to any scoring path.

## 9. Reproduction

```bash
# Stage A (one-time, network; only needed to rebuild frozen artifacts)
python3 precompute_role_semantic_index_fl.py \
  --candidates ./candidates.jsonl --variant fl_e \
  --out-dir ./artifacts/role_semantic_index_fl/fl_e

# Stage B (official; offline, CPU-only, deterministic, ~35 s)
python3 rank_ursi_fl.py \
  --candidates ./candidates.jsonl \
  --role-projection ./artifacts/role_semantic_index_fl/fl_e/candidate_role_projection.csv \
  --out ./submission_ursi_fl_fl_e.csv

python3 validate_submission.py submission_ursi_fl_fl_e.csv
```

Environment: Python 3.11 (ranker: stdlib only; precompute additionally needs
`numpy`), macOS/Linux, 8 GB RAM sufficient.

## 10. Honest limitations

1. **The availability bet.** FL-E trades ~7 tail slots (ranks 57–100) from
   cleaner-evidence/weaker-availability profiles toward
   slightly-lighter-evidence/stronger-availability ones. This follows the JD's
   explicit instruction to down-weight unavailable candidates; if the hidden
   ground truth scores pure technical fit only, these tail swaps are marginally
   negative. Exposure is limited to the lowest-weight region of the metric.
2. **One deliberate domain-gate relaxation.** A Computer Vision Engineer with
   genuine recsys/ranking career history is retained at rank 57 (manually
   reviewed); the hard CV-without-IR archetype remains excluded.
3. **Anchor sensitivity.** `production_ml` is the least stable anchor
   (LOO Jaccard@100 ≈ 0.887) — acceptable, documented, not re-tuned to avoid
   overfitting to our own audit.
4. **Closed-corpus assumption.** Percentile normalization within 44 templates
   is exactly right for this dataset but would need re-derivation on an open
   corpus; the production story generalizes by re-running Stage A on new data.
