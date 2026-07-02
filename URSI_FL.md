# URSI-FL Iteration Notes and Adoption Decision

Date: 2026-07-02

## Scope

URSI-FL was implemented as a successor experiment to URSI V3.2. The original
URSI files and frozen baseline artifacts were left untouched. New code and
artifacts live under:

- `ursi_fl_common.py`
- `precompute_role_semantic_index_fl.py`
- `rank_ursi_fl.py`
- `experiments_semantic/test_fl_invariants.py`
- `experiments_semantic/ursi_fl_report.py`
- `experiments_semantic/ursi_fl_validation.py`
- `artifacts/role_semantic_index_fl/`
- `experiments_semantic/reports/ursi_fl/`

The FL precompute embeds the 44 unique career-history role templates, 48 titles,
76 normalized profile-summary templates, and 133 unique skill names with Azure
`text-embedding-3-large`. The official ranker still uses career-history role
evidence plus title/logistics/availability/company/tenure metadata. Summary and
skill embeddings are saved for review and false-positive diagnostics only, so
keyword-stuffed skills or summaries cannot rescue a weak career history.

## Anchor Variants

| Variant | Change | Outcome |
| --- | --- | --- |
| `fl_a` | Retrieval negative rewritten to remove wrapper-vocabulary collision. | RD34 improved, but CV sentinel entered top 100. |
| `fl_b` | Matching/fine-tune positive enriched with LoRA/QLoRA/PEFT/reranking/eval wording. | Top 100 unchanged vs baseline and CV sentinel excluded, but negative-effect invariant failed. |
| `fl_c` | `fl_a + fl_b`. | RD34 improved, but CV sentinel entered top 100. |
| `fl_d` | `fl_c` plus out-of-domain production-ML negative. | Invariant passed, but CV sentinel entered and top-100 churn was too high. |
| `fl_d2` | Strict CV/speech/robotics/mechanical production negative. | Invariant passed, but CV sentinel still entered at rank 95. |
| `fl_d3` | `fl_d2` plus negative-only current-title domain gate for marginal CV/speech/robotics/perception titles. | Invariant passed and CV sentinel excluded, but false-positive counts and lower-tail swaps are not clean enough to adopt. |
| `fl_e` | `fl_d3` anchors plus expanded Redrob behavioral/logistics gates and tiny verified-assessment confirmation. | Promote-candidate: passes the handoff bar after corrected guarded false-positive sweep. |

## Variant Manifest

| Date | Variant | Change | CSV | Verdict |
| --- | --- | --- | --- | --- |
| 2026-07-02 | `fl_a` | Retrieval negative rewritten. | `submission_ursi_fl_fl_a.csv` | Reject: CV sentinel entered top 100. |
| 2026-07-02 | `fl_b` | Matching/fine-tune positive enriched. | `submission_ursi_fl_fl_b.csv` | Reject: negative-effect invariant failed despite clean top-100 overlap. |
| 2026-07-02 | `fl_c` | `fl_a + fl_b`. | `submission_ursi_fl_fl_c.csv` | Reject: CV sentinel entered top 100. |
| 2026-07-02 | `fl_d` | Added out-of-domain production-ML negative. | `submission_ursi_fl_fl_d.csv` | Reject: CV sentinel entered and churn was too high. |
| 2026-07-02 | `fl_d2` | Strict out-of-domain negative. | `submission_ursi_fl_fl_d2.csv` | Reject: CV sentinel still entered. |
| 2026-07-02 | `fl_d3` | Added structured current-title domain gate. | `submission_ursi_fl_fl_d3.csv` | Hold as fallback: passes sentinels/invariants but lower-tail FP regression remains. |
| 2026-07-02 | `fl_e` | Added conservative Redrob behavior/logistics factors. | `submission_ursi_fl_fl_e.csv` | Promote-candidate: corrected sweep shows no genuine guarded FP regression. |

## Title-Domain Gate Decision

FL-D3/FL-E keep `OFF_DOMAIN_TECH_TITLE_RE` in `rank_ursi_fl.py` and document it
as a structured JD-mirror gate, analogous to `CONSULTING_FIRMS`. It is
negative-only, title-field-only, and also requires marginal title coherence; it
does not match career descriptions and cannot create positive evidence.

The cleaner semantic alternative would be to expand `TITLE_NEGATIVE_ANCHOR` and
rerun the 48-title coherence audit. That is not the right FL-E move because it
requires a new Azure title-anchor embedding, shifts all title percentiles, and
the already-expanded FL-D2 title anchor still left `CAND_0088237` too coherent
(`current_title_coherence=0.447917`) to trigger the semantic title veto.

## End-to-End Results

All generated CSVs validated with `validate_submission.py`. FL-D3 was also rerun
for determinism and matched byte-for-byte:

- Final experimental CSV: `submission_ursi_fl_fl_d3.csv`
- Repeat check: `artifacts/role_semantic_index_fl/fl_d3/submission_repeat_check.csv`
- Readable top-100 review: `experiments_semantic/reports/ursi_fl/fl_d3/readable_top100_review.md`
- Top-100 index: `experiments_semantic/reports/ursi_fl/fl_d3/top100_review_index.csv`
- Entrants/removals: `experiments_semantic/reports/ursi_fl/fl_d3/top100_entrants_removals.csv`
- Sentinel table: `experiments_semantic/reports/ursi_fl/fl_d3/sentinel_table.csv`
- Stability validation: `experiments_semantic/reports/ursi_fl/fl_d3/validation.md`

FL-D3 summary:

- Top-10 overlap vs URSI: 10/10.
- Top-50 overlap vs URSI: 49/50.
- Top-100 overlap vs URSI: 95/100.
- Template tier-5 ranks: `[1, 3, 4, 5, 14, 15]`.
- `CAND_0008425`: rank 29 -> 23.
- `CAND_0005260`: rank 95 -> 86.
- `CAND_0033861`: full rank 184 -> 173, still outside top 100.
- `CAND_0088237`: full rank 104 -> 147, excluded from top 100.
- Honeypots in top 100: 0.
- Non-India in top 100: 0.
- Non-technical current titles in top 100: 0.

Readable review verdicts for FL-D3:

- Top 10: 7 strong fit, 3 probable fit.
- Top 50: 37 strong fit, 7 probable fit, 4 strong-but-check-risk, 2 mixed/adjacent.
- Top 100: 52 strong fit, 12 probable fit, 11 strong-but-check-risk, 25 mixed/adjacent.

## Loopholes Found

1. The FL-D family repairs the RD34 anchor gap but allows the "recommendation-style
   features at a mid-stage startup" template to populate too much of the lower
   top 100.
2. FL-D3 removes the known CV sentinel, but another stronger Computer Vision
   profile remains in the top 50 with summary-risk flags. This is explainable by
   repeated recommendation-style career roles, but it is still an adjacent-domain
   risk for the JD.
3. FL-D3's entrants are mostly available candidates with lighter recommendation
   evidence. Several removals are technically cleaner e-commerce/search profiles
   but have weaker Redrob availability/logistics signals. That tradeoff is not
   clearly better for hidden scoring.
4. Historical note: the original FL-D3 sweep appeared to worsen operations and
   content/SEO counts, but the later FL-E audit showed those were measurement
   artifacts from unguarded regexes over skill names and template phrases.
5. Anchor leave-one-out validation shows `production_ml` is the least stable FL-D3
   family (`Jaccard@100=0.887`, `top20_overlap=0.90`). That is acceptable for an
   experiment but not reassuring enough for promotion.

## Decision

FL-E is the promote-candidate from the URSI-FL family. It passes the handoff bar:
template quality is at least baseline, tier-5 templates are protected, RD34
sentinels improve, the CV sentinel is excluded, honeypots remain zero, the ranker
is deterministic/offline, and the corrected guarded false-positive sweep shows no
genuine archetype regression. Final promotion remains the owner's decision, but
`submission_ursi_fl_fl_e.csv` is now the recommended FL submission candidate.

## FL-E Results

FL-E keeps the FL-D3 anchor pack and adds conservative structured gates:

- `applications_submitted_30d`: recent market-activity multiplier.
- `avg_response_time_hours`: mild response-speed nudge.
- `offer_acceptance_rate`: mild offer-follow-through nudge; `-1` is neutral.
- `preferred_work_mode`: mild remote-only penalty only outside JD hubs and not
  willing to relocate.
- `skill_assessment_scores`: tiny `[0.99, 1.02]` confirmation term on
  Redrob-verified JD-adjacent skills only.
- `career_history[].industry`: generalized all-services career detection inside
  the consulting gate.

FL-E checks:

- Validator: pass.
- Deterministic repeat: byte-identical.
- Invariants: pass (`negative_effect_vs_grade_spearman=-0.001`, honeypots zero).
- Top-10 overlap vs official URSI: 9/10.
- Top-50 overlap vs official URSI: 45/50.
- Top-100 overlap vs official URSI: 92/100.
- Top-100 overlap vs FL-D3: 92/100.
- Verdict mix vs FL-D3: `mixed_or_adjacent` 25 -> 21; `strong_fit` 52 -> 59.
- FL-D3 -> FL-E diagnostic improvements: `summary_no_professional_ml` 32 -> 28,
  `zero_recent_applications` 8 -> 6, `remote_outside_hub_no_relocate` 6 -> 4,
  `strong_verified_jd_assessment` 35 -> 41.
- Corrected guarded false-positive sweep vs official URSI:
  `content_seo` 0 -> 0, `operations` 0 -> 0, `rag_support_chatbot` 0 -> 0,
  `generic_mlops_or_churn` 0 -> 0, `qa_test_automation` 0 -> 0,
  `cv_speech_robotics_without_ir` 0 -> 0, `wrapper_or_prompt_bait` 0 -> 0,
  `consulting_only` 0 -> 0, `serial_hopper` 0 -> 0,
  `nontechnical_current_title` 0 -> 0.
- Like-for-like summary diagnostics: official URSI -> FL-E
  `summary_no_professional_ml` 27 -> 28; FL-D3 -> FL-E 32 -> 28.
- `off_domain_current_title` is 0 -> 1 vs official because FL-E retains one
  manually reviewed Computer Vision-titled profile with repeated
  recommendation/ranking career history; the hard CV-without-IR archetype remains
  0 -> 0 and `CAND_0088237` remains excluded.

FL-E sentinel full ranks:

- `CAND_0008425`: 29 -> 25.
- `CAND_0005260`: 95 -> 66.
- `CAND_0033861`: 184 -> 180.
- `CAND_0088237`: 104 -> 148, excluded from top 100.
- `CAND_0088025`: 24 -> 10; top-10 entrant with strong ranking/retrieval roles
  and verified Pinecone/QLoRA assessments.
- `CAND_0076163`: 10 -> 13; remains strong but falls just outside top 10.

FL-E report files:

- `experiments_semantic/reports/ursi_fl/fl_e/readable_top100_review.md`
- `experiments_semantic/reports/ursi_fl/fl_e/top100_review_index.csv`
- `experiments_semantic/reports/ursi_fl/fl_e/top100_entrants_removals.csv`
- `experiments_semantic/reports/ursi_fl/fl_e/validation.md`
- `experiments_semantic/reports/ursi_fl/fl_e_vs_fl_d3/`

Decision: mark FL-E as the promote-candidate. The earlier do-not-promote verdict
was driven by measurement artifacts in the sweep, now corrected in
`experiments_semantic/ursi_fl_report.py`.

Tail manual review:

- `CAND_0069905` (#93, career evidence 0.611): keep. Current and prior roles show
  65 relevant months of semantic search with FAISS/BGE, query expansion, human
  relevance judgments, and ranking/evaluation evidence.
- `CAND_0005649` (#98, career evidence 0.612): keep. The profile has 76 relevant
  months across semantic search, production ML, discovery-feed ranking,
  offline-online evaluation, and strong availability/external-validation signals.

## Signal Utilization Table

| Signal | Used/Omitted | Where / Why |
| --- | --- | --- |
| `career_history[].description` | Used | Azure role-template evidence in precompute; official ranking spine. |
| `career_history[].title` | Used | Title coherence in precompute. |
| `career_history[].industry` | Used in FL-E | Generalizes all-services consulting gate when every role is IT Services/Consulting. |
| `profile.current_title` | Used | Title coherence and JD-mirror off-domain current-title gate. |
| `profile.current_company` | Used | Consulting/services gate via company field. |
| `profile.current_industry` | Omitted | Career-history industry is more precise; current industry alone can over-penalize candidates with product-company history. |
| `profile.location`, `profile.country` | Used | India/no-visa and JD hub logistics gates. |
| `profile.years_of_experience` | Used | Seniority prior. |
| `profile.summary` | Diagnostic only | Embedded and reviewed for summary-risk flags; not scored to avoid keyword-stuffing rescue. |
| Self-reported `skills` | Diagnostic only | Embedded and reviewed; not scored as positive evidence. |
| `profile_completeness_score` | Omitted | Hygiene signal, weak role fit evidence; likely popularity/completeness bias. |
| `signup_date` | Omitted | Tenure on platform is not role fit or availability enough for ranking. |
| `last_active_date` | Used | Behavioral availability gate. |
| `open_to_work_flag` | Used | Behavioral availability gate, calibrated by applications/activity in FL-E. |
| `profile_views_received_30d` | Omitted | Recruiter popularity feedback can amplify platform exposure bias. |
| `applications_submitted_30d` | Used in FL-E | JD-aligned job-market activity signal. |
| `recruiter_response_rate` | Used | Behavioral availability gate. |
| `avg_response_time_hours` | Used in FL-E | Mild responsiveness nudge. |
| `skill_assessment_scores` | Used in FL-E | Tiny bounded confirmation from Redrob-verified JD-adjacent assessments only; non-JD/CV/speech assessments are not positive evidence. |
| `connection_count` | Omitted | Network-size popularity signal, not role evidence. |
| `endorsements_received` | Omitted | Social proof signal, weaker than verified assessment/career evidence. |
| `notice_period_days` | Used | Behavioral/logistics gate. |
| `expected_salary_range_inr_lpa` | Omitted | The JD does not specify a budget; salary matching can be arbitrary. |
| `preferred_work_mode` | Used in FL-E | Mild penalty only for remote-only, outside hubs, and not willing to relocate. |
| `willing_to_relocate` | Used | Location/work-mode logistics. |
| `github_activity_score` | Used | Mild external-validation factor; `-1` is a small negative. |
| `search_appearance_30d` | Omitted | Recruiter-search exposure can amplify platform ranking bias. |
| `saved_by_recruiters_30d` | Omitted | Recruiter-interest proxy, but can duplicate platform exposure bias. |
| `interview_completion_rate` | Used | Behavioral reliability gate. |
| `offer_acceptance_rate` | Used in FL-E | Mild offer-follow-through nudge; `-1` no-history is neutral. |
| `verified_email`, `verified_phone` | Omitted | Account hygiene, not differentiating enough for senior AI fit. |
| `linkedin_connected` | Omitted | Profile hygiene/social-link signal, not role evidence. |
| `education`, `certifications`, `languages` | Omitted | The JD emphasizes hands-on production AI/search/ranking work; these are weak proxies and risk credential bias. |
