---
title: URSI-FL Candidate Ranker
emoji: "🎯"
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "5.34.2"
app_file: app.py
pinned: false
license: mit
---

# Redrob Candidate Ranking — Senior AI Engineer (Founding Team)

**URSI-FL (Unified Role Semantic Index, Fixed-Leak anchors — promoted variant
FL-E)** — the submitted ranker for the Redrob Intelligent Candidate Discovery
& Ranking Challenge. It produces the top-100 candidates for the released JD
with a grounded, JD-connected 1–2 sentence reasoning for each.

Full methodology: **[`METHODOLOGY_URSI_FL.md`](METHODOLOGY_URSI_FL.md)** — a
self-contained description of the submitted system (architecture, scoring
formula, every gate mapped to a JD requirement, honeypot logic, validation
evidence). Iteration history: [`URSI_FL.md`](URSI_FL.md), [`URSI_V3.md`](URSI_V3.md).

## What's in this repo

This repo contains **only the official, graded reproduction path** — the
ranking step that must run in ≤5 min, ≤16 GB RAM, CPU-only, no network
(per `submission_spec.md` §3). Nothing here is required beyond what's listed
below; there is no hidden step and no manual editing of the output CSV.

## Quickstart (reproduce the submission)

```bash
# 1) Clone this repo, then place the released dataset next to it
#    (candidates.jsonl is distributed by the organizers, not redistributed here).
#    If you only have the .gz bundle:
gunzip -k candidates.jsonl.gz

# 2) Create a clean virtual environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --no-index -r requirements.txt

# 3) Run the official ranker -- OFFLINE, CPU-only, no model load, deterministic.
#    Uses only candidates.jsonl + the committed CSV artifacts below.
python3 rank_ursi_fl.py \
  --candidates ./candidates.jsonl \
  --role-projection ./artifacts/role_semantic_index_fl/fl_e/candidate_role_projection.csv \
  --out ./submission.csv

# 4) Validate the output format
python3 validate_submission.py submission.csv
```

Step 3 is the **single command** that reproduces the submission CSV from
`candidates.jsonl`, as required by `submission_spec.md` §10.3.

For a one-command local reproduction that also creates/uses `.venv` and runs the
validator:

```bash
./reproduce.sh ./candidates.jsonl ./submission.csv
```

The script installs `requirements.txt` with `--no-index`; the official ranking
path has no third-party dependencies, so setup does not need network access.

**Measured on this repo's reference machine** (Apple M2, 8 CPU cores, 8 GB
RAM, Python 3.11.4, macOS 14.8.7): **~35 s wall-clock, ~1.7 GB peak RSS** for
the full 100,000-candidate pool — comfortably inside the 5 min / 16 GB / CPU-only
/ no-network compute budget. Output is **byte-identical across repeated runs**
(fully deterministic; ties broken by ascending `candidate_id`).

The ranking step (`rank_ursi_fl.py`) imports **no third-party package** — only
Python standard library (`csv`, `json`, `re`, `argparse`, `pathlib`).
`requirements.txt` is intentionally dependency-free. `numpy` is listed only in
`requirements-precompute.txt` for the optional precompute step below.

### Running on a small sample

`rank_ursi_fl.py` accepts `--limit N` to read only the first N candidates and
`--top-n N` to emit fewer than 100 rows for sandbox/demo smoke tests:

```bash
python3 rank_ursi_fl.py --candidates ./candidates.jsonl --limit 50 --top-n 50 \
  --role-projection ./artifacts/role_semantic_index_fl/fl_e/candidate_role_projection.csv \
  --out ./sample_out.csv
```

This partial output is for sandbox/demonstration only. Official submissions must
use the default `--top-n 100` and pass `validate_submission.py`.

### Docker reproduction

The repo also includes a minimal Dockerfile. Build the image, then mount the
organizer-provided `candidates.jsonl` into the container:

```bash
docker build -t redrob-ursi-fl .
docker run --rm \
  -v "$PWD/candidates.jsonl:/app/candidates.jsonl:ro" \
  -v "$PWD/docker_out:/out" \
  redrob-ursi-fl \
  python rank_ursi_fl.py \
    --candidates /app/candidates.jsonl \
    --role-projection /app/artifacts/role_semantic_index_fl/fl_e/candidate_role_projection.csv \
    --out /out/submission.csv
docker run --rm -v "$PWD/docker_out:/out" redrob-ursi-fl \
  python validate_submission.py /out/submission.csv
```

## Optional: rebuilding the precomputed artifacts

The official ranker consumes frozen CSV artifacts already committed at
`artifacts/role_semantic_index_fl/fl_e/`. You do **not** need to rebuild them
to reproduce the submission. They were produced once, network-allowed, by:

```bash
export AZURE_OPENAI_API_KEY="your-azure-openai-key"
export AZURE_OPENAI_RESOURCE_URL="https://<your-resource>.openai.azure.com"
python -m pip install -r requirements-precompute.txt
python3 precompute_role_semantic_index_fl.py \
  --candidates ./candidates.jsonl --variant fl_e \
  --out-dir ./artifacts/role_semantic_index_fl/fl_e
```

This step embeds the deduplicated **44 role documents + 48 titles** plus the
FL-E JD anchor pack with Azure `text-embedding-3-large`, and (for
false-positive diagnostics only, never scored) 76 summary templates and 133
skill names. Embeddings are used **exclusively** in this precompute step —
never at ranking time.

## Approach (why this beats keyword/embedding matching)

The dataset is synthetic and **closed**: the 300,171 career-history
descriptions deduplicate to only **44 unique role documents** (discovered
automatically — not assumed). Titles and skills are deliberately noisy traps.

URSI-FL treats ranking as **hybrid retrieval** with a single semantic signal:

1. **JD → contrastive query.** Seven JD-derived facets, each with a positive
   concept (systems the JD wants built) and a negative concept (look-alike
   non-fits the JD's trap section warns about). Encoded as text — never
   matched literally.
2. **Score each unique role document once** with Azure `text-embedding-3-large`:
   `contrast = pos − 0.70·neg`, weighted over facets, **rank-percentile
   normalized within the discovered corpus** — this is what makes
   plain-language fits score high and non-technical roles collapse to ~0,
   with no keyword list.
3. **Project to candidates** by recency, duration, currentness, breadth, and a
   separate semantic **title-coherence** guard.
4. **Rank offline** = career-history role evidence (the one semantic signal) +
   structured **business gates** mapped 1:1 to explicit JD statements
   (India/visa, behavioral availability, seniority, title coherence,
   consulting/tenure guards). Skills and summaries are never scored, so
   keyword-stuffed skill sections have zero effect.
5. **Honeypots** (structural impossibilities + tech anachronisms) are forced
   to score 0 (95 caught in the full pool, 0 in the top 100).

**No manual template grades and no career-evidence keyword lexicons in the
official path.**

See [`METHODOLOGY_URSI_FL.md`](METHODOLOGY_URSI_FL.md) for the full scoring
formula, the gate-by-gate JD justification table, and the complete validation
evidence (fixed-leak invariant, template audit, honeypot audit, anchor
stability, sentinel checks).

## Files

- `rank_ursi_fl.py` — **the official offline ranker** (run this for the submission).
- `ursi_fl_common.py` — shared date/honeypot/anchor logic (used by both the
  ranker and precompute, so honeypot detection can never diverge between them).
- `precompute_role_semantic_index_fl.py` — builds the FL-E role-semantic index
  (optional; network-allowed; not needed to reproduce the submission).
- `artifacts/role_semantic_index_fl/fl_e/` — frozen FL-E artifacts consumed by
  the ranker: `candidate_role_projection.csv`, `unique_role_scores.csv`,
  `manifest.json`, `jd_anchor_pack.json`, `validation_report.md`, plus
  diagnostics-only summary/skill projections (never scored).
- `submission.csv` — the reproduced top-100 submission. **Rename
  to your registered participant ID before uploading to the portal**
  (e.g. `team_xxx.csv`), per `submission_spec.md` §2.
- `validate_submission.py` — format validator (provided by the organizers).
- `submission_metadata.yaml` — mirrors the portal metadata (see
  `submission_spec.md` §10.2); fill in the identity fields before submitting.
- `METHODOLOGY_URSI_FL.md` — the definitive methodology for the submitted system.
- `URSI_FL.md`, `URSI_V3.md` — iteration history / changelogs.
- `requirements.txt` — official ranking dependencies; intentionally empty of
  packages because the ranker is standard-library only.
- `requirements-precompute.txt` — optional `numpy` dependency for rebuilding
  the frozen Azure embedding artifacts.
- `reproduce.sh` — local helper that creates/uses `.venv`, runs the official
  ranker, and validates the 100-row CSV.
- `Dockerfile` / `.dockerignore` — container reproduction path for sandbox or
  Stage 3-style checks.

## Honeypots and traps

The dataset plants ~80 honeypot candidates with structurally impossible
profiles (e.g. a role longer than the entire stated career, expert skills with
zero months of use, technology referenced before its public release date).
`ursi_fl_common.honeypot_reasons()` catches these via pure consistency checks
— no semantics, no keyword lists. On the full 100K pool: **95 caught, 0 in the
top 100**.

## AI tools declaration

See `submission_metadata.yaml` for the full, honest declaration. Summary: AI
tools (Claude, Cursor) assisted with research, design discussion, and code
review; Azure OpenAI `text-embedding-3-large` is used in the precompute step
only. The official ranker makes no API calls and loads no model.

## Sandbox / demo

A hosted sandbox (small-sample reproduction, per `submission_spec.md` §10.5)
is linked from `submission_metadata.yaml` → `sandbox_link`. If a hosted sandbox
is unavailable, the Docker commands above are a self-contained reproduction
recipe: they build the ranker image and run it with the released candidates file
mounted read-only.
