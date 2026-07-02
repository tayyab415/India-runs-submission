"""URSI-FL ranking sandbox — Gradio interface for the Redrob hackathon.

Accepts a candidate file (JSONL, JSON array, or CSV with a 'json' column),
runs the offline ranker, and returns the ranked CSV for download.
"""

import csv
import io
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

import gradio as gr

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from rank_ursi_fl import (
    build_reasoning,
    iter_candidates,
    load_role_doc_evidence,
    load_role_metadata,
    load_role_previews,
    load_role_projection,
    score_candidate,
    write_submission,
)

ROLE_PROJ_PATH = REPO_ROOT / "artifacts" / "role_semantic_index_fl" / "fl_e" / "candidate_role_projection.csv"
UNIQUE_ROLE_PATH = REPO_ROOT / "artifacts" / "role_semantic_index_fl" / "fl_e" / "unique_role_scores.csv"

MAX_CANDIDATES = 500


def parse_upload(file_path: str) -> list[dict]:
    """Parse uploaded file into a list of candidate dicts.

    Accepted formats:
      - .jsonl: one JSON object per line (the official dataset format).
      - .json: a JSON array of candidate objects, or one object per line.
      - .csv: must contain a column named 'json' or 'candidate_json' whose
        values are JSON-encoded candidate objects; OR if the CSV has the
        standard candidate_schema.json top-level keys as columns, each row
        is reconstructed into a dict (flat fields only — nested objects like
        career_history must be JSON-encoded cell values).
    """
    path = Path(file_path)
    raw = path.read_text(encoding="utf-8")
    ext = path.suffix.lower()

    candidates: list[dict] = []

    if ext in (".jsonl",):
        for line in raw.splitlines():
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    elif ext in (".json",):
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                candidates = data
            elif isinstance(data, dict):
                candidates = [data]
            else:
                raise ValueError(f"Expected JSON array or object, got {type(data).__name__}")
        except json.JSONDecodeError:
            for line in raw.splitlines():
                line = line.strip()
                if line:
                    candidates.append(json.loads(line))
    elif ext in (".csv",):
        reader = csv.DictReader(io.StringIO(raw))
        cols = reader.fieldnames or []
        json_col = None
        for c in ("json", "candidate_json", "data"):
            if c in cols:
                json_col = c
                break
        if json_col:
            for row in reader:
                candidates.append(json.loads(row[json_col]))
        elif "candidate_id" in cols:
            for row in reader:
                c = {}
                for k, v in row.items():
                    v = v.strip() if v else ""
                    if not v:
                        continue
                    if v.startswith(("{", "[", '"')) or v in ("true", "false", "null"):
                        try:
                            c[k] = json.loads(v)
                            continue
                        except (json.JSONDecodeError, ValueError):
                            pass
                    try:
                        c[k] = float(v) if "." in v else int(v)
                    except ValueError:
                        c[k] = v
                candidates.append(c)
        else:
            raise ValueError(
                f"CSV must have a 'json'/'candidate_json' column with full candidate JSON, "
                f"or standard candidate_schema.json column names. Found columns: {cols}"
            )
    else:
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                candidates = data
            elif isinstance(data, dict):
                candidates = [data]
        except json.JSONDecodeError:
            for line in raw.splitlines():
                line = line.strip()
                if line:
                    candidates.append(json.loads(line))

    return candidates


def validate_candidate(c: dict, idx: int) -> list[str]:
    """Light validation — check the minimum fields the ranker needs."""
    issues = []
    if "candidate_id" not in c:
        issues.append(f"Row {idx}: missing 'candidate_id'")
    if "career_history" not in c and "profile" not in c:
        issues.append(f"Row {idx} ({c.get('candidate_id', '?')}): missing both 'career_history' and 'profile'")
    return issues


def rank_candidates(file_obj) -> tuple[str, str | None]:
    """Run the URSI-FL ranker on the uploaded file and return (status, csv_path)."""
    if file_obj is None:
        return "No file uploaded.", None

    file_path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)

    try:
        candidates = parse_upload(file_path)
    except Exception as e:
        return f"Failed to parse file: {e}\n\n{traceback.format_exc()}", None

    if not candidates:
        return "File parsed but contained 0 candidates.", None

    if len(candidates) > MAX_CANDIDATES:
        return (
            f"Upload contains {len(candidates)} candidates — this sandbox accepts "
            f"at most {MAX_CANDIDATES} for demo purposes. The full 100K pool runs "
            f"locally via reproduce.sh.",
            None,
        )

    issues = []
    for i, c in enumerate(candidates):
        issues.extend(validate_candidate(c, i))
    if issues:
        return "Validation errors:\n" + "\n".join(issues[:20]), None

    proj = load_role_projection(ROLE_PROJ_PATH)
    role_doc_evidence = load_role_doc_evidence(UNIQUE_ROLE_PATH)
    previews = load_role_previews(UNIQUE_ROLE_PATH)
    role_meta = load_role_metadata(UNIQUE_ROLE_PATH)

    scored = []
    missing_proj = 0
    for c in candidates:
        row = score_candidate(c, proj, role_doc_evidence)
        if row["score"] == 0.0 and not row.get("honeypot") and row.get("reasons") == ["missing role projection"]:
            missing_proj += 1
        scored.append((-row["score"], c["candidate_id"], row))

    scored.sort()
    top_n = min(len(scored), 100)
    top = [r for _, _, r in scored[:top_n]]

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, dir=REPO_ROOT, prefix="sandbox_out_"
    ) as f:
        out_path = f.name
        w = csv.writer(f)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        prev = float("inf")
        for rank, row in enumerate(top, start=1):
            score = max(0.0, min(1.0, min(float(row["score"]), prev - 1e-9)))
            prev = score
            w.writerow([
                row["candidate_id"],
                rank,
                f"{score:.9f}",
                build_reasoning(row, rank, previews, role_meta),
            ])

    status_lines = [
        f"Ranked {len(candidates)} candidates — top {top_n} written.",
    ]
    if missing_proj:
        status_lines.append(
            f"{missing_proj} candidate(s) had no precomputed role projection "
            f"(scored 0). This is expected when running on candidates outside "
            f"the original 100K pool — they weren't embedded at precompute time."
        )
    honeypots = sum(1 for _, _, r in scored if r.get("honeypot"))
    if honeypots:
        status_lines.append(f"{honeypots} honeypot(s) detected and scored 0.")

    return "\n".join(status_lines), out_path


with gr.Blocks(title="URSI-FL Candidate Ranker — Redrob Hackathon Sandbox") as app:
    gr.Markdown(
        "## URSI-FL Candidate Ranker — Sandbox\n"
        "Upload a candidate file to rank against the Redrob Senior AI Engineer JD. "
        "Accepts **.jsonl** (one JSON object per line), **.json** (array of objects), "
        "or **.csv** (with a `json` or `candidate_json` column containing full candidate JSON). "
        f"Demo limit: {MAX_CANDIDATES} candidates."
    )
    with gr.Row():
        file_input = gr.File(
            label="Candidate file (.jsonl, .json, or .csv)",
            file_types=[".jsonl", ".json", ".csv"],
        )
    run_btn = gr.Button("Run Ranker", variant="primary")
    status_box = gr.Textbox(label="Status", lines=5, interactive=False)
    output_file = gr.File(label="Download ranked CSV")

    run_btn.click(rank_candidates, inputs=[file_input], outputs=[status_box, output_file])


if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)),
               show_error=True)
