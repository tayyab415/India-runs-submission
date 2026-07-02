#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

CANDIDATES="${1:-./candidates.jsonl}"
OUT="${2:-./submission.csv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
ROLE_PROJECTION="${ROLE_PROJECTION:-./artifacts/role_semantic_index_fl/fl_e/candidate_role_projection.csv}"
TOP_N="${TOP_N:-100}"

if [[ ! -f "$CANDIDATES" ]]; then
  echo "Missing candidates file: $CANDIDATES" >&2
  echo "Place the released candidates.jsonl at the repo root or pass its path:" >&2
  echo "  ./reproduce.sh /path/to/candidates.jsonl ./submission.csv" >&2
  exit 2
fi

if [[ ! -f "$ROLE_PROJECTION" ]]; then
  echo "Missing role projection artifact: $ROLE_PROJECTION" >&2
  exit 2
fi

if [[ "${USE_VENV:-1}" == "1" ]]; then
  if [[ ! -d .venv ]]; then
    "$PYTHON_BIN" -m venv .venv
  fi
  # shellcheck disable=SC1091
  source .venv/bin/activate
  python -m pip install --no-index -r requirements.txt
else
  python() {
    "$PYTHON_BIN" "$@"
  }
fi

python rank_ursi_fl.py \
  --candidates "$CANDIDATES" \
  --role-projection "$ROLE_PROJECTION" \
  --top-n "$TOP_N" \
  --out "$OUT"

if [[ "$TOP_N" == "100" ]]; then
  python validate_submission.py "$OUT"
else
  echo "Skipped official validator because TOP_N=$TOP_N; official submissions must use TOP_N=100."
fi
