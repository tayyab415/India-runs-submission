FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN python -m pip install --no-index -r requirements.txt

CMD ["python", "rank_ursi_fl.py", "--candidates", "./candidates.jsonl", "--role-projection", "./artifacts/role_semantic_index_fl/fl_e/candidate_role_projection.csv", "--out", "./submission.csv"]
