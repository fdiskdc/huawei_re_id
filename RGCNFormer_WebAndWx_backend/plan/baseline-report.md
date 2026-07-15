# Phase 0: Baseline Report — 建立基线与保护措施

Generated: 2026-06-13

---

## 1. Git Status

**Branch:** `main` (up to date with `origin/main`)

**Untracked files:**
```
plan/
```

**Working tree:** Clean (no staged or modified files; only untracked `plan/` directory)

### Last 5 Commits

```
071494d update sh
5693dfb add readme
377c5f9 update docker
2d7134f update run docker
1d37727 add run docker
```

---

## 2. LinearFold/ Directory Verification

- `git diff -- LinearFold/` → **(no output)** — no changes
- `git status --short -- LinearFold/` → **(no output)** — clean

**Status:** LinearFold/ is unchanged and pristine.

---

## 3. Root Directory File List

```
.dockerignore
.env.example
.git/
.gitignore
.idea/
check_speed.py
check.py
common.py
config_docker.py
config.py
docker-compose.yml
Dockerfile
epoch_040.pt
human.py
json/
LinearFold/
main_model_onnx.py
main_model.py
onnx.py
onnx2.py
plan/
README.md
requirements.txt
run_docker.sh
scripts/
server.py
start_backend.sh
stop_backend.sh
tasks_docker.py
tasks.py
test_cache.py
tests/
wsgi.py
```

**Total entries:** 33

---

## 4. Existing API Routes (from `server.py`)

| Method | Path | Handler Function | Description |
|--------|------|------------------|-------------|
| GET | `/api/health` | `health()` | Health check endpoint |
| POST | `/api/v1/wx/login` | `wx_login()` | WeChat Mini Program login (code→openid) |
| POST | `/api/v1/wx-submit-task` | `wx_submit_task()` | Batch submit up to 5 RNA sequences (WeChat) |
| GET | `/api/v1/wx-task-progress/<job_id>` | `wx_task_progress()` | Poll batch task progress |
| POST | `/api/v1/submit-task` | `submit_task()` | Submit single prediction task (async via Celery) |
| GET | `/api/v1/results/<job_id>` | `get_result()` | Retrieve prediction result by job_id |
| GET | `/api/v1/model-architecture` | `get_model_architecture()` | Get model architecture JSON |
| GET | `/api/v1/model-graph` | `get_model_graph()` | Get ONNX model computation graph |
| POST | `/api/v1/integrated-gradients` | `integrated_gradients()` | Compute Integrated Gradients attributions |
| POST | `/api/v1/visualize-gcn-aggregation` | `visualize_gcn_aggregation()` | Visualize GCN message passing |

**Total routes:** 10

---

## 5. Celery Task Names (from `tasks.py`)

| Task Name | Function | Description |
|-----------|----------|-------------|
| `tasks.run_prediction_task` | `run_prediction_task()` | Main RNA prediction pipeline (LinearFold → model → results) |
| `tasks.process_sequence_in_batch` | `process_sequence_in_batch()` | Process single sequence within a WeChat batch job |

**Celery app name:** `rna_prediction_tasks`

**Broker/Backend:** Redis (configured via `config.CELERY_BROKER_URL` / `config.CELERY_RESULT_BACKEND`)

---

## 6. Current Startup Commands

### `start_backend.sh` (Local Development)

```bash
# Conda env: learn (miniconda3)
# Gunicorn: 1 worker on 0.0.0.0:8000, timeout 120s
gunicorn -w 1 -b 0.0.0.0:8000 --timeout 120 --access-logfile gunicorn_access.log --error-logfile gunicorn_error.log wsgi:app

# Celery: 1 concurrency
celery -A tasks.celery_app worker --concurrency=1 --loglevel=info --pidfile=celery.pid --logfile=celery.log &
```

### `Dockerfile`

```dockerfile
FROM python:3.9-slim
WORKDIR /app
# Installs: build-essential, CPU PyTorch 2.0.1, PyG stack, requirements.txt
# Compiles LinearFold, copies config_docker.py → config.py, tasks_docker.py → tasks.py
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:8000", "--timeout", "120", "wsgi:app"]
```

### `docker-compose.yml`

Three services:
- **backend** — Gunicorn (1 worker, port 8000), depends on redis
- **worker** — Celery (`tasks_docker.celery_app`, concurrency=1), depends on redis + backend
- **redis** — `redis:alpine`

---

## Summary

| Item | Status |
|------|--------|
| Git branch | `main`, clean working tree |
| LinearFold/ | Unchanged (no diff, no modifications) |
| API routes | 10 endpoints defined in `server.py` |
| Celery tasks | 2 tasks defined in `tasks.py` |
| Startup | Local (start_backend.sh), Docker (Dockerfile + docker-compose.yml) |
| Key config files | `config.py`, `config_docker.py`, `tasks.py`, `tasks_docker.py` |
