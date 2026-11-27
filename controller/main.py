import base64
import itertools
import os
import pickle
import sqlite3
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import joblib
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# SQLite (checkpoints)
# ---------------------------------------------------------------------------
DB_PATH = "checkpoints.db"
DB_LOCK = threading.Lock()
db = sqlite3.connect(DB_PATH, check_same_thread=False)
db.execute("""
CREATE TABLE IF NOT EXISTS checkpoints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    node_name TEXT NOT NULL,
    epoch INTEGER NOT NULL,
    total_epochs INTEGER NOT NULL,
    accuracy REAL,
    model_blob BLOB,
    created_at REAL NOT NULL,
    UNIQUE(task_id, node_name, epoch)
)""")
db.commit()

app = FastAPI(title="Fog-Cloud Controller")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# Each key must equal the NODE_ID of the fog (fog.py derives it from --port).
FOG_NODES = {
    # "dell-8001": "http://127.0.0.1:8001",
    "dell-8003": "http://127.0.0.1:8003",
    "dell-8004": "http://127.0.0.1:8004",
    "dell-8005": "http://127.0.0.1:8005",
    # "lenovo-8001": "http://192.168.43.94:8001",
    # "lenovo-8002": "http://192.168.43.221:8002",
}

HEARTBEAT_TIMEOUT = 15
NODE_HEARTBEATS = {}
NODE_RESOURCES = {}

STANDBY_NODE_ID = "dell-8003"
CLOUD_HEARTBEAT_INTERVAL = 5
CLOUD_ID = "cloud-controller"

TASKS = {}
TASKS_LOCK = threading.Lock()

# ---- scheduler config ----
RESOURCE_MAX_AGE = 30      # seconds; older reports => node ignored
MIN_RAM_MB = 300
MAX_TEMP_C = 85
MIN_TEMP_C = 40
W_CPU, W_CORES, W_RAM, W_TEMP = 0.35, 0.15, 0.30, 0.20
NO_NODE_WAIT = 60          # seconds without ANY healthy node before a task fails
ROUND_TIMEOUT = 60         # max seconds to wait for all nodes in one epoch
MAX_JOBS_PER_NODE = 2      # how many tasks may train on the same node at once

ACTIVE_JOBS = {}           # node -> number of shards currently running on it
ACTIVE_LOCK = threading.Lock()

ROUND_SEQ = itertools.count(1)       # unique id for every dispatch attempt
ROUND_RESULTS = {}                   # (task_id, round_id) -> {node_id: result}
RESULT_COND = threading.Condition()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class Heartbeat(BaseModel):
    node_id: str
    timestamp: float
    status: str


class ResourceReport(BaseModel):
    node_id: str
    timestamp: float
    cpu_percent: float
    ram_percent: float
    ram_available_mb: float
    cpu_temperature_c: float | None = None
    cpu_count: int | None = None


class LaunchRequest(BaseModel):
    model_type: str = "sgd_logistic"
    params: dict = {}
    epochs: int = 100          # number of rounds; each round = 1 pass over ALL the data
    local_epochs: int = 1      # passes each node makes over its own slice per round
    batch_size: int = 15       # mini-batch size used by the fog nodes (partial_fit)


class LaunchManyRequest(BaseModel):
    tasks: list[LaunchRequest]


class TaskResult(BaseModel):
    task_id: str
    node_id: str
    round_id: int | None = None
    status: str
    accuracy: float | None = None
    samples: int | None = None
    duration_seconds: float | None = None
    error: str | None = None
    model_b64: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def with_progress(task: dict):
    pct = 100 * task.get("current_epoch", 0) / max(task["epochs"], 1)
    return {**task, "progress_percent": round(pct, 1)}


def save_checkpoint(task_id, name, epoch, total, acc, blob):
    with DB_LOCK:
        db.execute(
            "INSERT OR REPLACE INTO checkpoints "
            "(task_id,node_name,epoch,total_epochs,accuracy,model_blob,created_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (task_id, name, epoch, total, acc, blob, time.time()))
        db.commit()


_EVAL = {}
_EVAL_LOCK = threading.Lock()


def eval_data():
    """Same test split as the fog nodes (same seed) -> global model accuracy."""
    with _EVAL_LOCK:
        if not _EVAL:
            from sklearn.datasets import load_iris
            from sklearn.model_selection import train_test_split
            iris = load_iris()
            _, X_te, _, y_te = train_test_split(
                iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target)
            _EVAL["X"], _EVAL["y"] = X_te, y_te
    return _EVAL["X"], _EVAL["y"]


def average_models(items):
    """items: list of (model_b64, weight). Weighted average of coef_/intercept_."""
    models = [pickle.loads(base64.b64decode(b)) for b, _ in items]
    weights = [w for _, w in items]
    total = sum(weights)
    base = models[0]
    base.coef_ = sum(m.coef_ * w for m, w in zip(models, weights)) / total
    base.intercept_ = sum(m.intercept_ * w for m, w in zip(models, weights)) / total
    return base


# ---------------------------------------------------------------------------
# Basic endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def home():
    return {"message": "Fog-Cloud Controller is running", "time": datetime.now().isoformat()}


@app.get("/nodes")
def get_nodes():
    results = {}
    for node_name, address in FOG_NODES.items():
        try:
            response = requests.get(f"{address}/resources", timeout=3)
            response.raise_for_status()
            results[node_name] = {"status": "available", "resources": response.json()}
        except requests.RequestException as error:
            results[node_name] = {"status": "unavailable", "error": str(error)}
    return results


@app.post("/heartbeat")
def receive_heartbeat(heartbeat: Heartbeat):
    NODE_HEARTBEATS[heartbeat.node_id] = {"last_heartbeat": time.time(), "status": heartbeat.status}
    print(f"[HEARTBEAT] Received from {heartbeat.node_id}")
    return {"message": "Heartbeat received", "node_id": heartbeat.node_id}


def is_node_alive(node_id):
    if node_id not in NODE_HEARTBEATS:
        return False
    return (time.time() - NODE_HEARTBEATS[node_id]["last_heartbeat"]) < HEARTBEAT_TIMEOUT


@app.post("/resources-report")
def receive_resources(report: ResourceReport):
    NODE_RESOURCES[report.node_id] = {
        "timestamp": report.timestamp,
        "received_at": time.time(),
        "cpu_percent": report.cpu_percent,
        "ram_percent": report.ram_percent,
        "ram_available_mb": report.ram_available_mb,
        "cpu_temperature_c": report.cpu_temperature_c,
        "cpu_count": report.cpu_count,
    }
    print(f"[RESOURCES] Received from {report.node_id}")
    return {"message": "Resources received successfully", "node_id": report.node_id}


@app.get("/resources/latest")
def get_latest_resources():
    return NODE_RESOURCES


def send_cloud_heartbeat():
    while True:
        if STANDBY_NODE_ID in FOG_NODES:
            try:
                response = requests.post(
                    f"{FOG_NODES[STANDBY_NODE_ID]}/cloud-heartbeat",
                    json={"cloud_id": CLOUD_ID, "timestamp": time.time(), "status": "alive"},
                    timeout=3)
                response.raise_for_status()
                print(f"[CLOUD HEARTBEAT] Sent to {STANDBY_NODE_ID}")
            except requests.RequestException as error:
                print(f"[CLOUD HEARTBEAT ERROR] {error}")
        time.sleep(CLOUD_HEARTBEAT_INTERVAL)


@app.on_event("startup")
def startup_event():
    threading.Thread(target=send_cloud_heartbeat, daemon=True).start()


# ---------------------------------------------------------------------------
# Scheduler: score nodes by live resources + static capacity + current load
# ---------------------------------------------------------------------------
def node_score(name):
    """0-100 score, or None if the node must not be used right now."""
    if not is_node_alive(name):
        return None
    r = NODE_RESOURCES.get(name)
    if not r or time.time() - r["received_at"] > RESOURCE_MAX_AGE:
        return None
    if r["ram_available_mb"] < MIN_RAM_MB:
        return None
    t = r.get("cpu_temperature_c")
    if t is not None and t >= MAX_TEMP_C:
        return None

    cpu_free = 100 - r["cpu_percent"]
    ram_score = min(r["ram_available_mb"] / 4096, 1) * 100
    cores = r.get("cpu_count")
    cores_score = min(cores / 8, 1) * 100 if cores else 50
    temp_score = 100 if t is None else max(0, min(100, (MAX_TEMP_C - t) / (MAX_TEMP_C - MIN_TEMP_C) * 100))
    return W_CPU * cpu_free + W_CORES * cores_score + W_RAM * ram_score + W_TEMP * temp_score


def reserve_nodes(exclude=()):
    """Atomically reserve a job slot on every eligible node that is not full.
    Returns {node: effective_score}, best first. The effective score is divided by
    (1 + jobs already running there), so busy nodes receive smaller data shares."""
    with ACTIVE_LOCK:
        cands = []
        for n in FOG_NODES:
            if n in exclude:
                continue
            s = node_score(n)
            if s is None:
                continue
            a = ACTIVE_JOBS.get(n, 0)
            if a >= MAX_JOBS_PER_NODE:
                continue
            cands.append((s / (1 + a), n))
        cands.sort(reverse=True)
        for _, n in cands:
            ACTIVE_JOBS[n] = ACTIVE_JOBS.get(n, 0) + 1
        return {n: eff for eff, n in cands}


def release_nodes(names):
    with ACTIVE_LOCK:
        for n in names:
            ACTIVE_JOBS[n] = max(0, ACTIVE_JOBS.get(n, 0) - 1)


@app.get("/scheduler/ranking")
def scheduler_ranking():
    rows = []
    for n in FOG_NODES:
        s = node_score(n)
        if s is None:
            continue
        a = ACTIVE_JOBS.get(n, 0)
        rows.append({"node": n, "score": round(s, 1), "active_jobs": a,
                     "effective_score": round(s / (1 + a), 1)})
    return sorted(rows, key=lambda r: -r["effective_score"])


@app.get("/scheduler/load")
def scheduler_load():
    return {"active_jobs": dict(ACTIVE_JOBS), "max_jobs_per_node": MAX_JOBS_PER_NODE}


@app.get("/scheduler/debug")
def scheduler_debug():
    out = {}
    for n in FOG_NODES:
        r = NODE_RESOURCES.get(n)
        reasons = []
        if not is_node_alive(n):
            reasons.append("no recent heartbeat")
        if not r:
            reasons.append("no resource report")
        else:
            if time.time() - r["received_at"] > RESOURCE_MAX_AGE:
                reasons.append("resource report too old")
            if r["ram_available_mb"] < MIN_RAM_MB:
                reasons.append("low RAM")
            t = r.get("cpu_temperature_c")
            if t is not None and t >= MAX_TEMP_C:
                reasons.append("too hot")
        out[n] = {"eligible": not reasons, "reasons": reasons,
                  "active_jobs": ACTIVE_JOBS.get(n, 0), "heartbeat_seen": n in NODE_HEARTBEATS}
    unknown = [k for k in set(NODE_HEARTBEATS) | set(NODE_RESOURCES) if k not in FOG_NODES]
    return {"nodes": out, "reporting_but_not_in_FOG_NODES": unknown}


# ---------------------------------------------------------------------------
# Data-parallel training: every epoch is split across ALL eligible nodes
# ---------------------------------------------------------------------------
def dispatch_shard(node, task_id, req, epoch, round_id, shard, blob_b64):
    try:
        r = requests.post(f"{FOG_NODES[node]}/train", json={
            "task_id": task_id,
            "node_name": node,
            "model_type": req.model_type,
            "params": req.params,
            "epochs": req.epochs,
            "epoch": epoch,
            "round_id": round_id,
            "shard_start": shard[0],
            "shard_end": shard[1],
            "local_epochs": req.local_epochs,
            "batch_size": req.batch_size,
            "resume_checkpoint_b64": blob_b64,
        }, timeout=10)
        r.raise_for_status()
        return True
    except requests.RequestException as e:
        print(f"[DISPATCH ERROR] {node}: {e}")
        return False


def run_round(task_id, task, req, epoch, blob):
    """Run one epoch on all eligible nodes in parallel. Returns (new_blob, accuracy)."""
    bad, waited = set(), 0
    blob_b64 = base64.b64encode(blob).decode() if blob else None

    while True:
        # no healthy node at all => count towards the failure timeout
        eligible = [n for n in FOG_NODES if n not in bad and node_score(n) is not None]
        if not eligible:
            if waited >= NO_NODE_WAIT:
                raise RuntimeError(f"no eligible node for epoch {epoch}")
            task["phase"] = "waiting_for_node"
            time.sleep(3)
            waited += 3
            continue
        waited = 0

        reserved = reserve_nodes(exclude=bad)
        if not reserved:
            # healthy nodes exist but all are full with other tasks => just queue
            task["phase"] = "queued (all nodes at capacity)"
            time.sleep(0.5)
            continue

        ranking = list(reserved)
        try:
            # data share of each node is proportional to its effective score
            scores = {n: max(v, 0.1) for n, v in reserved.items()}
            total = sum(scores.values())
            assignment, shares, cursor = {}, {}, 0.0
            for i, n in enumerate(ranking):
                end = 1.0 if i == len(ranking) - 1 else cursor + scores[n] / total
                assignment[n] = (cursor, end)
                shares[n] = end - cursor
                cursor = end

            round_id = next(ROUND_SEQ)
            key = (task_id, round_id)
            with RESULT_COND:
                ROUND_RESULTS[key] = {}
            task["phase"] = f"epoch {epoch}: training on {', '.join(ranking)}"
            print(f"[ROUND] task={task_id} epoch {epoch} shares: " +
                  ", ".join(f"{n}={shares[n]*100:.0f}%" for n in ranking))

            with ThreadPoolExecutor(len(ranking)) as ex:
                oks = list(ex.map(
                    lambda n: dispatch_shard(n, task_id, req, epoch, round_id,
                                             assignment[n], blob_b64), ranking))
            dispatched = [n for n, ok in zip(ranking, oks) if ok]

            deadline = time.time() + ROUND_TIMEOUT
            with RESULT_COND:
                while time.time() < deadline and not all(
                        n in ROUND_RESULTS[key] for n in dispatched):
                    RESULT_COND.wait(1)
                results = ROUND_RESULTS.pop(key)

            good = {n: r for n, r in results.items()
                    if n in dispatched and r["status"] == "completed" and r.get("model_b64")}
            failed = set(ranking) - set(good)
            if failed:
                print(f"[ROUND RETRY] task={task_id} epoch {epoch}: "
                      f"{sorted(failed)} failed, redistributing")
                bad |= failed
                continue
        finally:
            release_nodes(ranking)

        # every node answered: average the models (weighted by data share)
        model = average_models([(good[n]["model_b64"], shares[n]) for n in ranking])
        new_blob = pickle.dumps(model)
        X_te, y_te = eval_data()
        acc = float(model.score(X_te, y_te))
        save_checkpoint(task_id, "+".join(ranking), epoch, req.epochs, acc, new_blob)

        with TASKS_LOCK:
            task["current_epoch"] = epoch
            task["last_accuracy"] = acc
            task["last_progress_at"] = time.time()
            for n in ranking:
                st = task["nodes"].setdefault(n, {"rounds": 0, "work_fraction_sum": 0.0})
                st["rounds"] += 1
                st["work_fraction_sum"] += shares[n]
                st["avg_share_percent"] = round(100 * st["work_fraction_sum"] / epoch, 1)
                st["last_share_percent"] = round(shares[n] * 100, 1)
                st["last_local_accuracy"] = good[n].get("accuracy")
            task["recent_rounds"] = (task.get("recent_rounds", []) + [{
                "epoch": epoch, "accuracy": round(acc, 4),
                "shares_percent": {n: round(shares[n] * 100, 1) for n in ranking},
            }])[-5:]

        print(f"[EPOCH DONE] {task_id} epoch={epoch}/{req.epochs} acc={acc:.4f}")
        return new_blob, acc


def run_task(task_id: str, req: LaunchRequest):
    task = TASKS[task_id]
    blob = None
    try:
        for epoch in range(1, req.epochs + 1):
            blob, _ = run_round(task_id, task, req, epoch, blob)
        if blob:
            os.makedirs("models", exist_ok=True)
            task["model_path"] = f"models/{task_id}.joblib"
            joblib.dump(pickle.loads(blob), task["model_path"])
    except Exception as e:
        task.update(status="failed", phase="failed", error=str(e), finished_at=time.time())
        print(f"[TASK FAILED] {task_id}: {e}")
        return
    task.update(status="completed", phase="done", finished_at=time.time(),
                final_accuracy=task.get("last_accuracy"))
    print(f"[TASK COMPLETE] {task_id} accuracy={task['final_accuracy']}")


def start_task(req: LaunchRequest) -> str:
    task_id = str(uuid.uuid4())[:8]
    with TASKS_LOCK:
        TASKS[task_id] = {
            "created_at": time.time(),
            "last_progress_at": time.time(),
            "model_type": req.model_type,
            "params": req.params,
            "epochs": req.epochs,
            "local_epochs": req.local_epochs,
            "current_epoch": 0,
            "status": "running",
            "phase": "starting",
            "nodes": {n: {"rounds": 0, "work_fraction_sum": 0.0, "avg_share_percent": 0.0}
                      for n in FOG_NODES},
        }
    threading.Thread(target=run_task, args=(task_id, req), daemon=True).start()
    return task_id


@app.post("/tasks/launch")
def launch_tasks(req: LaunchRequest):
    task_id = start_task(req)
    return with_progress(TASKS[task_id]) | {"task_id": task_id}


@app.post("/tasks/launch-many")
def launch_many(body: LaunchManyRequest):
    """Start several independent tasks at the same time."""
    return {"task_ids": [start_task(r) for r in body.tasks]}


# ---------------------------------------------------------------------------
# Results from fog nodes
# ---------------------------------------------------------------------------
@app.post("/tasks/result")
def receive_task_result(result: TaskResult):
    if result.task_id not in TASKS:
        raise HTTPException(404, "Unknown task")

    key = (result.task_id, result.round_id)
    with RESULT_COND:
        if key in ROUND_RESULTS:            # otherwise: stale result, ignore
            ROUND_RESULTS[key][result.node_id] = result.dict()
            RESULT_COND.notify_all()

    print(f"[SHARD DONE] {result.node_id} task={result.task_id} "
          f"round={result.round_id} -> {result.status}")
    return {"message": "Result received"}


# ---------------------------------------------------------------------------
# Task / checkpoint read endpoints
# ---------------------------------------------------------------------------
@app.get("/tasks")
def list_tasks():
    return {tid: with_progress(t) for tid, t in TASKS.items()}


@app.get("/tasks/summary")
def tasks_summary():
    """Compact view of all tasks (define BEFORE /tasks/{task_id})."""
    out = {}
    for tid, t in TASKS.items():
        out[tid] = {
            "status": t["status"],
            "phase": t.get("phase"),
            "epoch": f'{t["current_epoch"]}/{t["epochs"]}',
            "progress_percent": with_progress(t)["progress_percent"],
            "accuracy": t.get("last_accuracy"),
            "avg_share_percent": {n: s.get("avg_share_percent", 0) for n, s in t["nodes"].items()},
        }
    return out


@app.get("/tasks/{task_id}")
def get_task(task_id: str):
    if task_id not in TASKS:
        raise HTTPException(404, "Unknown task")
    return with_progress(TASKS[task_id])


@app.get("/tasks/{task_id}/checkpoints")
def list_checkpoints(task_id: str):
    with DB_LOCK:
        rows = db.execute(
            "SELECT id,node_name,epoch,total_epochs,accuracy,created_at "
            "FROM checkpoints WHERE task_id=? ORDER BY epoch, id", (task_id,)).fetchall()
    cols = ["id", "node_name", "epoch", "total_epochs", "accuracy", "created_at"]
    return [dict(zip(cols, r)) for r in rows]
