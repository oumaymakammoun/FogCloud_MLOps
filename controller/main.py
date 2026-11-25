import base64
import sqlite3
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

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
FOG_NODES = {
    "dell-8001": "http://127.0.0.1:8001",
    # "dell-8002": "http://127.0.0.1:8002",
    "dell-8003": "http://127.0.0.1:8003",
    "lenovo-8001": "http://192.168.43.94:8001",
    "lenovo-8002": "http://192.168.43.221:8002",
    # "fog_node_3": "http://192.168.43.188:8002"
}

HEARTBEAT_TIMEOUT = 15
NODE_HEARTBEATS = {}
NODE_RESOURCES = {}

STANDBY_NODE_ID = "dell-8003"
CLOUD_HEARTBEAT_INTERVAL = 5
CLOUD_ID = "cloud-controller"

TASKS = {}  # task_id -> {"created_at", "model_type", "epochs", "status", "nodes": {node_name: {...}}}
TASKS_LOCK = threading.Lock()

FINAL_STATES = {"completed", "failed"}


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


class LaunchRequest(BaseModel):
    model_type: str = "sgd_logistic"
    params: dict = {}
    epochs: int = 100
    segment_epochs: int | None = None   # default: epochs / number of nodes


class TaskResult(BaseModel):
    task_id: str
    node_id: str
    segment_index: int | None = None
    status: str
    accuracy: float | None = None
    duration_seconds: float | None = None
    error: str | None = None


class CheckpointIn(BaseModel):
    task_id: str
    node_name: str
    epoch: int
    total_epochs: int
    accuracy: float | None = None
    model_b64: str
    timestamp: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def maybe_finalize(task_id: str):
    """If every node is in a final state, close the task."""
    task = TASKS.get(task_id)
    if not task or task.get("status") != "running":
        return

    nodes = task["nodes"].values()
    if not all(n.get("status") in FINAL_STATES for n in nodes):
        return

    statuses = [n["status"] for n in task["nodes"].values()]
    if all(s == "completed" for s in statuses):
        task["status"] = "completed"
    elif all(s == "failed" for s in statuses):
        task["status"] = "failed"
    else:
        task["status"] = "completed_with_failures"

    accs = [n["accuracy"] for n in task["nodes"].values()
            if n.get("status") == "completed" and n.get("accuracy") is not None]
    task["finished_at"] = time.time()
    task["avg_accuracy"] = round(sum(accs) / len(accs), 4) if accs else None
    print(f"[TASK COMPLETE] {task_id} status={task['status']} avg_accuracy={task['avg_accuracy']}")


def with_progress(task: dict):
    nodes = list(task["nodes"].values())
    done = sum(n.get("epoch", 0) for n in nodes)
    total = sum(n.get("total_epochs", 0) for n in nodes) or 1
    return {**task, "progress_percent": round(100 * done / total, 1)}


# ---------------------------------------------------------------------------
# Basic endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def home():
    return {
        "message": "Fog-Cloud Controller is running",
        "time": datetime.now().isoformat()
    }


@app.get("/nodes")
def get_nodes():
    results = {}

    for node_name, address in FOG_NODES.items():
        try:
            response = requests.get(f"{address}/resources", timeout=3)
            response.raise_for_status()

            results[node_name] = {
                "status": "available",
                "resources": response.json()
            }

        except requests.RequestException as error:
            results[node_name] = {
                "status": "unavailable",
                "error": str(error)
            }

    return results


@app.post("/heartbeat")
def receive_heartbeat(heartbeat: Heartbeat):
    NODE_HEARTBEATS[heartbeat.node_id] = {
        "last_heartbeat": time.time(),
        "status": heartbeat.status
    }

    print(f"[HEARTBEAT] Received from {heartbeat.node_id}")

    return {
        "message": "Heartbeat received",
        "node_id": heartbeat.node_id
    }


def is_node_alive(node_id):
    if node_id not in NODE_HEARTBEATS:
        return False

    last_seen = NODE_HEARTBEATS[node_id]["last_heartbeat"]
    return (time.time() - last_seen) < HEARTBEAT_TIMEOUT


@app.post("/resources-report")
def receive_resources(report: ResourceReport):
    NODE_RESOURCES[report.node_id] = {
        "timestamp": report.timestamp,
        "received_at": time.time(),
        "cpu_percent": report.cpu_percent,
        "ram_percent": report.ram_percent,
        "ram_available_mb": report.ram_available_mb,
        "cpu_temperature_c": report.cpu_temperature_c
    }

    print(f"[RESOURCES] Received from {report.node_id}")

    return {
        "message": "Resources received successfully",
        "node_id": report.node_id
    }


@app.get("/resources/latest")
def get_latest_resources():
    return NODE_RESOURCES


def send_cloud_heartbeat():
    while True:
        if STANDBY_NODE_ID in FOG_NODES:
            standby_url = FOG_NODES[STANDBY_NODE_ID]

            try:
                response = requests.post(
                    f"{standby_url}/cloud-heartbeat",
                    json={
                        "cloud_id": CLOUD_ID,
                        "timestamp": time.time(),
                        "status": "alive"
                    },
                    timeout=3
                )
                response.raise_for_status()
                print(f"[CLOUD HEARTBEAT] Sent to {STANDBY_NODE_ID}")

            except requests.RequestException as error:
                print(f"[CLOUD HEARTBEAT ERROR] {error}")

        time.sleep(CLOUD_HEARTBEAT_INTERVAL)


@app.on_event("startup")
def startup_event():
    thread = threading.Thread(target=send_cloud_heartbeat, daemon=True)
    thread.start()


# ---------------------------------------------------------------------------
# Task dispatch
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Pipeline: epochs are split in segments, each segment runs on one node
# ---------------------------------------------------------------------------
STALL_TIMEOUT = 30        # seconds without a checkpoint => node considered dead
SEGMENT_EVENTS = {}       # (task_id, seg_index) -> threading.Event
SEGMENT_RESULTS = {}      # (task_id, seg_index) -> result dict


def with_progress(task: dict):
    pct = 100 * task.get("current_epoch", 0) / max(task["epochs"], 1)
    return {**task, "progress_percent": round(pct, 1)}


def latest_checkpoint(task_id):
    """Return (epoch, model_blob) of the most advanced checkpoint, or (0, None)."""
    with DB_LOCK:
        row = db.execute(
            "SELECT epoch, model_blob FROM checkpoints WHERE task_id=? "
            "ORDER BY epoch DESC, id DESC LIMIT 1", (task_id,)).fetchone()
    return (row[0], row[1]) if row else (0, None)


def dispatch_segment(node_name, task_id, req, seg, resume_epoch, blob):
    try:
        r = requests.post(f"{FOG_NODES[node_name]}/train", json={
            "task_id": task_id,
            "node_name": node_name,
            "model_type": req.model_type,
            "params": req.params,
            "epochs": req.epochs,
            "resume_epoch": resume_epoch,
            "end_epoch": seg["end_epoch"],
            "segment_index": seg["index"],
            "resume_checkpoint_b64": base64.b64encode(blob).decode() if blob else None,
        }, timeout=10)
        r.raise_for_status()
        return True
    except requests.RequestException as e:
        print(f"[DISPATCH ERROR] {node_name}: {e}")
        return False


def run_pipeline(task_id: str, req: LaunchRequest):
    task = TASKS[task_id]
    names = list(FOG_NODES)

    for seg in task["segments"]:
        tried = set()
        # preferred node first, then the others as fallback
        start = names.index(seg["node"])
        order = names[start:] + names[:start]

        while True:
            done_epoch, blob = latest_checkpoint(task_id)
            if done_epoch >= seg["end_epoch"]:
                seg["status"] = "completed"
                break

            candidates = [n for n in order if n not in tried]
            if not candidates:
                seg["status"] = "failed"
                task["status"] = "failed"
                task["finished_at"] = time.time()
                print(f"[TASK FAILED] {task_id} segment {seg['index']}: no node left")
                return

            node = candidates[0]
            event = threading.Event()
            SEGMENT_EVENTS[(task_id, seg["index"])] = event
            seg.update(node=node, status="running")
            task["last_progress_at"] = time.time()

            if not dispatch_segment(node, task_id, req, seg, done_epoch, blob):
                tried.add(node)
                continue

            # wait for the result, but detect a stalled/dead node
            while not event.wait(1):
                if time.time() - task["last_progress_at"] > STALL_TIMEOUT:
                    print(f"[STALL] {node} stopped sending checkpoints")
                    break

            result = SEGMENT_RESULTS.pop((task_id, seg["index"]), None)
            if result and result["status"] == "completed":
                seg["status"] = "completed"
                break

            seg["status"] = "retrying"
            tried.add(node)          # re-dispatch from the last checkpoint

    _, _ = latest_checkpoint(task_id)
    accs = [n.get("last_accuracy") for n in task["nodes"].values()
            if n.get("last_accuracy") is not None]
    task["status"] = "completed"
    task["finished_at"] = time.time()
    task["final_accuracy"] = task.get("last_accuracy")
    print(f"[TASK COMPLETE] {task_id} accuracy={task['final_accuracy']}")


@app.post("/tasks/launch")
def launch_tasks(req: LaunchRequest):
    task_id = str(uuid.uuid4())[:8]
    names = list(FOG_NODES)
    seg_len = req.segment_epochs or math.ceil(req.epochs / len(names))

    segments, start, i = [], 1, 0
    while start <= req.epochs:
        end = min(start + seg_len - 1, req.epochs)
        segments.append({"index": i, "node": names[i % len(names)],
                         "start_epoch": start, "end_epoch": end,
                         "status": "pending"})
        start, i = end + 1, i + 1

    with TASKS_LOCK:
        TASKS[task_id] = {
            "created_at": time.time(),
            "last_progress_at": time.time(),
            "model_type": req.model_type,
            "epochs": req.epochs,
            "current_epoch": 0,
            "status": "running",
            "segments": segments,
            "nodes": {n: {"epochs_trained": 0} for n in names},
        }

    threading.Thread(target=run_pipeline, args=(task_id, req), daemon=True).start()
    return with_progress(TASKS[task_id]) | {"task_id": task_id}


# ---------------------------------------------------------------------------
# Results & checkpoints from fog nodes
# ---------------------------------------------------------------------------
@app.post("/tasks/result")
def receive_task_result(result: TaskResult):
    task = TASKS.get(result.task_id)
    if not task:
        raise HTTPException(404, "Unknown task")

    key = (result.task_id, result.segment_index)
    seg = next((s for s in task["segments"] if s["index"] == result.segment_index), None)
    # ignore stale results from a node that was already replaced
    if seg and seg["node"] == result.node_id and key in SEGMENT_EVENTS:
        SEGMENT_RESULTS[key] = result.dict()
        SEGMENT_EVENTS[key].set()

    print(f"[SEGMENT DONE] {result.node_id} task={result.task_id} "
          f"seg={result.segment_index} -> {result.status}")
    return {"message": "Result received"}


@app.post("/checkpoints")
def receive_checkpoint(cp: CheckpointIn):
    task = TASKS.get(cp.task_id)
    if not task:
        raise HTTPException(404, "Unknown task")

    blob = base64.b64decode(cp.model_b64)
    with DB_LOCK:
        db.execute(
            "INSERT OR REPLACE INTO checkpoints "
            "(task_id,node_name,epoch,total_epochs,accuracy,model_blob,created_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (cp.task_id, cp.node_name, cp.epoch, cp.total_epochs,
             cp.accuracy, blob, cp.timestamp))
        db.commit()

    with TASKS_LOCK:
        node = task["nodes"].setdefault(cp.node_name, {"epochs_trained": 0})
        node["epochs_trained"] += 1
        node["last_accuracy"] = cp.accuracy
        if cp.epoch >= task["current_epoch"]:
            task["current_epoch"] = cp.epoch
            task["last_accuracy"] = cp.accuracy
        task["last_progress_at"] = time.time()

    print(f"[CHECKPOINT] {cp.node_name} task={cp.task_id} epoch={cp.epoch}/{cp.total_epochs} acc={cp.accuracy}")
    return {"message": "Checkpoint saved", "epoch": cp.epoch}


# ---------------------------------------------------------------------------
# Task / checkpoint read endpoints
# ---------------------------------------------------------------------------
@app.get("/tasks")
def list_tasks():
    return {tid: with_progress(t) for tid, t in TASKS.items()}


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
            "FROM checkpoints WHERE task_id=? ORDER BY epoch, id",
            (task_id,)).fetchall()
    cols = ["id", "node_name", "epoch", "total_epochs", "accuracy", "created_at"]
    return [dict(zip(cols, r)) for r in rows]


@app.get("/checkpoints/latest/{task_id}/{node_name}")
def latest_checkpoint_endpoint(task_id: str, node_name: str):
    with DB_LOCK:
        row = db.execute(
            "SELECT epoch,accuracy,created_at FROM checkpoints "
            "WHERE task_id=? AND node_name=? ORDER BY epoch DESC LIMIT 1",
            (task_id, node_name)).fetchone()
    if not row:
        raise HTTPException(404, "No checkpoint")
    return {"epoch": row[0], "accuracy": row[1], "created_at": row[2]}