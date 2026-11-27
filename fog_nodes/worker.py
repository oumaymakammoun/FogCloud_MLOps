import base64
import importlib.util
import json
import os
import pickle
import platform
import sqlite3
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import psutil
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="Fog Worker API",
    description="API de surveillance d'un noeud Fog",
    version="3.1.0"
)


def _port_from_argv():
    if "--port" in sys.argv:
        return sys.argv[sys.argv.index("--port") + 1]
    return os.getenv("NODE_PORT", "8002")


NODE_PORT = _port_from_argv()
# Must match the node's key in FOG_NODES on the controller (e.g. dell-8003)
NODE_ID = os.getenv("NODE_ID", f"{platform.node()}-{NODE_PORT}")

# ---------------------------------------------------------------------------
# Where do I send my heartbeat / resources / results ?
#   PRIMARY_CLOUD_URL = the real cloud controller
#   CLOUD_NODE_URL    = the address I use RIGHT NOW (it changes during a failover)
# ---------------------------------------------------------------------------
PRIMARY_CLOUD_URL = os.getenv("CLOUD_NODE_URL", "http://127.0.0.1:8002").rstrip("/")
CLOUD_NODE_URL = PRIMARY_CLOUD_URL

# ---------------------------------------------------------------------------
# Failover settings (only used by the standby fog)
# ---------------------------------------------------------------------------
STANDBY_NODE_ID = "dell-8003"          # same value as STANDBY_NODE_ID in the controller
IS_STANDBY = os.getenv("IS_STANDBY") == "1" or NODE_ID == STANDBY_NODE_ID
# Address the OTHER fogs must use to reach the standby (use its real IP on a real network)
STANDBY_PUBLIC_URL = os.getenv("STANDBY_PUBLIC_URL", f"http://127.0.0.1:{NODE_PORT}").rstrip("/")
ACTING_CLOUD_URL = f"{STANDBY_PUBLIC_URL}/acting-cloud"
FAILOVER_TIMEOUT = int(os.getenv("FAILOVER_TIMEOUT", "15"))   # seconds without cloud heartbeat
CONTROLLER_FILE = Path(os.getenv(
    "CONTROLLER_FILE",
    Path(__file__).resolve().parent.parent / "controller" / "main.py"))

# SQLite copy of the checkpoints (same file name and same table as the controller)
CHECKPOINT_DB = os.getenv("CHECKPOINT_DB", "checkpoints.db")
# only tasks updated during the last N seconds before the takeover are resumed
RESUME_MAX_AGE = int(os.getenv("RESUME_MAX_AGE", str(FAILOVER_TIMEOUT + 60)))
MIRROR_LOCK = threading.Lock()

ACTING_AS_CLOUD = False
CTRL = None                  # the controller module, loaded only when needed
LAST_ANNOUNCE = 0.0

HEARTBEAT_INTERVAL = 5
RESOURCE_INTERVAL = 10
START_TIME = time.time()
CLOUD_HEARTBEAT_TIMEOUT = 5
LAST_CLOUD_HEARTBEAT = None

RUNNING = {}                    # "task:round" -> info, several shards may run at once
LAST_RESULT = None
TASK_LOCK = threading.Lock()
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT", "4"))
DATA_LOCK = threading.Lock()
SIM = {}          # test overrides for the resources reported to the controller


class CloudHeartbeat(BaseModel):
    cloud_id: str
    timestamp: float
    status: str


class SwitchRequest(BaseModel):
    controller_url: str


class TrainRequest(BaseModel):
    task_id: str
    node_name: str
    model_type: str = "sgd_logistic"
    params: dict = {}
    epochs: int = 10                      # total epochs of the whole task (info)
    epoch: int = 1                        # epoch (round) being computed now
    round_id: int = 0
    shard_start: float = 0.0              # fraction of the training set for THIS node
    shard_end: float = 1.0
    local_epochs: int = 1
    batch_size: int = 15
    resume_checkpoint_b64: str | None = None   # current global model


class SimRequest(BaseModel):
    cpu_percent: float | None = None
    ram_available_mb: float | None = None
    cpu_temperature_c: float | None = None
    cpu_count: int | None = None
    clear: bool = False


@app.get("/")
def home():
    return {"message": "Fog Worker is running", "status": "available", "node_id": NODE_ID}


@app.get("/health")
def health():
    return {"status": "healthy", "uptime_seconds": round(time.time() - START_TIME, 2)}


@app.get("/resources")
def resources():
    return {
        "node_name": NODE_ID,
        "cpu_percent": psutil.cpu_percent(interval=1),
        "ram_percent": psutil.virtual_memory().percent,
        "ram_available_mb": round(psutil.virtual_memory().available / (1024 * 1024), 2),
        "cpu_count": psutil.cpu_count(logical=True),
    }


@app.post("/simulate")
def simulate(s: SimRequest):
    if s.clear:
        SIM.clear()
    else:
        SIM.update({k: v for k, v in s.dict(exclude={"clear"}).items() if v is not None})
    return {"node_id": NODE_ID, "overrides": SIM}


def get_cpu_temperature():
    try:
        temperatures = psutil.sensors_temperatures()
        for sensor_list in temperatures.values():
            for sensor in sensor_list:
                if sensor.current is not None:
                    return sensor.current
    except (AttributeError, OSError):
        pass
    return None


def send_resources():
    while True:
        start_time = time.time()
        try:
            resource_data = {
                "node_id": NODE_ID,
                "timestamp": time.time(),
                "cpu_percent": psutil.cpu_percent(interval=1),
                "ram_percent": psutil.virtual_memory().percent,
                "ram_available_mb": round(psutil.virtual_memory().available / (1024 * 1024), 2),
                "cpu_temperature_c": get_cpu_temperature(),
                "cpu_count": psutil.cpu_count(logical=True),
            }
            resource_data.update(SIM)
            response = requests.post(f"{CLOUD_NODE_URL}/resources-report",
                                     json=resource_data, timeout=5)
            response.raise_for_status()
            print(f"[RESOURCES] Sent to {CLOUD_NODE_URL}: {resource_data}")
        except requests.RequestException as error:
            print(f"[RESOURCES ERROR] {error}")

        time.sleep(max(0, RESOURCE_INTERVAL - (time.time() - start_time)))


def send_heartbeat():
    while True:
        try:
            response = requests.post(
                f"{CLOUD_NODE_URL}/heartbeat",
                json={"node_id": NODE_ID, "timestamp": time.time(), "status": "alive"},
                timeout=3)
            print(f"[HEARTBEAT] Sent to {CLOUD_NODE_URL}: {response.status_code}")
        except requests.RequestException as error:
            print(f"[HEARTBEAT ERROR] {error}")
        time.sleep(HEARTBEAT_INTERVAL)


# ---------------------------------------------------------------------------
# Failover, part 1: every fog can be told "send everything to this address"
# ---------------------------------------------------------------------------
def set_controller_url(url: str):
    global CLOUD_NODE_URL
    url = url.rstrip("/")
    if url != CLOUD_NODE_URL:
        print(f"[SWITCH] {NODE_ID}: from now on I send to {url}")
    CLOUD_NODE_URL = url


@app.post("/switch-controller")
def switch_controller(req: SwitchRequest):
    set_controller_url(req.controller_url)
    return {"node_id": NODE_ID, "sending_to": CLOUD_NODE_URL}


@app.get("/controller-status")
def controller_status():
    return {
        "node_id": NODE_ID,
        "sending_to": CLOUD_NODE_URL,
        "primary_cloud": PRIMARY_CLOUD_URL,
        "is_standby": IS_STANDBY,
        "acting_as_cloud": ACTING_AS_CLOUD,
    }


# ---------------------------------------------------------------------------
# Failover, part 2: the standby keeps its OWN copy of the checkpoints (SQLite)
# ---------------------------------------------------------------------------
def mirror_db():
    db = sqlite3.connect(CHECKPOINT_DB, timeout=10)
    # same table as the controller's checkpoints table
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
    # what the standby needs to restart a task
    db.execute("""
    CREATE TABLE IF NOT EXISTS mirror_tasks (
        task_id TEXT PRIMARY KEY,
        model_type TEXT,
        params_json TEXT,
        total_epochs INTEGER,
        local_epochs INTEGER,
        batch_size INTEGER,
        last_epoch INTEGER,
        resumed INTEGER DEFAULT 0,
        updated_at REAL
    )""")
    return db


def mirror_checkpoint(req: TrainRequest):
    """Standby only: copy the global model received with every /train request."""
    if not req.resume_checkpoint_b64:      # epoch 1: no model yet
        return
    try:
        blob = base64.b64decode(req.resume_checkpoint_b64)
        done = req.epoch - 1               # epochs already finished inside this model
        acc = None
        try:
            d = load_data()
            acc = float(pickle.loads(blob).score(d["X_te"], d["y_te"]))
        except Exception:
            pass
        mirror_name = f"mirror@{NODE_ID}"
        with MIRROR_LOCK:
            db = mirror_db()
            try:
                db.execute(
                    "INSERT OR REPLACE INTO checkpoints "
                    "(task_id,node_name,epoch,total_epochs,accuracy,model_blob,created_at) "
                    "VALUES (?,?,?,?,?,?,?)",
                    (req.task_id, mirror_name, done, req.epochs, acc, blob, time.time()))
                # keep only the latest copy per task (small and enough to resume)
                db.execute("DELETE FROM checkpoints WHERE task_id=? AND node_name=? AND epoch<?",
                           (req.task_id, mirror_name, done))
                db.execute(
                    "INSERT INTO mirror_tasks "
                    "(task_id,model_type,params_json,total_epochs,local_epochs,batch_size,"
                    "last_epoch,resumed,updated_at) VALUES (?,?,?,?,?,?,?,0,?) "
                    "ON CONFLICT(task_id) DO UPDATE SET last_epoch=excluded.last_epoch, "
                    "updated_at=excluded.updated_at",
                    (req.task_id, req.model_type, json.dumps(req.params), req.epochs,
                     req.local_epochs, req.batch_size, done, time.time()))
                db.commit()
            finally:
                db.close()
    except Exception:
        traceback.print_exc()


def _resume_one(task_id, model_type, params_json, total, local_epochs, batch_size,
                last_epoch, blob):
    """Re-create one interrupted task inside the acting controller and continue it."""
    ctrl = CTRL
    params = json.loads(params_json)
    req = ctrl.LaunchRequest(model_type=model_type, params=params, epochs=total,
                             local_epochs=local_epochs, batch_size=batch_size)
    now = time.time()
    with ctrl.TASKS_LOCK:
        ctrl.TASKS[task_id] = {
            "created_at": now,
            "last_progress_at": now,
            "model_type": model_type,
            "params": params,
            "epochs": total,
            "local_epochs": local_epochs,
            "current_epoch": last_epoch,
            "status": "running",
            "phase": f"resumed from epoch {last_epoch} by {NODE_ID}",
            "resumed_from_epoch": last_epoch,
            "nodes": {n: {"rounds": 0, "work_fraction_sum": 0.0, "avg_share_percent": 0.0}
                      for n in ctrl.FOG_NODES},
        }

    def runner():
        task = ctrl.TASKS[task_id]
        current = blob
        try:
            for epoch in range(last_epoch + 1, total + 1):
                current, _ = ctrl.run_round(task_id, task, req, epoch, current)
            os.makedirs("models", exist_ok=True)
            task["model_path"] = f"models/{task_id}.joblib"
            ctrl.joblib.dump(pickle.loads(current), task["model_path"])
        except Exception as e:
            task.update(status="failed", phase="failed", error=str(e), finished_at=time.time())
            print(f"[RESUME FAILED] {task_id}: {e}")
            return
        task.update(status="completed", phase="done", finished_at=time.time(),
                    final_accuracy=task.get("last_accuracy"))
        print(f"[RESUME COMPLETE] {task_id} accuracy={task.get('last_accuracy')}")

    threading.Thread(target=runner, daemon=True).start()
    print(f"[RESUME] task {task_id} restarts at epoch {last_epoch + 1}/{total}")


def resume_interrupted_tasks():
    """Look in the local SQLite copy for tasks that were running when the cloud died."""
    mirror_name = f"mirror@{NODE_ID}"
    with MIRROR_LOCK:
        db = mirror_db()
        try:
            rows = db.execute(
                "SELECT t.task_id,t.model_type,t.params_json,t.total_epochs,t.local_epochs,"
                "t.batch_size,t.last_epoch,c.model_blob "
                "FROM mirror_tasks t JOIN checkpoints c "
                "ON c.task_id=t.task_id AND c.node_name=? AND c.epoch=t.last_epoch "
                "WHERE t.resumed=0 AND t.last_epoch<t.total_epochs AND t.updated_at>?",
                (mirror_name, time.time() - RESUME_MAX_AGE)).fetchall()
            for r in rows:
                db.execute("UPDATE mirror_tasks SET resumed=1 WHERE task_id=?", (r[0],))
            db.commit()
        finally:
            db.close()
    for r in rows:
        _resume_one(*r)
    if not rows:
        print("[RESUME] no interrupted task found in the local SQLite copy")


# ---------------------------------------------------------------------------
# Failover, part 3: the standby can become the cloud (and give it back)
# ---------------------------------------------------------------------------
def load_controller():
    """Load controller/main.py (unchanged) and mount it under /acting-cloud."""
    global CTRL
    if CTRL is None:
        spec = importlib.util.spec_from_file_location("acting_controller", CONTROLLER_FILE)
        module = importlib.util.module_from_spec(spec)
        sys.modules["acting_controller"] = module
        spec.loader.exec_module(module)
        app.mount("/acting-cloud", module.app)
        CTRL = module
        print(f"[FAILOVER] controller loaded from {CONTROLLER_FILE}, mounted on /acting-cloud")
    return CTRL


def announce(url: str):
    """Tell every fog (and myself) where the controller is now."""
    ctrl = load_controller()
    set_controller_url(url)
    for node_id, base in ctrl.FOG_NODES.items():
        if node_id == NODE_ID:
            continue
        for attempt in range(2):
            try:
                r = requests.post(f"{base}/switch-controller",
                                  json={"controller_url": url}, timeout=3)
                r.raise_for_status()
                break
            except requests.RequestException as e:
                if attempt == 1:
                    print(f"[ANNOUNCE ERROR] {node_id}: {e}")


def become_cloud():
    global ACTING_AS_CLOUD, LAST_ANNOUNCE
    ctrl = load_controller()
    # start from a clean picture: the fogs will report to me within a few seconds
    ctrl.NODE_HEARTBEATS.clear()
    ctrl.NODE_RESOURCES.clear()
    with ctrl.ACTIVE_LOCK:
        ctrl.ACTIVE_JOBS.clear()
    try:
        resume_interrupted_tasks()
    except Exception:
        traceback.print_exc()
    ACTING_AS_CLOUD = True
    announce(ACTING_CLOUD_URL)
    LAST_ANNOUNCE = time.time()
    print(f"[FAILOVER] {NODE_ID} is now ACTING AS CLOUD ({ACTING_CLOUD_URL})")


def give_back_to_cloud():
    global ACTING_AS_CLOUD
    ACTING_AS_CLOUD = False
    announce(PRIMARY_CLOUD_URL)
    print(f"[FAILBACK] cloud is back, control returned to {PRIMARY_CLOUD_URL}")


def watch_cloud():
    """Standby only: listen to the cloud heartbeat all the time."""
    global LAST_ANNOUNCE
    while True:
        time.sleep(2)
        # never take over a cloud that was never seen alive
        if LAST_CLOUD_HEARTBEAT is None:
            continue
        silent = time.time() - LAST_CLOUD_HEARTBEAT
        try:
            if silent > FAILOVER_TIMEOUT and not ACTING_AS_CLOUD:
                print(f"[FAILOVER] no cloud heartbeat for {silent:.0f}s -> taking over")
                become_cloud()
            elif silent <= FAILOVER_TIMEOUT and ACTING_AS_CLOUD:
                print("[FAILBACK] cloud heartbeat is back -> giving control back")
                give_back_to_cloud()
            elif ACTING_AS_CLOUD and time.time() - LAST_ANNOUNCE > 10:
                # repeat the announcement so a fog that restarted still finds me
                announce(ACTING_CLOUD_URL)
                LAST_ANNOUNCE = time.time()
        except Exception:
            traceback.print_exc()


@app.on_event("startup")
def startup_event():
    threading.Thread(target=send_heartbeat, daemon=True).start()
    threading.Thread(target=send_resources, daemon=True).start()
    if IS_STANDBY:
        threading.Thread(target=watch_cloud, daemon=True).start()
        print(f"[STANDBY] {NODE_ID} is the standby: watching the cloud heartbeat")


@app.post("/cloud-heartbeat")
def receive_cloud_heartbeat(heartbeat: CloudHeartbeat):
    global LAST_CLOUD_HEARTBEAT
    LAST_CLOUD_HEARTBEAT = time.time()
    print(f"[CLOUD HEARTBEAT] Received from {heartbeat.cloud_id}")
    return {"message": "Cloud heartbeat received", "status": "alive"}


@app.get("/cloud-status")
def cloud_status():
    if LAST_CLOUD_HEARTBEAT is None:
        return {"cloud_status": "unknown", "acting_as_cloud": ACTING_AS_CLOUD}
    elapsed = time.time() - LAST_CLOUD_HEARTBEAT
    return {
        "cloud_status": "alive" if elapsed < CLOUD_HEARTBEAT_TIMEOUT else "unavailable",
        "last_heartbeat_seconds_ago": round(elapsed, 2),
        "acting_as_cloud": ACTING_AS_CLOUD,
    }


# ---------------------------------------------------------------------------
# Training on a slice of the data
# ---------------------------------------------------------------------------
_DATA = {}


def load_data():
    """Same dataset and same train/test split on every node (same seed)."""
    with DATA_LOCK:
        if not _DATA:
            from sklearn.datasets import load_iris
            from sklearn.model_selection import train_test_split
            iris = load_iris()
            X_tr, X_te, y_tr, y_te = train_test_split(
                iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target)
            _DATA.update(X_tr=X_tr, X_te=X_te, y_tr=y_tr, y_te=y_te,
                         classes=np.array([0, 1, 2]))
    return _DATA


def run_training(req: TrainRequest):
    global LAST_RESULT
    key = f"{req.task_id}:{req.round_id}"
    started = time.time()
    try:
        from sklearn.linear_model import SGDClassifier

        d = load_data()
        n = len(d["X_tr"])
        a, b = int(n * req.shard_start), int(n * req.shard_end)
        if b - a < 1:
            raise ValueError("empty data slice")
        Xs, ys = d["X_tr"][a:b], d["y_tr"][a:b]

        if req.resume_checkpoint_b64:
            model = pickle.loads(base64.b64decode(req.resume_checkpoint_b64))
        else:
            model = SGDClassifier(loss="log_loss", random_state=42, **req.params)

        for _ in range(req.local_epochs):
            for s in range(0, len(Xs), req.batch_size):
                model.partial_fit(Xs[s:s + req.batch_size], ys[s:s + req.batch_size],
                                  classes=d["classes"])
        acc = float(model.score(d["X_te"], d["y_te"]))

        result = {"status": "completed", "accuracy": acc, "samples": b - a,
                  "model_b64": base64.b64encode(pickle.dumps(model)).decode()}
        print(f"[TRAIN] epoch {req.epoch} slice {req.shard_start:.2f}-{req.shard_end:.2f} "
              f"({b - a} samples) acc={acc:.4f}")
    except Exception as e:
        traceback.print_exc()
        result = {"status": "failed", "error": str(e)}

    result.update({
        "task_id": req.task_id,
        "node_id": req.node_name,
        "round_id": req.round_id,
        "duration_seconds": round(time.time() - started, 2),
    })

    with TASK_LOCK:
        RUNNING.pop(key, None)
        LAST_RESULT = {k: v for k, v in result.items() if k != "model_b64"}

    for attempt in range(1, 4):
        try:
            r = requests.post(f"{CLOUD_NODE_URL}/tasks/result", json=result, timeout=10)
            r.raise_for_status()
            break
        except requests.RequestException as e:
            print(f"[TASK RESULT ERROR] attempt {attempt}/3: {e}")
            time.sleep(2)


@app.post("/train", status_code=202)
def train(req: TrainRequest):
    key = f"{req.task_id}:{req.round_id}"
    with TASK_LOCK:
        if len(RUNNING) >= MAX_CONCURRENT:
            raise HTTPException(status_code=409, detail="Node busy")
        RUNNING[key] = {"task_id": req.task_id, "epoch": req.epoch,
                        "round_id": req.round_id, "started_at": time.time()}

    if IS_STANDBY:
        threading.Thread(target=mirror_checkpoint, args=(req,), daemon=True).start()
    threading.Thread(target=run_training, args=(req,), daemon=True).start()
    return {"message": "Training started", "task_id": req.task_id, "node_id": NODE_ID}


@app.get("/train/status")
def train_status():
    return {"status": "running" if RUNNING else "idle",
            "running": list(RUNNING.values()), "last_result": LAST_RESULT}