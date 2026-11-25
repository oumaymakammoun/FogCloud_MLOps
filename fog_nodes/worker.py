import pickle, base64
import numpy as np
from fastapi import FastAPI
import psutil
import platform
import time
from pydantic import BaseModel
import requests
import threading
from concurrent.futures import ThreadPoolExecutor  # optional
import traceback
from fastapi import HTTPException

app = FastAPI(
    title="Fog Worker API",
    description="API de surveillance d'un noeud Fog",
    version="1.0.0"
)

import os

NODE_PORT = os.getenv("NODE_PORT", "8002")
NODE_ID = f"{platform.node()}-{NODE_PORT}"

CLOUD_NODE_URL = os.getenv(
    "CLOUD_NODE_URL",
    "http://127.0.0.1:8002"
)

HEARTBEAT_INTERVAL = 5
RESOURCE_INTERVAL = 20
START_TIME = time.time()
CLOUD_HEARTBEAT_TIMEOUT = 5
LAST_CLOUD_HEARTBEAT = None

class CloudHeartbeat(BaseModel):
    cloud_id: str
    timestamp: float
    status: str


CURRENT_TASK = None          # dict describing the running/last task
TASK_LOCK = threading.Lock()

class TrainRequest(BaseModel):
    task_id: str
    node_name: str
    model_type: str = "sgd_logistic"
    params: dict = {}
    shard_index: int = 0
    num_shards: int = 1
    epochs: int = 10                      # total epochs of the whole task
    resume_epoch: int = 0                 # last epoch already done
    end_epoch: int | None = None          # last epoch THIS node must train
    segment_index: int = 0
    resume_checkpoint_b64: str | None = None



@app.get("/")
def home():
    return {
        "message": "Fog Worker is running",
        "status": "available"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "uptime_seconds": round(time.time() - START_TIME, 2)
    }


@app.get("/resources")
def resources():
    return {
        "node_name": platform.node(),
        "cpu_percent": psutil.cpu_percent(interval=1),
        "ram_percent": psutil.virtual_memory().percent,
        "ram_available_mb": round(
            psutil.virtual_memory().available / (1024 * 1024), 2
        )
    }

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
                "ram_available_mb": round(
                    psutil.virtual_memory().available / (1024 * 1024), 2
                ),
                "cpu_temperature_c": get_cpu_temperature()
            }

            response = requests.post(
                f"{CLOUD_NODE_URL}/resources-report",
                json=resource_data,
                timeout=5
            )

            response.raise_for_status()

            print(f"[RESOURCES] Sent to Cloud: {resource_data}")

        except requests.RequestException as error:
            print(f"[RESOURCES ERROR] {error}")

        elapsed = time.time() - start_time
        time.sleep(max(0, RESOURCE_INTERVAL - elapsed))

def send_heartbeat():

    while True:
        try:
            response = requests.post(
                f"{CLOUD_NODE_URL}/heartbeat",
                json={
                    "node_id": NODE_ID,
                    "timestamp": time.time(),
                    "status": "alive"
                },
                timeout=3
            )

            print(f"[HEARTBEAT] Sent to Cloud: {response.status_code}")

        except requests.RequestException as error:
            print(f"[HEARTBEAT ERROR] {error}")

        time.sleep(HEARTBEAT_INTERVAL)

@app.on_event("startup")
def startup_event():

    heartbeat_thread = threading.Thread(
        target=send_heartbeat,
        daemon=True
    )
    heartbeat_thread.start()

    resource_thread = threading.Thread(
        target=send_resources,
        daemon=True
    )
    resource_thread.start()


@app.post("/cloud-heartbeat")
def receive_cloud_heartbeat(heartbeat: CloudHeartbeat):

    global LAST_CLOUD_HEARTBEAT

    LAST_CLOUD_HEARTBEAT = time.time()

    print(f"[CLOUD HEARTBEAT] Received from {heartbeat.cloud_id}")

    return {
        "message": "Cloud heartbeat received",
        "status": "alive"
    }

@app.get("/cloud-status")
def cloud_status():

    if LAST_CLOUD_HEARTBEAT is None:
        return {
            "cloud_status": "unknown"
        }

    elapsed = time.time() - LAST_CLOUD_HEARTBEAT

    return {
        "cloud_status": (
            "alive" if elapsed < CLOUD_HEARTBEAT_TIMEOUT
            else "unavailable"
        ),
        "last_heartbeat_seconds_ago": round(elapsed, 2)
    }

def run_training(req: TrainRequest):
    global CURRENT_TASK
    started = time.time()
    try:
        from sklearn.datasets import make_classification
        from sklearn.linear_model import SGDClassifier
        from sklearn.model_selection import train_test_split

        X, y = make_classification(n_samples=20000, n_features=20, random_state=42)
        X, y = X[req.shard_index::req.num_shards], y[req.shard_index::req.num_shards]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        classes = np.unique(y)

        if req.resume_checkpoint_b64:
            model = pickle.loads(base64.b64decode(req.resume_checkpoint_b64))
        else:
            model = SGDClassifier(loss="log_loss", random_state=42, **req.params)

        acc = None
        end_epoch = req.end_epoch or req.epochs
        for epoch in range(req.resume_epoch + 1, end_epoch + 1):
            model.partial_fit(X_tr, y_tr, classes=classes)
            acc = float(model.score(X_te, y_te))

            with TASK_LOCK:
                CURRENT_TASK = {"task_id": req.task_id, "status": "running",
                                "epoch": epoch, "total_epochs": req.epochs,
                                "accuracy": acc}

            send_checkpoint(req, epoch, acc, model)

        result = {"status": "completed", "accuracy": acc}
    except Exception as e:
        traceback.print_exc()
        result = {"status": "failed", "error": str(e)}

    result.update({
        "task_id": req.task_id,
        "node_id": req.node_name,
        "segment_index": req.segment_index,
        "duration_seconds": round(time.time() - started, 2),
    })
    
    with TASK_LOCK:
        CURRENT_TASK = result

    # report to cloud
        # notify cloud that training is done (with retries)
    for attempt in range(1, 4):
        try:
            r = requests.post(f"{CLOUD_NODE_URL}/tasks/result", json=result, timeout=5)
            r.raise_for_status()
            print(f"[TRAINING DONE] Cloud notified for task {req.task_id}")
            break
        except requests.RequestException as e:
            print(f"[TASK RESULT ERROR] attempt {attempt}/3: {e}")
            time.sleep(2)



@app.post("/train", status_code=202)
def train(req: TrainRequest):
    global CURRENT_TASK
    with TASK_LOCK:
        if CURRENT_TASK and CURRENT_TASK.get("status") == "running":
            raise HTTPException(status_code=409, detail="Node busy")
        CURRENT_TASK = {"task_id": req.task_id, "status": "running"}

    threading.Thread(target=run_training, args=(req,), daemon=True).start()
    return {"message": "Training started", "task_id": req.task_id, "node_id": NODE_ID}

@app.get("/train/status")
def train_status():
    return CURRENT_TASK or {"status": "idle"}

def send_checkpoint(req, epoch, accuracy, model):
    payload = {
        "task_id": req.task_id,
        "node_name": req.node_name,
        "epoch": epoch,
        "total_epochs": req.epochs,
        "accuracy": accuracy,
        "model_b64": base64.b64encode(pickle.dumps(model)).decode(),
        "timestamp": time.time(),
    }
    for attempt in range(1, 4):
        try:
            r = requests.post(f"{CLOUD_NODE_URL}/checkpoints", json=payload, timeout=10)
            r.raise_for_status()
            return True
        except requests.RequestException as e:
            print(f"[CHECKPOINT ERROR] epoch {epoch} attempt {attempt}/3: {e}")
            time.sleep(1)
    return False   # don't kill training if the cloud is unreachable