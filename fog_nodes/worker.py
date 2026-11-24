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
    model_type: str = "logistic_regression"
    params: dict = {}
    shard_index: int = 0     # which part of the dataset this node trains on
    num_shards: int = 1



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
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split

        # Replace with your real dataset loading + sharding
        X, y = make_classification(n_samples=20000, n_features=20, random_state=42)
        X, y = X[req.shard_index::req.num_shards], y[req.shard_index::req.num_shards]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)

        models = {
            "logistic_regression": LogisticRegression,
            "random_forest": RandomForestClassifier,
        }
        model = models[req.model_type](**req.params)
        model.fit(X_tr, y_tr)

        result = {
            "status": "completed",
            "accuracy": float(model.score(X_te, y_te)),
        }
    except Exception as e:
        traceback.print_exc()
        result = {"status": "failed", "error": str(e)}

    result.update({
        "task_id": req.task_id,
        "node_id": NODE_ID,
        "duration_seconds": round(time.time() - started, 2),
    })

    with TASK_LOCK:
        CURRENT_TASK = result

    # report to cloud
    try:
        requests.post(f"{CLOUD_NODE_URL}/tasks/result", json=result, timeout=5)
    except requests.RequestException as e:
        print(f"[TASK RESULT ERROR] {e}")



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