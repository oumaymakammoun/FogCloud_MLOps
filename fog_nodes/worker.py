from fastapi import FastAPI
import psutil
import platform
import time
import requests
import threading

app = FastAPI(
    title="Fog Worker API",
    description="API de surveillance d'un noeud Fog",
    version="1.0.0"
)

import os

NODE_PORT = os.getenv("NODE_PORT", "8001")
NODE_ID = f"{platform.node()}-{NODE_PORT}"

CLOUD_NODE_URL = os.getenv(
    "CLOUD_NODE_URL",
    "http://127.0.0.1:8000"
)

HEARTBEAT_INTERVAL = 5
START_TIME = time.time()


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

    thread = threading.Thread(
        target=send_heartbeat,
        daemon=True
    )

    thread.start()