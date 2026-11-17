from fastapi import FastAPI
import requests
from datetime import datetime
from fastapi import HTTPException
from pydantic import BaseModel
import time
import threading

app = FastAPI(title="Fog-Cloud Controller")

# Addresses of our three Fog Workers

FOG_NODES = {
    "dell-8001": "http://127.0.0.1:8001",
    "dell-8002": "http://127.0.0.1:8002",
    "dell-8003": "http://127.0.0.1:8003"
    # "fog_node_2": "http://192.168.43.94:8001",
    # "fog_node_3": "http://192.168.43.188:8002"
}


HEARTBEAT_TIMEOUT = 15
NODE_HEARTBEATS = {}
NODE_RESOURCES = {}

STANDBY_NODE_ID = "dell-8002"
CLOUD_HEARTBEAT_INTERVAL = 5
CLOUD_ID = "cloud-controller"

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
            response = requests.get(
                f"{address}/resources",
                timeout=3
            )
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

    thread = threading.Thread(
        target=send_cloud_heartbeat,
        daemon=True
    )

    thread.start()