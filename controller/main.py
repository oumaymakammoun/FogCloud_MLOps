from fastapi import FastAPI
import requests
from datetime import datetime
from fastapi import HTTPException
from pydantic import BaseModel
import time

app = FastAPI(title="Fog-Cloud Controller")

# Addresses of our three Fog Workers

FOG_NODES = {
    "fog_node_1": "http://127.0.0.1:8001",
    "fog_node_2": "http://127.0.0.1:8002",
    "fog_node_3": "http://127.0.0.1:8003"
    # "fog_node_2": "http://192.168.43.94:8001",
    # "fog_node_3": "http://192.168.43.188:8002"
}

HEARTBEAT_TIMEOUT = 15
NODE_HEARTBEATS = {}

class Heartbeat(BaseModel):
    node_id: str
    timestamp: float
    status: str


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