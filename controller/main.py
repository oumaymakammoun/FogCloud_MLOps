from fastapi import FastAPI
import requests
from datetime import datetime

app = FastAPI(title="Fog-Cloud Controller")

# Addresses of our three Fog Workers

FOG_NODES = {
    "fog_node_1": "http://127.0.0.1:8001",
    "fog_node_2": "http://127.0.0.1:8002",
    "fog_node_3": "http://127.0.0.1:8003"
    # "fog_node_2": "http://192.168.43.94:8001",
    # "fog_node_3": "http://192.168.43.188:8002"
}


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
