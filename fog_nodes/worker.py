from fastapi import FastAPI
import psutil
import platform
import time

app = FastAPI(
    title="Fog Worker API",
    description="API de surveillance d'un noeud Fog",
    version="1.0.0"
)

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