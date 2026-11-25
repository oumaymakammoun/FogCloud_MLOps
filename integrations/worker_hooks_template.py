import base64
import os
import platform
import threading
import time
from typing import Optional
import psutil
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Imports de tes modules (Résilience + Base partagée + Scheduler)
from core.database import MLOpsDatabase
from core.resilience import ResilienceManager
from integrations.scheduler_stub import select_task_for_worker

app = FastAPI(title="Fog Worker & Standby API (Template d'intégration)")

NODE_PORT = os.getenv("NODE_PORT", "8001")
NODE_ID = f"{platform.node()}-{NODE_PORT}"
MY_LAN_URL = os.getenv("MY_LAN_URL", f"http://192.168.43.122:{NODE_PORT}")
IS_STANDBY = os.getenv("IS_STANDBY", "false").lower() == "true"

PRIMARY_CLOUD_ID = os.getenv("PRIMARY_CLOUD_ID", "cloud-main")
PRIMARY_CLOUD_URL = os.getenv("CLOUD_NODE_URL", "http://192.168.43.122:8002")
PEER_FOG_URLS = [u.strip() for u in os.getenv("PEER_FOG_URLS", "").split(",") if u.strip()]

# 1. Initialisation de tes deux composants
db = MLOpsDatabase()
resilience = ResilienceManager(
    node_id=NODE_ID,
    my_lan_url=MY_LAN_URL,
    primary_cloud_id=PRIMARY_CLOUD_ID,
    primary_cloud_url=PRIMARY_CLOUD_URL,
    peer_fog_urls=PEER_FOG_URLS,
    is_standby=IS_STANDBY,
    db=db,
)

IS_TRAINING = False


# --- MODÈLES PYDANTIC ---
class CloudHeartbeat(BaseModel):
    cloud_id: str
    timestamp: float
    status: str
    is_primary: bool = True


class NewCloudAnnouncement(BaseModel):
    new_cloud_id: str
    new_cloud_url: str
    timestamp: float


class ResourceReport(BaseModel):
    node_id: str
    node_url: Optional[str] = None
    timestamp: float
    cpu_percent: float
    ram_percent: float
    ram_available_mb: float
    cpu_temperature_c: Optional[float] = None


class CheckpointPayload(BaseModel):
    task_id: str
    node_id: str
    current_epoch: int
    model_weights_b64: str
    file_ext: str = "pkl"
    metrics: Optional[dict] = None


# --- ENDPOINTS DE RÉSILIENCE (TON CODE) ---
@app.post("/cloud-heartbeat")
def cloud_heartbeat(hb: CloudHeartbeat):
    return resilience.handle_cloud_heartbeat(hb.cloud_id, hb.is_primary)


@app.post("/new-cloud")
def new_cloud(announcement: NewCloudAnnouncement):
    return resilience.handle_new_cloud_announcement(
        announcement.new_cloud_id, announcement.new_cloud_url
    )


@app.get("/cloud-status")
def cloud_status():
    return resilience.get_status()


# --- ENDPOINTS CLOUD DE SECOURS (ACTIFS UNIQUEMENT APRÈS FAILOVER) ---
@app.post("/resources-report")
def backup_receive_resources(report: ResourceReport):
    if not resilience.is_active_cloud:
        raise HTTPException(status_code=503, detail="Ce noeud n'est pas le Cloud actif")

    worker_dict = report.model_dump()
    db.upsert_fog_node(worker_dict)
    chosen = select_task_for_worker(worker_dict, db.get_pending_tasks())
    if not chosen:
        return {"status": "idle", "assigned_task": None}

    assigned = db.assign_task(chosen["task_id"], report.node_id)
    raw_bytes = db.load_checkpoint_bytes(assigned["task_id"]) if assigned else None
    checkpoint_b64 = base64.b64encode(raw_bytes).decode("utf-8") if raw_bytes else None

    return {
        "status": "assigned",
        "assigned_task": assigned,
        "resume_checkpoint_b64": checkpoint_b64,
    }


@app.post("/checkpoint")
def backup_receive_checkpoint(payload: CheckpointPayload):
    if not resilience.is_active_cloud:
        raise HTTPException(status_code=503, detail="Ce noeud n'est pas le Cloud actif")

    model_bytes = base64.b64decode(payload.model_weights_b64)
    saved_path = db.save_checkpoint_bytes(
        payload.task_id, payload.current_epoch, model_bytes, payload.file_ext
    )
    updated = db.update_checkpoint(
        payload.task_id,
        payload.node_id,
        payload.current_epoch,
        saved_path,
        str(payload.metrics) if payload.metrics else None,
    )
    return {"status": "checkpoint_saved", "task": updated}


# --- EMPLACEMENT VIDE : ENTRAÎNEMENT ML SUR LE WORKER ---
def run_ml_training(task: dict, resume_checkpoint_b64: Optional[str]):
    """
    [EMPLACEMENT VIDE - À COMPLÉTER PAR L'ÉQUIPE ML / WORKER]
    """
    global IS_TRAINING
    IS_TRAINING = True
    try:
        start_epoch = task["current_epoch"] + 1
        total_epochs = task["total_epochs"]

        # 1. Charger les poids précédents si resume_checkpoint_b64 n'est pas None
        initial_weights = (
            base64.b64decode(resume_checkpoint_b64) if resume_checkpoint_b64 else None
        )

        # TODO [Équipe ML] : Charger le dataset (task["dataset_shard"]) et initial_weights

        for epoch in range(start_epoch, total_epochs + 1):
            # TODO [Équipe ML] : Exécuter 1 epoch d'entraînement et sérialiser le modèle en bytes
            time.sleep(2)  # Simulation du calcul
            fake_weights = f"weights_{task['task_id']}_epoch_{epoch}".encode("utf-8")

            # Envoi du checkpoint vers resilience.current_cloud_url (Cloud ou Standby automatiquement)
            requests.post(
                f"{resilience.current_cloud_url}/checkpoint",
                json={
                    "task_id": task["task_id"],
                    "node_id": NODE_ID,
                    "current_epoch": epoch,
                    "model_weights_b64": base64.b64encode(fake_weights).decode("utf-8"),
                    "metrics": {"epoch": epoch},
                },
                timeout=5,
            )
    finally:
        IS_TRAINING = False


def resource_reporting_loop():
    """Envoie les ressources au Cloud actif et lance l'entraînement si une tâche est reçue."""
    while True:
        if not resilience.is_active_cloud:
            payload = {
                "node_id": NODE_ID,
                "node_url": MY_LAN_URL,
                "timestamp": time.time(),
                "cpu_percent": psutil.cpu_percent(interval=1),
                "ram_percent": psutil.virtual_memory().percent,
                "ram_available_mb": round(psutil.virtual_memory().available / (1024 * 1024), 2),
                "cpu_temperature_c": None,  # TODO [Équipe Worker] : brancher get_cpu_temperature()
            }
            try:
                res = requests.post(
                    f"{resilience.current_cloud_url}/resources-report",
                    json=payload,
                    timeout=5,
                )
                data = res.json()
                if data.get("status") == "assigned" and not IS_TRAINING:
                    threading.Thread(
                        target=run_ml_training,
                        args=(data["assigned_task"], data.get("resume_checkpoint_b64")),
                        daemon=True,
                    ).start()
            except requests.RequestException as err:
                print(f"[WORKER ERROR] ({resilience.current_cloud_url}) {err}")
        time.sleep(10)


@app.on_event("startup")
def on_startup():
    resilience.start()
    threading.Thread(target=resource_reporting_loop, daemon=True).start()