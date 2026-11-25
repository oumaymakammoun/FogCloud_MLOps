import base64
import json
import os
import threading
import time
from typing import Optional
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Imports de tes modules (Base de données + Scheduler)
from core.database import MLOpsDatabase
from integrations.scheduler_stub import select_task_for_worker

app = FastAPI(title="Cloud Principal API (Template d'intégration)")

# 1. Connexion à la base SQLite partagée (mode WAL)
db = MLOpsDatabase()

CLOUD_ID = os.getenv("PRIMARY_CLOUD_ID", "cloud-main")
STANDBY_FOG_URL = os.getenv("STANDBY_FOG_URL", "http://192.168.43.122:8001")
HEARTBEAT_INTERVAL = 5


# --- MODÈLES DE DONNÉES ---
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
    model_weights_b64: str  # Poids sérialisés en base64
    file_ext: str = "pkl"
    metrics: Optional[dict] = None


# --- BOUCLE HEARTBEAT VERS LE STANDBY FOG ---
def send_heartbeats_to_standby():
    """Envoie le signal de vie du Cloud principal au Standby Fog."""
    while True:
        try:
            requests.post(
                f"{STANDBY_FOG_URL}/cloud-heartbeat",
                json={
                    "cloud_id": CLOUD_ID,
                    "timestamp": time.time(),
                    "status": "alive",
                    "is_primary": True,  # OBLIGATOIRE pour déclencher le Failback
                },
                timeout=3,
            )
        except requests.RequestException as err:
            print(f"[CLOUD HEARTBEAT ERROR] Standby injoignable : {err}")
        time.sleep(HEARTBEAT_INTERVAL)


@app.on_event("startup")
def on_startup():
    # TODO [Équipe Cloud/ML] : Définir ici les partitions initiales du modèle
    db.register_tasks([
        {"task_id": "part_1", "dataset_shard": "shard_1.csv", "total_epochs": 5},
        {"task_id": "part_2", "dataset_shard": "shard_2.csv", "total_epochs": 5},
        {"task_id": "part_3", "dataset_shard": "shard_3.csv", "total_epochs": 5},
    ])

    # Envoi immédiat d'un premier heartbeat dès le démarrage (Failback rapide)
    threading.Thread(target=send_heartbeats_to_standby, daemon=True).start()


# --- ENDPOINTS BRANCHÉS SUR TA BASE DE DONNÉES ---
@app.post("/resources-report")
def receive_resources_and_assign_task(report: ResourceReport):
    """
    Reçoit les ressources d'un Worker, les sauvegarde en base,
    interroge le scheduler et renvoie une tâche à entraîner (avec son dernier checkpoint).
    """
    worker_dict = report.model_dump()
    db.upsert_fog_node(worker_dict)

    pending_tasks = db.get_pending_tasks()
    chosen_task = select_task_for_worker(worker_dict, pending_tasks)

    if not chosen_task:
        return {"status": "idle", "assigned_task": None}

    assigned = db.assign_task(chosen_task["task_id"], report.node_id)
    if not assigned:
        return {"status": "idle", "assigned_task": None}

    # Si la tâche avait déjà un checkpoint (ex: reprise après panne), on l'attache
    checkpoint_b64 = None
    raw_bytes = db.load_checkpoint_bytes(assigned["task_id"])
    if raw_bytes:
        checkpoint_b64 = base64.b64encode(raw_bytes).decode("utf-8")

    return {
        "status": "assigned",
        "assigned_task": assigned,
        "resume_checkpoint_b64": checkpoint_b64,
    }


@app.post("/checkpoint")
def receive_checkpoint(payload: CheckpointPayload):
    """Sauvegarde les poids reçus d'un Worker sur le disque partagé et met à jour SQLite."""
    try:
        model_bytes = base64.b64decode(payload.model_weights_b64)
        saved_path = db.save_checkpoint_bytes(
            task_id=payload.task_id,
            epoch=payload.current_epoch,
            model_bytes=model_bytes,
            ext=payload.file_ext,
        )
        updated_task = db.update_checkpoint(
            task_id=payload.task_id,
            worker_id=payload.node_id,
            current_epoch=payload.current_epoch,
            checkpoint_path=saved_path,
            metrics_json=json.dumps(payload.metrics) if payload.metrics else None,
        )
        return {"status": "checkpoint_saved", "task": updated_task}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))