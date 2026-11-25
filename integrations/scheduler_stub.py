from typing import Optional


def select_task_for_worker(worker_resources: dict, pending_tasks: list[dict]) -> Optional[dict]:
    """
    [EMPLACEMENT VIDE - À COMPLÉTER PAR L'ÉQUIPE SCHEDULER / LOAD BALANCING]

    Cette fonction est appelée par le Cloud Principal (ou par le Standby Fog en cas de panne)
    à chaque fois qu'un Worker envoie ses ressources disponibles.

    Args:
        worker_resources (dict): Métriques envoyées par le Worker, par exemple :
            {
                "node_id": "DESKTOP-XYZ-8003",
                "cpu_percent": 25.4,
                "ram_percent": 60.0,
                "ram_available_mb": 2048.5,
                "cpu_temperature_c": 58.0
            }
        pending_tasks (list[dict]): Liste des tâches disponibles issues de MLOpsDatabase,
            triées avec les tâches déjà entamées (current_epoch > 0) en priorité :
            [
                {
                    "task_id": "part_1",
                    "dataset_shard": "shard_1.csv",
                    "status": "PENDING",
                    "current_epoch": 3,
                    "total_epochs": 10,
                    "checkpoint_path": "C:/.../part_1_epoch_3.pkl"
                },
                ...
            ]

    Returns:
        dict | None: La tâche sélectionnée à assigner à ce Worker, ou None si aucune
        tâche ne doit lui être attribuée (ex: surcharge CPU/RAM ou température trop haute).
    """
    if not pending_tasks:
        return None

    # ------------------------------------------------------------------
    # TODO [Équipe Scheduler] : Remplacer cette logique par vos règles
    # de décision selon cpu_percent, ram_available_mb et cpu_temperature_c.
    # ------------------------------------------------------------------

    # Exemple minimal par défaut (FIFO) pour permettre les tests :
    ram_available = worker_resources.get("ram_available_mb", 0)
    cpu_temp = worker_resources.get("cpu_temperature_c")

    # Sécurité basique : refuser si RAM < 500 Mo ou Température > 85°C
    if ram_available < 500:
        return None
    if cpu_temp is not None and cpu_temp > 85.0:
        return None

    # Retourne la première tâche disponible
    return pending_tasks[0]