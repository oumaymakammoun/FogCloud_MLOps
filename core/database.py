import os
import sqlite3
import time
from pathlib import Path
from typing import Optional

# Chemins absolus basés sur la racine du projet (C:\FogCloud_MLOps_Project)
BASE_DIR = Path(__file__).resolve().parent.parent
SHARED_DIR = BASE_DIR / "shared_data"
CHECKPOINTS_DIR = SHARED_DIR / "checkpoints"
DEFAULT_DB_PATH = SHARED_DIR / "mlops.db"


class MLOpsDatabase:
    """
    Gère la persistance partagée (SQLite WAL + fichiers checkpoints)
    entre le Cloud Principal et le Standby Fog sur la même machine.
    """

    def __init__(self, db_path: Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        # Création automatique des dossiers shared_data/ et shared_data/checkpoints/
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Ouvre une connexion SQLite configurée pour l'accès concurrent."""
        conn = sqlite3.connect(str(self.db_path), timeout=10.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row  # Permet d'accéder aux colonnes par leur nom (dict)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self):
        """Crée les tables si elles n'existent pas encore."""
        with self._get_connection() as conn:
            # Table des partitions du modèle ML à entraîner
            conn.execute("""
                CREATE TABLE IF NOT EXISTS training_tasks (
                    task_id TEXT PRIMARY KEY,
                    dataset_shard TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'PENDING',
                    assigned_worker TEXT,
                    current_epoch INTEGER NOT NULL DEFAULT 0,
                    total_epochs INTEGER NOT NULL,
                    checkpoint_path TEXT,
                    metrics_json TEXT,
                    updated_at REAL NOT NULL
                )
            """)

            # Table des derniers états connus des nœuds Fog (utile pour le scheduler)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS fog_nodes (
                    node_id TEXT PRIMARY KEY,
                    node_url TEXT,
                    cpu_percent REAL,
                    ram_percent REAL,
                    ram_available_mb REAL,
                    cpu_temperature_c REAL,
                    status TEXT NOT NULL DEFAULT 'ALIVE',
                    last_seen REAL NOT NULL
                )
            """)
            conn.commit()

    # ------------------------------------------------------------------
    # 1. GESTION DES TÂCHES D'ENTRAÎNEMENT (PARTITIONS DU MODÈLE)
    # ------------------------------------------------------------------

    def register_tasks(self, tasks: list[dict]):
        """
        Initialise les partitions du modèle dans la base (ex: au démarrage du projet).
        Ignore les tâches déjà existantes pour ne pas écraser une progression.
        Exemple de task: {"task_id": "part_1", "dataset_shard": "shard_1.csv", "total_epochs": 10}
        """
        now = time.time()
        with self._get_connection() as conn:
            for task in tasks:
                conn.execute("""
                    INSERT OR IGNORE INTO training_tasks
                    (task_id, dataset_shard, status, current_epoch, total_epochs, updated_at)
                    VALUES (?, ?, 'PENDING', 0, ?, ?)
                """, (
                    task["task_id"],
                    task["dataset_shard"],
                    task["total_epochs"],
                    now
                ))
            conn.commit()

    def get_pending_tasks(self) -> list[dict]:
        """
        Retourne toutes les tâches prêtes à être distribuées (PENDING).
        Priorise les tâches déjà entamées (current_epoch > 0) pour finir le travail commencé.
        """
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT * FROM training_tasks
                WHERE status = 'PENDING'
                ORDER BY current_epoch DESC, task_id ASC
            """).fetchall()
            return [dict(row) for row in rows]

    def assign_task(self, task_id: str, worker_id: str) -> Optional[dict]:
        """Marque une tâche comme IN_PROGRESS pour un Worker donné."""
        now = time.time()
        with self._get_connection() as conn:
            conn.execute("""
                UPDATE training_tasks
                SET status = 'IN_PROGRESS',
                    assigned_worker = ?,
                    updated_at = ?
                WHERE task_id = ? AND status = 'PENDING'
            """, (worker_id, now, task_id))
            conn.commit()

            row = conn.execute(
                "SELECT * FROM training_tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
            return dict(row) if row else None

    # ------------------------------------------------------------------
    # 2. SAUVEGARDE ET LECTURE DES CHECKPOINTS
    # ------------------------------------------------------------------

    def save_checkpoint_bytes(self, task_id: str, epoch: int, model_bytes: bytes, ext: str = "pkl") -> str:
        """
        Écrit le fichier binaire du modèle entraîné dans shared_data/checkpoints/
        et retourne son chemin relatif propre.
        """
        filename = f"{task_id}_epoch_{epoch}.{ext}"
        file_path = CHECKPOINTS_DIR / filename

        # Écriture atomique via fichier temporaire pour éviter un fichier corrompu en cas de crash
        temp_path = CHECKPOINTS_DIR / f"{filename}.tmp"
        with open(temp_path, "wb") as f:
            f.write(model_bytes)
        os.replace(temp_path, file_path)

        return str(file_path)

    def update_checkpoint(
        self,
        task_id: str,
        worker_id: str,
        current_epoch: int,
        checkpoint_path: str,
        metrics_json: Optional[str] = None
    ) -> dict:
        """
        Met à jour l'avancement d'une tâche lors de la réception d'un checkpoint.
        Si current_epoch >= total_epochs, la tâche passe automatiquement à COMPLETED.
        """
        now = time.time()
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT total_epochs FROM training_tasks WHERE task_id = ?", (task_id,)
            ).fetchone()

            if not row:
                raise ValueError(f"Tâche inconnue : {task_id}")

            total_epochs = row["total_epochs"]
            new_status = "COMPLETED" if current_epoch >= total_epochs else "IN_PROGRESS"

            conn.execute("""
                UPDATE training_tasks
                SET status = ?,
                    assigned_worker = ?,
                    current_epoch = ?,
                    checkpoint_path = ?,
                    metrics_json = COALESCE(?, metrics_json),
                    updated_at = ?
                WHERE task_id = ?
            """, (
                new_status,
                worker_id,
                current_epoch,
                checkpoint_path,
                metrics_json,
                now,
                task_id
            ))
            conn.commit()

            updated = conn.execute(
                "SELECT * FROM training_tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
            return dict(updated)

    def load_checkpoint_bytes(self, task_id: str) -> Optional[bytes]:
        """Lit les octets du dernier checkpoint sauvegardé pour reprise d'entraînement."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT checkpoint_path FROM training_tasks WHERE task_id = ?", (task_id,)
            ).fetchone()

        if not row or not row["checkpoint_path"]:
            return None

        path = Path(row["checkpoint_path"])
        if not path.exists():
            return None

        return path.read_bytes()

    # ------------------------------------------------------------------
    # 3. REPRISE APRÈS PANNE (UTILISÉ PAR LE STANDBY LORS DU FAILOVER)
    # ------------------------------------------------------------------

    def recover_interrupted_tasks(self, stale_timeout_sec: int = 45) -> list[dict]:
        """
        Repère les tâches restées en IN_PROGRESS sans mise à jour depuis trop longtemps
        (ex: pendant une panne du Cloud ou d'un Worker) et les remet en PENDING
        tout en conservant leur current_epoch et checkpoint_path.
        """
        cutoff = time.time() - stale_timeout_sec
        with self._get_connection() as conn:
            stale_rows = conn.execute("""
                SELECT * FROM training_tasks
                WHERE status = 'IN_PROGRESS' AND updated_at < ?
            """, (cutoff,)).fetchall()

            if stale_rows:
                conn.execute("""
                    UPDATE training_tasks
                    SET status = 'PENDING',
                        assigned_worker = NULL,
                        updated_at = ?
                    WHERE status = 'IN_PROGRESS' AND updated_at < ?
                """, (time.time(), cutoff))
                conn.commit()

            return [dict(r) for r in stale_rows]

    def get_all_tasks_summary(self) -> list[dict]:
        """Retourne l'état complet de toutes les partitions (monitoring / debug)."""
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM training_tasks ORDER BY task_id ASC").fetchall()
            return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # 4. SUIVI DES RESSOURCES DES NŒUDS FOG
    # ------------------------------------------------------------------

    def upsert_fog_node(self, node_data: dict):
        """Enregistre ou met à jour les métriques d'un nœud Fog."""
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO fog_nodes
                (node_id, node_url, cpu_percent, ram_percent, ram_available_mb, cpu_temperature_c, status, last_seen)
                VALUES (?, ?, ?, ?, ?, ?, 'ALIVE', ?)
                ON CONFLICT(node_id) DO UPDATE SET
                    node_url = COALESCE(excluded.node_url, fog_nodes.node_url),
                    cpu_percent = excluded.cpu_percent,
                    ram_percent = excluded.ram_percent,
                    ram_available_mb = excluded.ram_available_mb,
                    cpu_temperature_c = excluded.cpu_temperature_c,
                    status = 'ALIVE',
                    last_seen = excluded.last_seen
            """, (
                node_data["node_id"],
                node_data.get("node_url"),
                node_data.get("cpu_percent"),
                node_data.get("ram_percent"),
                node_data.get("ram_available_mb"),
                node_data.get("cpu_temperature_c"),
                time.time()
            ))
            conn.commit()