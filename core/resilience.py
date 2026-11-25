import threading
import time
from typing import Callable, Optional
import requests

from core.database import MLOpsDatabase


class ResilienceManager:
    """
    Gère le cycle complet de tolérance aux pannes (Failover + Failback)
    pour un nœud Fog (Standby ou Worker standard).
    """

    def __init__(
        self,
        node_id: str,
        my_lan_url: str,
        primary_cloud_id: str,
        primary_cloud_url: str,
        peer_fog_urls: list[str],
        is_standby: bool = False,
        db: Optional[MLOpsDatabase] = None,
        heartbeat_timeout: float = 15.0,
        heartbeat_interval: float = 5.0,
        required_recovery_heartbeats: int = 3,
        on_become_cloud: Optional[Callable[[], None]] = None,
        on_step_down: Optional[Callable[[], None]] = None,
    ):
        self.node_id = node_id
        # IMPORTANT : doit être l'IP réseau (ex: http://192.168.43.122:8001), jamais 127.0.0.1
        self.my_lan_url = my_lan_url.rstrip("/")

        self.primary_cloud_id = primary_cloud_id
        self.primary_cloud_url = primary_cloud_url.rstrip("/")
        self.peer_fog_urls = [u.rstrip("/") for u in peer_fog_urls if u.strip()]

        self.is_standby = is_standby
        self.db = db or MLOpsDatabase()

        # Paramètres temporels
        self.heartbeat_timeout = heartbeat_timeout
        self.heartbeat_interval = heartbeat_interval
        self.required_recovery_heartbeats = required_recovery_heartbeats

        # Callbacks optionnels pour brancher la logique de tes coéquipiers
        self.on_become_cloud = on_become_cloud
        self.on_step_down = on_step_down

        # État interne protégé par un verrou
        self._lock = threading.Lock()
        self.current_cloud_url: str = self.primary_cloud_url
        self.is_active_cloud: bool = False
        self.last_primary_heartbeat: Optional[float] = None
        self._consecutive_primary_heartbeats: int = 0
        self._running: bool = False

    # ------------------------------------------------------------------
    # 1. DÉMARRAGE DES THREADS DE SURVEILLANCE
    # ------------------------------------------------------------------

    def start(self):
        """Lance les threads de surveillance (Watchdog + Heartbeats de secours)."""
        if self._running:
            return
        self._running = True

        threading.Thread(target=self._watch_cloud_health_loop, daemon=True).start()
        threading.Thread(target=self._send_backup_heartbeats_loop, daemon=True).start()
        role = "STANDBY CLOUD" if self.is_standby else "FOG WORKER"
        print(f"[RESILIENCE] Démarré sur {self.node_id} (Rôle: {role} | Cible: {self.current_cloud_url})")

    def stop(self):
        self._running = False

    # ------------------------------------------------------------------
    # 2. RÉCEPTION DES ÉVÉNEMENTS RÉSEAU (APPELÉS PAR FASTAPI)
    # ------------------------------------------------------------------

    def handle_cloud_heartbeat(self, cloud_id: str, is_primary: bool = True) -> dict:
        """
        À appeler dans la route POST /cloud-heartbeat.
        Détecte que le Cloud est vivant et déclenche le Failback si le Cloud principal revient.
        """
        with self._lock:
            if is_primary or cloud_id == self.primary_cloud_id:
                self.last_primary_heartbeat = time.time()
                self._consecutive_primary_heartbeats += 1

                # Si ce Standby avait pris le relais, vérifier le retour stable du Cloud principal
                if self.is_standby and self.is_active_cloud:
                    print(
                        f"[FAILBACK CHECK] Heartbeat du Cloud principal reçu "
                        f"({self._consecutive_primary_heartbeats}/{self.required_recovery_heartbeats})"
                    )
                    if self._consecutive_primary_heartbeats >= self.required_recovery_heartbeats:
                        self._demote_to_standby(cloud_id)
            else:
                # Heartbeat émis par le Standby pendant qu'il assure le rôle de Cloud
                pass

        return {"status": "alive", "active_cloud_url": self.current_cloud_url}

    def handle_new_cloud_announcement(self, new_cloud_id: str, new_cloud_url: str) -> dict:
        """
        À appeler dans la route POST /new-cloud sur les Workers distants.
        Met à jour dynamiquement l'adresse du Cloud actif.
        """
        clean_url = new_cloud_url.rstrip("/")
        with self._lock:
            if clean_url != self.my_lan_url:
                self.is_active_cloud = False
                self.current_cloud_url = clean_url
                print(f"[CLUSTER SWITCH] Nouvelle cible Cloud active : {new_cloud_id} -> {self.current_cloud_url}")

        return {"status": "updated", "active_cloud_url": self.current_cloud_url}

    def get_status(self) -> dict:
        """Retourne l'état complet de résilience (pour route GET /cloud-status)."""
        with self._lock:
            elapsed = (
                round(time.time() - self.last_primary_heartbeat, 2)
                if self.last_primary_heartbeat is not None
                else None
            )
            return {
                "node_id": self.node_id,
                "is_standby": self.is_standby,
                "is_active_cloud": self.is_active_cloud,
                "active_cloud_url": self.current_cloud_url,
                "primary_cloud_url": self.primary_cloud_url,
                "last_primary_heartbeat_ago_sec": elapsed,
            }

    # ------------------------------------------------------------------
    # 3. LOGIQUE INTERNE : FAILOVER (PROMOTION) & FAILBACK (RÉTROGRADATION)
    # ------------------------------------------------------------------

    def _promote_to_cloud(self):
        """FAILOVER : Le Standby devient le Cloud actif et récupère la base SQLite."""
        with self._lock:
            if self.is_active_cloud:
                return
            self.is_active_cloud = True
            self.current_cloud_url = self.my_lan_url
            self._consecutive_primary_heartbeats = 0

        print(f"\n[FAILOVER TRIGGERED] Cloud principal hors ligne (> {self.heartbeat_timeout}s) !")
        print(f"[FAILOVER] {self.node_id} prend le contrôle sur {self.my_lan_url}")

        # 1. Récupérer les tâches ML interrompues dans la base partagée SQLite
        recovered = self.db.recover_interrupted_tasks(stale_timeout_sec=int(self.heartbeat_timeout))
        if recovered:
            print(f"[FAILOVER DB] {len(recovered)} tâche(s) interrompue(s) remise(s) en file d'attente.")

        # 2. Informer tous les nœuds Fog distants de pointer vers ce Standby
        self._broadcast_new_cloud(self.node_id, self.my_lan_url)

        # 3. Exécuter le hook métier si défini par l'équipe
        if self.on_become_cloud:
            self.on_become_cloud()

    def _demote_to_standby(self, primary_cloud_id: str):
        """
        FAILBACK : Le Cloud principal est revenu de façon stable.
        Le Standby lui rend sa place et redirige tous les Workers distants.
        Note : appelé uniquement sous self._lock.
        """
        if not self.is_active_cloud:
            return

        self.is_active_cloud = False
        self.current_cloud_url = self.primary_cloud_url
        self._consecutive_primary_heartbeats = 0

        print(f"\n[FAILBACK TRIGGERED] Retour confirmé du Cloud principal ({primary_cloud_id}) !")
        print(f"[FAILBACK] {self.node_id} redevient Standby et redirige le cluster vers {self.primary_cloud_url}")

        # Informer les pairs hors du verrou réseau pour éviter de bloquer les requêtes entrantes
        threading.Thread(
            target=self._broadcast_new_cloud,
            args=(primary_cloud_id, self.primary_cloud_url),
            daemon=True,
        ).start()

        if self.on_step_down:
            self.on_step_down()

    def _broadcast_new_cloud(self, cloud_id: str, cloud_url: str):
        """Diffuse l'adresse du nouveau chef à tous les nœuds Fog du cluster."""
        payload = {
            "new_cloud_id": cloud_id,
            "new_cloud_url": cloud_url,
            "timestamp": time.time(),
        }
        for peer_url in self.peer_fog_urls:
            if peer_url == self.my_lan_url:
                continue
            try:
                res = requests.post(f"{peer_url}/new-cloud", json=payload, timeout=3)
                print(f"[CLUSTER NOTIFY] {peer_url} -> {cloud_url} (HTTP {res.status_code})")
            except requests.RequestException as err:
                print(f"[CLUSTER NOTIFY ERROR] Échec vers {peer_url} : {err}")

    # ------------------------------------------------------------------
    # 4. BOUCLES D'ARRIÈRE-PLAN (WATCHDOG & HEARTBEATS SECOURS)
    # ------------------------------------------------------------------

    def _watch_cloud_health_loop(self):
        """Surveille le timeout des heartbeats du Cloud principal."""
        while self._running:
            time.sleep(2)
            should_promote = False

            with self._lock:
                if (
                    self.is_standby
                    and not self.is_active_cloud
                    and self.last_primary_heartbeat is not None
                ):
                    elapsed = time.time() - self.last_primary_heartbeat
                    if elapsed > self.heartbeat_timeout:
                        should_promote = True

            if should_promote:
                self._promote_to_cloud()

    def _send_backup_heartbeats_loop(self):
        """Quand ce Standby est promu Cloud, il envoie des heartbeats aux autres Fog."""
        while self._running:
            with self._lock:
                active = self.is_active_cloud

            if active:
                for peer_url in self.peer_fog_urls:
                    if peer_url == self.my_lan_url:
                        continue
                    try:
                        requests.post(
                            f"{peer_url}/cloud-heartbeat",
                            json={
                                "cloud_id": self.node_id,
                                "timestamp": time.time(),
                                "status": "alive",
                                "is_primary": False,
                            },
                            timeout=3,
                        )
                    except requests.RequestException:
                        pass

            time.sleep(self.heartbeat_interval)