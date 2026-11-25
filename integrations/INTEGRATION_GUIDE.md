# Guide d'Intégration — Résilience, Checkpoints & Base de Données

Ce document décrit les points de connexion entre la couche **Persistance / Résilience** (`core/database.py` et `core/resilience.py`) et les modules **Cloud**, **Scheduler** et **Fog Worker ML**.

---

## 1. Pour l'ami qui code le Scheduler (Load Balancing)
* **Fichier à modifier :** `integrations/scheduler_stub.py`
* **Ce qu'il faut faire :** Remplir le corps de la fonction :
  `select_task_for_worker(worker_resources: dict, pending_tasks: list[dict]) -> dict | None`
* **Règles à respecter :**
  * Ne pas modifier le nom de la fonction ni ses paramètres (elle est appelée automatiquement par le Cloud et par le Standby Fog en cas de panne).
  * `pending_tasks` est déjà triée pour placer les tâches interrompues (`current_epoch > 0`) en tête de liste.
  * Retourner `None` si le Worker est surchargé (CPU, RAM ou température).

---

## 2. Pour l'ami qui code le Serveur Cloud Principal
* **Base de données partagée :** Utiliser obligatoirement `db = MLOpsDatabase()` (située dans `shared_data/mlops.db` en mode SQLite WAL). Ne pas stocker l'état des tâches uniquement dans un dictionnaire Python en mémoire, sinon le Standby Fog ne pourra rien récupérer en cas de crash.
* **Heartbeat vers le Standby Fog :**
  * Envoyer un `POST` toutes les 5 secondes vers `http://<IP_HOTE>:8001/cloud-heartbeat`.
  * **Obligatoire :** Inclure `"is_primary": true` et `"cloud_id": "cloud-main"` dans le JSON envoyé (c'est ce signal qui déclenche le retour à la normale / *Failback* quand le Cloud redémarre).
* **Gestion des checkpoints :**
  * À la réception d'un checkpoint (`POST /checkpoint`), appeler `db.save_checkpoint_bytes(...)` puis `db.update_checkpoint(...)`.
  * Lors de l'attribution d'une tâche (`POST /resources-report`), si `db.load_checkpoint_bytes(task_id)` renvoie des données, les joindre dans la réponse (`resume_checkpoint_b64`) pour que le Worker reprenne à l'epoch interrompue.

---

## 3. Pour l'ami qui code les Fog Workers & l'entraînement ML
* **URL dynamique du Cloud (CRITIQUE) :**
  * Ne jamais coder en dur `http://192.168.43.122:8002` dans les requêtes `requests.post(...)`.
  * Toujours utiliser **`resilience.current_cloud_url`** (ex. `f"{resilience.current_cloud_url}/checkpoint"`). Cette variable bascule automatiquement vers le Standby lors d'une panne, puis revient vers le Cloud principal lors du Failback.
* **Reprise depuis un checkpoint :**
  * Au démarrage de l'entraînement d'une tâche reçue, vérifier si `resume_checkpoint_b64` est présent. Si oui, charger ces poids et démarrer la boucle à `task["current_epoch"] + 1`.
  * À la fin de **chaque epoch**, envoyer le checkpoint via `POST {resilience.current_cloud_url}/checkpoint`.
* **Variables d'environnement au lancement :**
  * Chaque nœud Fog doit être lancé avec son IP réseau réelle (`MY_LAN_URL`), **jamais `127.0.0.1`**, afin que les machines distantes puissent communiquer entre elles.