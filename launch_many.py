"""Launch several tasks at once and watch how the nodes share the work.

usage:  python launch_many.py [number_of_tasks] [epochs]

The script talks to the cloud controller. If the cloud goes down while the tasks
run, it switches to the standby fog (acting as the cloud) and keeps following
the same tasks.
"""
import os
import sys
import time

import requests

PRIMARY = os.getenv("CLOUD_URL", "http://localhost:8002")
STANDBY = os.getenv("STANDBY_URL", "http://localhost:8003/acting-cloud")


def find_controller():
    """Return (name, url) of the first reachable controller, or (None, None)."""
    for name, url in (("the cloud", PRIMARY), ("the standby (acting as cloud)", STANDBY)):
        try:
            if requests.get(f"{url}/", timeout=2).ok:
                return name, url
        except requests.RequestException:
            pass
    return None, None


name, BASE = find_controller()
if not BASE:
    sys.exit("No controller reachable: the cloud is down and the standby is not acting yet. "
             "Wait ~15 s after the cloud stopped, then try again.")
print(f"talking to {name}: {BASE}")

n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 200

alphas = [0.0001, 0.001, 0.01, 0.1]
tasks = [{"epochs": epochs, "params": {"alpha": alphas[i % len(alphas)]}} for i in range(n)]

ids = requests.post(f"{BASE}/tasks/launch-many", json={"tasks": tasks}).json()["task_ids"]
print("launched:", ids)

errors = 0
missing = {tid: 0 for tid in ids}
while True:
    try:
        summary = requests.get(f"{BASE}/tasks/summary", timeout=5).json()
        load = requests.get(f"{BASE}/scheduler/load", timeout=5).json()["active_jobs"]
        errors = 0
    except (requests.RequestException, ValueError):
        errors += 1
        print(f"controller unreachable ({errors}/40), failover in progress? looking for another one...")
        new_name, new_url = find_controller()
        if new_url and new_url != BASE:
            BASE = new_url
            print(f"switched to {new_name}: {BASE}")
        if errors >= 40:
            sys.exit("no controller came back")
        time.sleep(2)
        continue

    print(f"\n--- node load (running shards): {load}")
    status = {}
    for tid in ids:
        t = summary.get(tid)
        if t is None:
            missing[tid] += 1
            status[tid] = "lost" if missing[tid] >= 5 else "unknown"
            print(f"{tid}  {status[tid]:<10} (this controller does not know the task)")
            continue
        missing[tid] = 0
        status[tid] = t["status"]
        print(f"{tid}  {t['status']:<10} {t['epoch']:>9}  acc={t['accuracy']}  "
              f"shares={t['avg_share_percent']}  [{t['phase']}]")
    if all(status[tid] in ("completed", "failed", "lost") for tid in ids):
        break
    time.sleep(1)
print("\nall tasks finished")
