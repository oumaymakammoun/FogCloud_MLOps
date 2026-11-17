# config.py

# Choose the role of this machine: "controller" or "worker"
ROLE = "controller"


# ==========================================
# CONTROLLER CONFIGURATION
# ==========================================

# Address on which the Controller API listens.
# 0.0.0.0 means it accepts connections from other devices.
CONTROLLER_HOST = "0.0.0.0"
CONTROLLER_PORT = 8000

# How often the Controller checks each worker (seconds)
HEARTBEAT_INTERVAL = 3

# Maximum time without a successful heartbeat before marking OFFLINE
HEARTBEAT_TIMEOUT = 10


# ==========================================
# WORKER CONFIGURATION
# ==========================================

# Give every Fog node a unique ID.
WORKER_ID = "fog-node-1"

# Address on which the Worker API listens.
WORKER_HOST = "0.0.0.0"
WORKER_PORT = 8001


# ==========================================
# FOG NODES NETWORK CONFIGURATION
# Used by the Controller
# ==========================================

WORKERS = {
    "fog-node-1": "http://192.168.1.101:8001",
    "fog-node-2": "http://192.168.1.102:8001",
    "fog-node-3": "http://192.168.1.103:8001",
}