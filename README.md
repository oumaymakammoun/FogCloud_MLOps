# Technical Documentation
# Intelligent Fog–Cloud Distributed Training and Fault-Tolerant MLOps Platform

**Project documentation — Current development stage**

## 1. Project overview

The objective of this project is to develop an intelligent Fog–Cloud platform capable of monitoring several Fog computers, distributing machine-learning training tasks according to available resources, and recovering training after failures.

The architecture contains:

- **Cloud Controller:** Monitors Fog nodes and manages the cluster.
- **Fog Workers:** Run on individual computers and report their resources.
- **Training Engine:** Trains and evaluates machine-learning models.
- **Checkpoint System:** Saves training progress so that it can later be restored.
- **MLOps components (planned):** Model validation, versioning, deployment, and rollback.

### 1.1 Current architecture

We currently use two active computers. A third computer was part of the initial design but is switched off for now.

| Component | Computer | Role |
|---|---|---|
| Cloud Controller | Master (dell) | Monitors the Fog cluster |
| Fog Node 1 | Master (dell) | Reports Master resources |
| Fog Node 2 | PC2 (kammoun) | Reports PC2 resources |
| Fog Node 3 | PC3 | Temporarily inactive |

The Master runs both the Cloud Controller and Fog Node 1.

### 1.2 Network configuration

The current controller configuration contains these addresses:

| Node | Address | Port |
|---|---|---:|
| Fog Node 1 | `127.0.0.1` | 8001 |
| Fog Node 2 | `192.168.43.94` | 8001 |
| Fog Node 3 | `192.168.43.188` | 8002 |

These are examples of the current network configuration, not permanent addresses. If a computer receives a new IP address, the controller configuration must be updated.

---

## 2. Project structure

The main project directory is:

`C:\Users\dell1\Desktop\FogCloud_MLOps`

The important folders are:

```text
FogCloud_MLOps/
│
├── controller/
│   └── main.py
│
├── fog_nodes/
│   └── worker.py
│
├── training/
│   └── train_model.py
│
├── models/
│   └── iris_model.joblib
│
├── checkpoints/
│   ├── epoch_1.joblib
│   ├── epoch_2.joblib
│   ├── epoch_3.joblib
│   ├── epoch_4.joblib
│   └── epoch_5.joblib
│
└── venv/
```

**Explanation of the folders:**

- `controller/`: Contains the Cloud Controller application.
- `fog_nodes/`: Contains the API executed by each Fog computer.
- `training/`: Contains the machine-learning training program.
- `models/`: Stores the final trained model.
- `checkpoints/`: Stores intermediate training states.
- `venv/`: Contains the Python virtual environment and installed packages.

The `venv` folder is created separately on each computer. It should not be copied through Git as part of the project.

---

## 3. Development environment and required libraries

The project uses Python and several libraries.

| Library | Purpose |
|---|---|
| FastAPI | Creates the HTTP APIs |
| Uvicorn | Runs the APIs |
| Requests | Allows the Controller to communicate with Fog Workers |
| psutil | Measures CPU and RAM usage |
| scikit-learn | Provides machine-learning algorithms and datasets |
| joblib | Saves models and checkpoints |

### 3.1 Create a virtual environment

Open PowerShell in the project directory:

```powershell
cd "$env:USERPROFILE\Desktop\FogCloud_MLOps"
```

Create the environment:

```powershell
python -m venv venv
```

Activate it:

```powershell
.\venv\Scripts\Activate.ps1
```

When activation succeeds, the terminal displays `(venv)`.

### 3.2 Install the required libraries

```powershell
python -m pip install fastapi uvicorn requests psutil scikit-learn joblib
```

Check the installation:

```powershell
python -c "import sklearn, joblib, psutil, fastapi, uvicorn, requests; print('All packages installed successfully')"
```

Repeat the environment setup on each computer that will run a Fog Worker or training code.

---

## 4. Fog Worker implementation

**File:** `fog_nodes/worker.py`

### 4.1 Complete code

```python
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
```

### 4.2 Explanation

The Fog Worker is a small web service installed on each Fog computer. Its purpose is to provide information about the computer to the Cloud Controller.

**Imports**

- `FastAPI`: Creates the API.
- `psutil`: Collects CPU and memory information.
- `platform`: Retrieves the computer's hostname.
- `time`: Measures how long the worker has been running.

**FastAPI application**

```python
app = FastAPI(...)
```

This creates the API application that Uvicorn will execute.

**Start time**

```python
START_TIME = time.time()
```

Stores the time at which the worker process starts. It is used to calculate uptime.

### 4.3 Available API endpoints

**Endpoint 1: `/`**

```python
@app.get("/")
def home():
```

Returns a simple message confirming that the Fog Worker is running.

Example:

```json
{
  "message": "Fog Worker is running",
  "status": "available"
}
```

**Endpoint 2: `/health`**

```python
@app.get("/health")
def health():
```

Checks the health of the worker and reports its uptime.

Example:

```json
{
  "status": "healthy",
  "uptime_seconds": 120.5
}
```

**Endpoint 3: `/resources`**

```python
@app.get("/resources")
def resources():
```

Returns the computer's resource information:

- `node_name`: Computer hostname.
- `cpu_percent`: CPU utilization percentage.
- `ram_percent`: RAM utilization percentage.
- `ram_available_mb`: Available RAM in megabytes.

The Controller uses this endpoint to monitor each Fog computer.

### 4.4 Run the Fog Worker

On each computer, activate its virtual environment and execute:

```powershell
python -m uvicorn fog_nodes.worker:app --host 0.0.0.0 --port 8001
```

For Fog Node 3, the configured port was `8002`.

**Important:** Keep the terminal open while the worker is running. Closing it stops the worker.

The option `--host 0.0.0.0` allows other computers on the network to access the API, subject to firewall and network configuration.

---

## 5. Cloud Controller implementation

**File:** `controller/main.py`

### 5.1 Complete code

```python
from fastapi import FastAPI
import requests
from datetime import datetime

app = FastAPI(title="Fog-Cloud Controller")

# Addresses of our three Fog Workers

FOG_NODES = {
    "fog_node_1": "http://127.0.0.1:8001",
    "fog_node_2": "http://192.168.43.94:8001",
    "fog_node_3": "http://192.168.43.188:8002"
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
```

### 5.2 Explanation

The Cloud Controller is the central monitoring component. It communicates with the Fog Workers and collects their resource information.

**Imports**

- `FastAPI`: Creates the Controller API.
- `requests`: Sends HTTP requests to the Fog Workers.
- `datetime`: Provides the current date and time.

**Application creation**

```python
app = FastAPI(title="Fog-Cloud Controller")
```

Creates the Controller API.

**Fog node configuration**

```python
FOG_NODES = {
    "fog_node_1": "http://127.0.0.1:8001",
    "fog_node_2": "http://192.168.43.94:8001",
    "fog_node_3": "http://192.168.43.188:8002"
}
```

This dictionary contains the network addresses of the Fog Workers.

The Controller uses these addresses to contact the computers.

**Controller home endpoint**

```python
@app.get("/")
def home():
```

Confirms that the Cloud Controller is running and returns the current time.

**Node monitoring endpoint**

```python
@app.get("/nodes")
def get_nodes():
```

This is the main monitoring function.

Its operation is as follows:

1. Creates an empty dictionary called `results`.
2. Iterates through all configured Fog nodes.
3. Sends an HTTP request to each node's `/resources` endpoint.
4. Waits for a maximum of three seconds.
5. Checks whether the request was successful.
6. Stores the node's status and resource information.
7. If communication fails, records the node as unavailable and stores the error.

The `try/except` structure prevents one unavailable Fog Worker from stopping the monitoring of the other workers.

### 5.3 Run the Cloud Controller

On the Master computer:

```powershell
python -m uvicorn controller.main:app --host 0.0.0.0 --port 8000 --reload
```

The Controller is accessible locally at:

`http://127.0.0.1:8000`

Its monitoring endpoint is:

`http://127.0.0.1:8000/nodes`

---

## 6. Testing communication between computers

After starting the Fog Workers and the Cloud Controller, test the system from the Master computer.

Run:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/nodes
```

A successful result contains information similar to:

```json
{
  "fog_node_1": {
    "status": "available",
    "resources": {
      "node_name": "dell",
      "cpu_percent": 3.9,
      "ram_percent": 90.1,
      "ram_available_mb": 1595.89
    }
  },
  "fog_node_2": {
    "status": "available",
    "resources": {
      "node_name": "kammoun",
      "cpu_percent": 5.3,
      "ram_percent": 85.1,
      "ram_available_mb": 2401.4
    }
  }
}
```

The actual values change according to computer activity.

### 6.1 Meaning of the node status

- `available`: The Controller successfully contacted the worker and received its resource information.
- `unavailable`: The Controller could not communicate with the worker within the configured timeout or received an unsuccessful response.

An unavailable status does not necessarily mean that the computer is switched off. It may also indicate a network, firewall, port, or application problem.

### 6.2 Important endpoint distinction

The Fog Worker provides `/`, `/health`, and `/resources`.

The Cloud Controller provides `/` and `/nodes`.

Therefore, `/nodes` must be requested from the Controller on port `8000`, not from a Fog Worker on port `8001`.

---

## 7. Machine-learning training implementation

**File:** `training/train_model.py`

### 7.1 Complete code

```python
import os
import time
import joblib
import psutil

from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Create folders
os.makedirs("models", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

# Load dataset
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data,
    iris.target,
    test_size=0.2,
    random_state=42,
    stratify=iris.target
)

# Create model
model = SGDClassifier(
    loss="log_loss",
    random_state=42
)

# Training configuration
BATCH_SIZE = 15
EPOCHS = 5

print("FOG-CLOUD DISTRIBUTED TRAINING")
print("Starting training...")

start_time = time.time()


for epoch in range(EPOCHS):

    # Process the complete training dataset in batches
    for start in range(0, len(X_train), BATCH_SIZE):
        end = start + BATCH_SIZE

        X_batch = X_train[start:end]
        y_batch = y_train[start:end]

        model.partial_fit(
            X_batch,
            y_batch,
            classes=[0, 1, 2]
        )

    # Save checkpoint after each complete epoch
    checkpoint_path = f"checkpoints/epoch_{epoch + 1}.joblib"

    joblib.dump(
        {
            "model": model,
            "completed_epoch": epoch + 1,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS
        },
        checkpoint_path
    )

    print(f"Epoch {epoch + 1}/{EPOCHS} completed")
    print(f"Checkpoint saved: {checkpoint_path}")

# Evaluate model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("\nTRAINING RESULTS")
print("Accuracy:", round(accuracy * 100, 2), "%")
print("Training time:", round(time.time() - start_time, 3), "seconds")
print("CPU usage:", psutil.cpu_percent(interval=0.2), "%")
print("RAM usage:", psutil.virtual_memory().percent, "%")

# Save final model
joblib.dump(model, "models/iris_model.joblib")

print("Final model saved successfully!")
```

### 7.2 Explanation of the training program

This program demonstrates the machine-learning training process using the Iris dataset.

At the current stage, it executes locally on the computer where the script is launched. It is not yet distributed across the Fog nodes.

**Step 1: Import libraries**

- `os`: Creates directories.
- `time`: Measures training duration.
- `joblib`: Saves the model and checkpoints.
- `psutil`: Measures CPU and RAM.
- `scikit-learn`: Provides the dataset, model, data splitting, and evaluation functions.

**Step 2: Create output folders**

```python
os.makedirs("models", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)
```

Creates the two folders if they do not already exist.

**Step 3: Load the Iris dataset**

```python
iris = load_iris()
```

The Iris dataset is a small dataset containing measurements of flowers from three different classes.

It contains 150 samples and four numerical features per sample.

**Step 4: Split the dataset**

```python
train_test_split(
    iris.data,
    iris.target,
    test_size=0.2,
    random_state=42,
    stratify=iris.target
)
```

The dataset is divided into:

- 80% for training: 120 samples.
- 20% for testing: 30 samples.

`random_state=42` makes the split reproducible.

`stratify=iris.target` helps preserve the class proportions in both sets.

**Step 5: Create the machine-learning model**

```python
model = SGDClassifier(
    loss="log_loss",
    random_state=42
)
```

The project uses an SGDClassifier.

SGD stands for Stochastic Gradient Descent. It is a learning algorithm that updates model parameters using small batches of data.

The `log_loss` option makes the classifier perform logistic classification.

**Step 6: Configure training**

```python
BATCH_SIZE = 15
EPOCHS = 5
```

- `BATCH_SIZE = 15`: Processes up to 15 training samples per batch.
- `EPOCHS = 5`: Repeats the complete training dataset five times.

With 120 training samples, there are eight batches per epoch.

**Step 7: Train the model**

```python
model.partial_fit(
    X_batch,
    y_batch,
    classes=[0, 1, 2]
)
```

The `partial_fit()` method updates the model incrementally.

Unlike a normal `fit()` call, it allows the model to learn progressively from batches.

The class list is provided so the classifier knows all possible output classes, including during its first update.

**Step 8: Save checkpoints**

After every complete epoch, the program saves:

- The current model.
- The number of completed epochs.
- The batch size.
- The total number of epochs.

For example:

```text
checkpoints/
├── epoch_1.joblib
├── epoch_2.joblib
├── epoch_3.joblib
├── epoch_4.joblib
└── epoch_5.joblib
```

These files preserve intermediate model states.

**Important:** The current code saves checkpoints, but it does not yet automatically load a checkpoint and resume training after a failure. That recovery mechanism remains to be implemented.

**Step 9: Evaluate the model**

```python
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
```

The model predicts the classes of the test samples.

`accuracy_score()` compares the predictions with the actual labels and calculates the percentage of correct predictions.

**Step 10: Save the final model**

```python
joblib.dump(model, "models/iris_model.joblib")
```

Saves the final trained model so it can be loaded and used later.

---

## 8. First successful training experiment

The training script was executed successfully.

Observed results:

| Metric | Result |
|---|---:|
| Dataset | Iris |
| Training samples | 120 |
| Test samples | 30 |
| Batch size | 15 |
| Epochs | 5 |
| Accuracy | 83.33% |
| Training time | 0.137 seconds |
| CPU usage at measurement | 9.8% |
| RAM usage at measurement | 86.7% |
| Checkpoints created | 5 |

These measurements correspond to one execution on the Master computer. They are not yet a comparison between Fog nodes.

The accuracy is the result of this particular model configuration and train/test split. It should not be interpreted as a guarantee of future results.

---

## 9. How to reproduce the current system

Follow these steps in order.

### Step 1 — Prepare each computer

Install Python, create the project directory, create a virtual environment, activate it, and install the required packages.

### Step 2 — Start Fog Node 1

On the Master computer, open a PowerShell window:

```powershell
cd "$env:USERPROFILE\Desktop\FogCloud_MLOps"
.\venv\Scripts\Activate.ps1
python -m uvicorn fog_nodes.worker:app --host 0.0.0.0 --port 8001
```

### Step 3 — Start Fog Node 2

On PC2, open PowerShell and use the actual project directory on that computer:

```powershell
cd "$env:USERPROFILE\Desktop\FogCloud_MLOps"
.\venv\Scripts\Activate.ps1
python -m uvicorn fog_nodes.worker:app --host 0.0.0.0 --port 8001
```

The directory may have a different name on PC2. Use the directory that contains its `fog_nodes` folder.

### Step 4 — Start the Cloud Controller

On the Master computer, open another PowerShell window:

```powershell
cd "$env:USERPROFILE\Desktop\FogCloud_MLOps"
.\venv\Scripts\Activate.ps1
python -m uvicorn controller.main:app --host 0.0.0.0 --port 8000 --reload
```

### Step 5 — Test the cluster

On the Master computer:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/nodes
```

Check that Fog Node 1 and Fog Node 2 report `available`.

### Step 6 — Run the training experiment

On the Master computer:

```powershell
python .\training\train_model.py
```

Check the training results and confirm that the model and checkpoint files have been created.

---

## 10. Current achievements and remaining development

It is important to distinguish between implemented features and planned features.

### Implemented and tested

- Fog Worker API.
- CPU and RAM monitoring.
- Worker health endpoint.
- Cloud Controller API.
- Communication between the Master and PC2.
- Detection of unavailable Fog nodes.
- Local machine-learning training using scikit-learn.
- Batch-based incremental training.
- Checkpoint file creation after each epoch.
- Final model saving.

### Not implemented yet

- Intelligent scheduling based on CPU and RAM availability.
- Sending training tasks from the Controller to Fog Workers.
- Actual distributed training across multiple computers.
- Automatic checkpoint restoration after failure.
- Worker failure detection during an active training task.
- Model validation and version management.
- Automatic rollback to an earlier model version.
- A monitoring dashboard.
- Controlled resource limits and systematic performance experiments.

### Planned architecture

The next development stages will be:

1. **Resource-aware scheduling:** The Controller selects a Fog node according to its available resources.
2. **Distributed execution:** Fog Workers receive and execute training tasks.
3. **Checkpoint recovery:** Training resumes from a saved state after a worker failure.
4. **Fault tolerance:** The Controller detects failures and reallocates work.
5. **MLOps management:** Models are validated, versioned, and deployed.
6. **Performance evaluation:** Compare execution time, accuracy, resource consumption, and recovery time.

---

## 11. Important technical notes

1. **IP addresses can change.** If a Fog computer receives a different IP address, update `FOG_NODES` in `controller/main.py`.
2. **Ports must match.** The Controller must contact the same port on which each worker is running.
3. **Workers must remain running.** Closing a worker's terminal stops its API.
4. **Network access is required.** Windows Firewall and the local network must permit communication between the computers.
5. **Resource consumption must be controlled.** The current computers already have high RAM usage, so experiments should begin with small workloads.
6. **Monitoring is not scheduling.** The Controller currently collects resource information but does not yet decide where training should execute.
7. **Checkpoint saving is not automatic recovery.** The current training code saves model states, but recovery logic still needs to be developed.
8. **Hostname and node ID are different concepts.** The worker reports `platform.node()`, which returns the computer's hostname. The Controller uses its own configured identifiers such as `fog_node_1`.

---

## 12. Conclusion

The first development stage of the Intelligent Fog–Cloud Distributed Training and Fault-Tolerant MLOps Platform has been completed.

We have established communication between the Cloud Controller and two Fog Workers, implemented CPU and RAM monitoring, and developed a local machine-learning training program with batch processing and checkpoint saving.

The current implementation provides the foundation for the next stage: intelligent resource-aware scheduling, distributed training, and fault-tolerant recovery.

The project will evolve progressively from a working monitoring system into a distributed Fog–Cloud MLOps platform.