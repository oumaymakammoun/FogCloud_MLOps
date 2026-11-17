
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
