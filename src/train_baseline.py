import os
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import joblib

from src.dataio import load_fsdd
from src.features import extract_logmel_feat


def main():
    """Main training function."""
    print("Loading FSDD dataset...")
    dataset = load_fsdd(sr=8000)
    
    print("Extracting log-mel features...")
    X = []
    y = []
    
    for sample in dataset:
        audio = sample["audio"]["array"]
        label = int(sample["label"])
        features = extract_logmel_feat(audio, sr=8000)
        X.append(features)
        y.append(label)
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    
    # Split with stratified 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Create pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(solver="lbfgs", max_iter=300, random_state=42))
    ])
    
    print("Training model...")
    pipeline.fit(X_train, y_train)
    
    # Predictions
    y_pred = pipeline.predict(X_test)
    
    # Evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    
    print(f"\n=== EVALUATION RESULTS ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    
    print(f"\n=== CONFUSION MATRIX ===")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    print(f"\n=== CLASSIFICATION REPORT ===")
    print(classification_report(y_test, y_pred))
    
    # Latency measurement
    print(f"\n=== LATENCY MEASUREMENT ===")
    sample_features = X_test[0:1]
    
    # Warm-up
    for _ in range(10):
        _ = pipeline.predict(sample_features)
    
    # Measure latency
    times = []
    for _ in range(100):
        start_time = time.time()
        _ = pipeline.predict(sample_features)
        end_time = time.time()
        times.append((end_time - start_time) * 1000)  # Convert to ms
    
    avg_latency = np.mean(times)
    print(f"Average prediction latency: {avg_latency:.2f} ms")
    
    # Save model
    print(f"\n=== SAVING MODEL ===")
    os.makedirs("artifacts", exist_ok=True)
    model_path = "artifacts/logmel_logreg.joblib"
    joblib.dump(pipeline, model_path)
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()
