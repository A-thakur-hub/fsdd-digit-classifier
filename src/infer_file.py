# src/infer_file.py
import sys
import time
import numpy as np
import soundfile as sf
import librosa
import joblib
from src.features import extract_logmel_feat

MODEL_PATH = "artifacts/logmel_logreg.joblib"
TARGET_SR = 8000

def read_mono_resampled(path, target_sr=TARGET_SR):
    y, sr = sf.read(path, always_2d=False, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)  # mono
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return y, sr

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.infer_file <path/to/file.wav>")
        sys.exit(1)
    wav_path = sys.argv[1]
    model = joblib.load(MODEL_PATH)

    y, sr = read_mono_resampled(wav_path, TARGET_SR)
    
    # Measure end-to-end latency (features + predict)
    start_time = time.perf_counter()
    feats = extract_logmel_feat(y, sr).reshape(1, -1)
    pred = int(model.predict(feats)[0])
    end_time = time.perf_counter()
    
    # Calculate confidence score if available
    confidence = "N/A"
    try:
        probs = model.predict_proba(feats)[0]
        confidence = f"{probs[pred] * 100:.1f}%"
    except (AttributeError, IndexError):
        pass  # Fall back to N/A if predict_proba not available
    
    # Calculate latency in milliseconds
    latency_ms = (end_time - start_time) * 1000
    
    print(f"Predicted digit: {pred} | Confidence: {confidence} | End-to-end: {latency_ms:.2f} ms")

if __name__ == "__main__":
    main()
