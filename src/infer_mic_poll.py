import time, numpy as np, joblib, sounddevice as sd, librosa
from src.features import extract_logmel_feat

MODEL_PATH = "artifacts/logmel_logreg.joblib"
MIC_INDEX  = 1          # <-- set to a real input index from your list (try 1, 5, 12, 20, or 30)
WIN_SEC    = 0.9        # ~1s window
TARGET_SR  = 8000       # features expect 8k
DEFAULT_SR = 44100      # safe default for Windows devices

def main():
    model = joblib.load(MODEL_PATH)

    # Query device capabilities
    info = sd.query_devices(MIC_INDEX, "input")
    sr = int(info.get("default_samplerate") or DEFAULT_SR) or DEFAULT_SR
    ch = int(info.get("max_input_channels") or 1)
    print(f'Using device {MIC_INDEX} "{info["name"]}"  sr={sr}  channels={ch}')
    print("Press Ctrl+C to stop. Speak a digit…")

    while True:
        # Record a short chunk
        x = sd.rec(int(WIN_SEC * sr), samplerate=sr, channels=ch, dtype="float32", device=MIC_INDEX)
        sd.wait()

        # Mix to mono (handles silent channels in arrays)
        x = x.mean(axis=1) if ch > 1 else x.ravel()

        # Resample to 8 kHz for our feature extractor
        if sr != TARGET_SR:
            x = librosa.resample(x, orig_sr=sr, target_sr=TARGET_SR)

        # Features + predict
        feats = extract_logmel_feat(x, TARGET_SR).reshape(1, -1)
        pred  = int(model.predict(feats)[0])
        print(f"\rPredicted digit: {pred}   (sr={sr}, ch={ch})", end="", flush=True)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
    except FileNotFoundError:
        print("Model not found. Run: python -m src.train_baseline")
