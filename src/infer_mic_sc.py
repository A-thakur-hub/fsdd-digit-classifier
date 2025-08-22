import soundcard as sc, numpy as np, librosa, joblib, time, math
from src.features import extract_logmel_feat

MODEL_PATH = "artifacts/logmel_logreg.joblib"
WIN_SEC = 0.9
TARGET_SR = 8000
FLOOR_MIN = -60.0
FLOOR_BOOST = 20.0

def dbfs(x):
    x = x.astype(np.float32); rms = float(np.sqrt((x**2).mean() + 1e-12))
    return 20*np.log10(rms + 1e-12)

def main():
    model = joblib.load(MODEL_PATH)
    mic = sc.default_microphone()  # uses the input you pick in Windows Sound settings
    sr = 48000  # common default; resampled later
    print(f'Using microphone: "{mic.name}" at {sr} Hz (WASAPI)')

    # calibrate
    with mic.recorder(samplerate=sr) as rec:
        x0 = rec.record(numframes=int(1.0*sr))  # (N, C)
    x0 = x0.mean(axis=1)
    ambient = dbfs(x0)
    vad_th = max(ambient + FLOOR_BOOST, FLOOR_MIN)
    print(f"Ambient {ambient:.1f} dBFS -> VAD {vad_th:.1f} dBFS")

    print("Speak a digit… Ctrl+C to stop.")
    try:
        with mic.recorder(samplerate=sr) as rec:
            while True:
                x = rec.record(numframes=int(WIN_SEC*sr))  # (N, C)
                x = x.mean(axis=1)
                if sr != TARGET_SR:
                    x = librosa.resample(x, orig_sr=sr, target_sr=TARGET_SR)
                level = dbfs(x)
                if level < vad_th:
                    print(f"\r(silence) {level:.1f} dBFS   ", end="", flush=True)
                    continue
                feats = extract_logmel_feat(x, TARGET_SR).reshape(1, -1)
                pred = int(model.predict(feats)[0])
                print(f"\rPredicted digit: {pred}   (level {level:.1f} dBFS)   ", end="", flush=True)
    except KeyboardInterrupt:
        print("\nStopped.")

if __name__ == "__main__":
    main()
