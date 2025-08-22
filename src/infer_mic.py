import numpy as np
import joblib
import queue
import time

import librosa  # for resampling
from src.features import extract_logmel_feat

MIC_INDEX = 1  # Set to an integer to force a specific device like 5
MODEL_PATH = "artifacts/logmel_logreg.joblib"
TARGET_SR = 8000
BUFFER_SEC = 0.9       # ~1s context
BLOCK_SEC = 0.30       # 300 ms blocks to avoid overflow and stabilize input
VAD_DBFS = -60.0       # voice-activity threshold in dBFS (~-60 for quieter speech)

def rms_dbfs(x: np.ndarray) -> float:
    # x expected in [-1,1] float32
    rms = np.sqrt((x.astype(np.float32) ** 2).mean() + 1e-12)
    return 20.0 * np.log10(rms + 1e-12)

def main():
    # Load model
    try:
        model = joblib.load(MODEL_PATH)
        print("Model loaded:", MODEL_PATH)
    except FileNotFoundError:
        print("Model not found. Train first: python -m src.train_baseline")
        return

    # Sounddevice import here so we can show a friendly error if missing
    try:
        import sounddevice as sd
    except Exception:
        print("sounddevice not installed. Install with: pip install sounddevice")
        return

    # Determine mic sample rate
    try:
        in_dev = sd.query_devices(kind="input")
        stream_sr = int(in_dev["default_samplerate"]) if in_dev else TARGET_SR
    except Exception:
        stream_sr = TARGET_SR

    # Shared queue for audio chunks (callback -> main thread)
    q = queue.Queue()

    # Pre-allocate the 8 kHz rolling buffer used for features (model expects 8 kHz)
    feat_buf = np.zeros(int(TARGET_SR * BUFFER_SEC), dtype=np.float32)

    def audio_cb(indata, frames, time_info, status):
        # Keep callback ultra-light: just enqueue the raw float32 mono chunk
        if status:
            # We don't print here; printing can cause more overruns
            pass
        x = indata[:, 0] if indata.ndim > 1 else indata.ravel()
        q.put_nowait(x.copy())

    print(f"Starting mic… device_sr={stream_sr}Hz → resample → {TARGET_SR}Hz.")
    print("Speak a digit (0–9). Ctrl+C to stop.")
    blocksize = max(1, int(stream_sr * BLOCK_SEC))

    try:
        with sd.InputStream(
            device=MIC_INDEX,  # None = default; set an int to force a device
            samplerate=stream_sr,
            channels=1,
            blocksize=blocksize,
            dtype=np.float32,
            callback=audio_cb
        ):
            last_pred_print = 0.0
            while True:
                try:
                    # Pull one chunk from the queue (wait up to 1 second)
                    chunk = q.get(timeout=1.0)
                except queue.Empty:
                    continue

                # Resample chunk from device rate -> 8 kHz if needed
                if stream_sr != TARGET_SR:
                    chunk = librosa.resample(chunk.astype(np.float32),
                                             orig_sr=stream_sr,
                                             target_sr=TARGET_SR)

                # Slide into rolling buffer (vectorized)
                L = len(chunk)
                if L >= feat_buf.size:
                    feat_buf[:] = chunk[-feat_buf.size:]
                else:
                    feat_buf = np.roll(feat_buf, -L)
                    feat_buf[-L:] = chunk

                # Voice activity gate: only predict when above threshold
                level = rms_dbfs(feat_buf)
                now = time.time()
                
                # Print current level every ~0.3s
                if now - last_pred_print > 0.3:
                    print(f"\rLevel: {level:.1f} dBFS", end="", flush=True)
                    last_pred_print = now
                
                #if level < VAD_DBFS:
                #    continue

                # Build features + predict
                feats = extract_logmel_feat(feat_buf, TARGET_SR).reshape(1, -1)
                pred = int(model.predict(feats)[0])

                # Print prediction
                print(f"\rPredicted digit: {pred} (level {level:.1f} dBFS)  ", end="", flush=True)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as e:
        print("\nAudio error:", e)
        print("Tips:")
        print("- Try increasing BLOCK_SEC to 0.25 or 0.30")
        print("- Pick a specific input device index (see below)")
        print("- Ensure microphone access is allowed in Windows Privacy settings")

if __name__ == "__main__":
    main()
