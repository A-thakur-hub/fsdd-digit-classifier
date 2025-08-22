# src/mic_probe.py
import sys, numpy as np, sounddevice as sd, soundfile as sf, math

def dbfs(x: np.ndarray) -> float:
    x = x.astype("float32")
    rms = float(np.sqrt((x**2).mean() + 1e-12))
    return 20.0 * math.log10(rms + 1e-12)

def main():
    device = int(sys.argv[1]) if len(sys.argv) > 1 else None
    info = sd.query_devices(device, "input") if device is not None else sd.query_devices(kind="input")
    sr = int(info.get("default_samplerate") or 44100) or 44100
    print(f"Using device={device} ({info['name']}), sr={sr}")
    print("Recording 5s… speak now")

    rec = sd.rec(int(5*sr), samplerate=sr, channels=1, dtype="float32", device=device)
    sd.wait()
    x = rec.ravel()
    print(f"Level: {dbfs(x):.1f} dBFS")
    sf.write("mic_test.wav", x, sr)
    print("Wrote mic_test.wav — play it; you should hear your voice.")

if __name__ == "__main__":
    main()
