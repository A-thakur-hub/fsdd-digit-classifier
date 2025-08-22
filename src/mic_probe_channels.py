# src/mic_probe_channels.py
import sys, math, numpy as np, sounddevice as sd, soundfile as sf

def dbfs(x: np.ndarray) -> float:
    x = x.astype("float32")
    rms = float(np.sqrt((x**2).mean() + 1e-12))
    return 20.0 * math.log10(rms + 1e-12)

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.mic_probe_channels <device_index>")
        return
    dev = int(sys.argv[1])
    info = sd.query_devices(dev, "input")
    sr = int(info.get("default_samplerate") or 44100)
    ch = int(info.get("max_input_channels") or 1)
    print(f"Device {dev}: {info['name']}  sr={sr}  channels={ch}")
    dur = 4.0
    print("Recording 4s on ALL channels… speak now")

    rec = sd.rec(int(dur*sr), samplerate=sr, channels=ch, dtype="float32", device=dev)
    sd.wait()

    for c in range(ch):
        x = rec[:, c]
        level = dbfs(x)
        print(f"  ch{c}: {level:.1f} dBFS")
        sf.write(f"mic_dev{dev}_ch{c}.wav", x, sr)
    print("Listen to the wavs to see which channel actually has your voice.")
if __name__ == "__main__":
    main()
