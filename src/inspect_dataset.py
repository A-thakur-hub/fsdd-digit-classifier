# src/inspect_dataset.py
from collections import Counter
from datasets import load_dataset, Audio
import soundfile as sf
from pathlib import Path

def main():
    ds = load_dataset("mteb/free-spoken-digit-dataset", split="train")
    ds = ds.cast_column("audio", Audio(sampling_rate=8000))

    # label distribution
    counts = Counter(int(x) for x in ds["label"])
    print("Label counts:", dict(sorted(counts.items())))

    # export first example for each digit 0..9
    out = Path("debug_samples"); out.mkdir(exist_ok=True)
    seen = set()
    for ex in ds:
        lab = int(ex["label"])
        if lab in seen: 
            continue
        arr = ex["audio"]["array"]; sr = ex["audio"]["sampling_rate"]
        sf.write(out / f"digit_{lab}.wav", arr, sr)
        seen.add(lab)
        if len(seen) == 10:
            break
    print("Wrote:", sorted(p.name for p in out.glob("*.wav")))

if __name__ == "__main__":
    main()
