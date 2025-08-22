# src/debug_predict_random.py
import random, numpy as np, joblib
from datasets import load_dataset, Audio
from src.features import extract_logmel_feat

def main(n=50, seed=0):
    random.seed(seed)
    ds = load_dataset("mteb/free-spoken-digit-dataset", split="train")
    ds = ds.cast_column("audio", Audio(sampling_rate=8000))
    idxs = random.sample(range(len(ds)), n)
    model = joblib.load("artifacts/logmel_logreg.joblib")

    correct = 0
    for i in idxs:
        ex = ds[i]
        y = int(ex["label"])
        x = extract_logmel_feat(ex["audio"]["array"], ex["audio"]["sampling_rate"]).reshape(1, -1)
        p = int(model.predict(x)[0])
        print(f"idx={i:4d}  true={y}  pred={p}")
        correct += (p == y)
    print(f"\nSubset accuracy on {n} random samples: {correct/n:.3f}")

if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    main(n=n)
