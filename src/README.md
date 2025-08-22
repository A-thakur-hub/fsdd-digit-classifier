# FSDD Digit Classifier — Audio → Digit (0–9)

A tiny, fast prototype that listens to spoken digits and predicts the number.
- **Features:** 40-mel log spectrogram + Δ + ΔΔ, mean/std pooled → **240-D**
- **Model:** StandardScaler → Logistic Regression
- **Dataset:** Hugging Face `mteb/free-spoken-digit-dataset` (parquet, 8 kHz)
- **Results:** Accuracy **0.9704**, Macro-F1 **0.9704**
- **Latency:** Classifier ≈ **0.15 ms**; end-to-end (features+predict) a few ms on CPU

---

## Quickstart

```bash
# 1) Create a virtual env
python -m venv .venv

# Windows
.\.venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate

# 2) Install deps (pin the audio extra to avoid torchcodec)
python -m pip install --upgrade pip
pip install -r requirements.txt
# or: pip install "datasets[audio]==2.19.1" librosa soundfile scikit-learn numpy scipy joblib
