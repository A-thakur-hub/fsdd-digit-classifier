# scripts/verify.ps1
# One-shot setup + run for Windows PowerShell

$ErrorActionPreference = "Stop"

Write-Host "== Creating venv =="
if (!(Test-Path ".venv")) {
  python -m venv .venv
}
Write-Host "== Activating venv =="
.\.venv\Scripts\Activate.ps1

Write-Host "== Upgrading pip and installing requirements =="
python -m pip install --upgrade pip
pip install -r requirements.txt

Write-Host "== Training (saves artifacts/logmel_logreg.joblib) =="
python -m src.train_baseline

Write-Host "== Exporting 10 sample wavs to debug_samples/ =="
python -m src.inspect_dataset

Write-Host "== File inference (example: 7) =="
python -m src.infer_file debug_samples\digit_7.wav

Write-Host "== Random subset sanity check (10 clips) =="
python -m src.debug_predict_random 10

Write-Host "`nAll done"
