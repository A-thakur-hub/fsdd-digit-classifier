# FSDD Digit Classifier

A fast, lightweight spoken digit classifier that converts audio input to predicted digits (0-9) using machine learning.

## Features

- **Audio Processing**: Real-time microphone input and WAV file support
- **Feature Extraction**: 40-mel log spectrogram with delta features (240-dimensional)
- **Machine Learning**: Logistic Regression with StandardScaler preprocessing
- **High Performance**: Sub-millisecond prediction latency
- **Multiple Input Methods**: File-based inference, live microphone, and dataset training

## Performance

- **Accuracy**: 97.04%
- **Macro-F1**: 97.04%
- **Latency**: ~0.15ms (classifier only), few ms end-to-end
- **Model Size**: Lightweight (few MB)

## Architecture

```
Audio Input → Preprocessing → Feature Extraction → ML Model → Prediction
     ↓              ↓              ↓              ↓          ↓
  8kHz Mono → Silence Trim → Log-Mel + Δ + ΔΔ → Logistic → Digit (0-9)
                                    ↓
                              240-D Features
```

## Project Structure

```
fsdd-digit-classifier/
├── src/                    # Source code
│   ├── dataio.py          # Dataset loading (Hugging Face)
│   ├── features.py        # Audio feature extraction
│   ├── train_baseline.py  # Model training script
│   ├── infer_file.py      # WAV file inference
│   ├── infer_mic.py       # Live microphone inference
│   └── export_one.py      # Sample export utility
├── artifacts/              # Trained models
├── scripts/                # Utility scripts
└── requirements.txt        # Python dependencies
```

## Installation

### Prerequisites
- Python 3.8+
- Microphone (for live inference)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/A-thakur-hub/fsdd-digit-classifier.git
   cd fsdd-digit-classifier
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   
   # Windows
   .\.venv\Scripts\Activate.ps1
   
   # macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Train the Model

Train a new model on the FSDD dataset:

```bash
python -m src.train_baseline
```

This will:
- Download the free-spoken-digit-dataset from Hugging Face
- Extract log-mel features from audio samples
- Train a logistic regression model
- Save the model to `artifacts/logmel_logreg.joblib`
- Display performance metrics

### 2. Predict from WAV File

Classify a spoken digit from an audio file:

```bash
python -m src.infer_file path/to/your/audio.wav
```

Output: `Predicted digit: 5 | Confidence: 95.2% | End-to-end: 12.34 ms`

### 3. Live Microphone Inference

Real-time digit classification from your microphone:

```bash
python -m src.infer_mic
```

**Features:**
- Continuous audio monitoring
- Voice activity detection
- Real-time predictions
- Audio level monitoring

### 4. Export Sample Audio

Export a sample from the dataset to test:

```bash
python -m src.export_one
```

Creates `sample.wav` for testing.

## Configuration

### Microphone Settings

Edit `src/infer_mic.py` to customize:

```python
MIC_INDEX = 1        # Set to your microphone device index
VAD_DBFS = -60.0     # Voice activity threshold
BLOCK_SEC = 0.30     # Audio block size
```

### Feature Parameters

Modify `src/features.py` for different feature extraction:

```python
n_mels = 40          # Number of mel frequency bins
n_fft = 512          # FFT window size
hop = 128            # Hop length
```

## Technical Details

### Feature Extraction Pipeline

1. **Audio Preprocessing**
   - Load audio at 8kHz
   - Trim silence (top_db=25)
   - Convert to mono if stereo

2. **Spectrogram Generation**
   - Compute mel spectrogram (40 bins)
   - Apply log transformation
   - Add delta (Δ) and delta-delta (ΔΔ) features

3. **Feature Pooling**
   - Concatenate [log-mel; Δ; ΔΔ]
   - Compute mean and standard deviation over time
   - Output: 240-dimensional feature vector

### Model Architecture

```
Input (240-D) → StandardScaler → LogisticRegression → Output (0-9)
```

- **StandardScaler**: Normalizes features to zero mean, unit variance
- **LogisticRegression**: Multi-class classifier with LBFGS solver
- **Training**: 80/20 stratified split, random_state=42

## Dataset

**Free Spoken Digit Dataset (FSDD)**
- **Source**: Hugging Face `mteb/free-spoken-digit-dataset`
- **Content**: Spoken digits 0-9
- **Format**: Audio files with labels
- **Sampling Rate**: 8kHz (converted from original)
- **License**: MIT

## Troubleshooting

### Common Issues

1. **Microphone not working**
   - Check Windows Privacy Settings → Microphone access
   - Verify device index in `MIC_INDEX`
   - Try different `BLOCK_SEC` values

2. **Model not found**
   - Run `python -m src.train_baseline` first
   - Check `artifacts/` folder exists

3. **Audio import errors**
   - Install `soundfile` and `librosa`
   - Check audio file format (WAV recommended)

### Performance Tips

- Increase `BLOCK_SEC` if you see audio overruns
- Use specific microphone device index for stability
- Ensure microphone access permissions are granted

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source. See the repository for license details.

## Acknowledgments

- **Dataset**: [Free Spoken Digit Dataset](https://huggingface.co/datasets/mteb/free-spoken-digit-dataset)
- **Audio Processing**: [librosa](https://librosa.org/)
- **Machine Learning**: [scikit-learn](https://scikit-learn.org/)

---

**Made with love for audio machine learning enthusiasts**
