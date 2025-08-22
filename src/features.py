import numpy as np
import librosa


def extract_logmel_feat(y, sr=8000, n_mels=40, n_fft=512, hop=128):
    """
    Extract log-mel spectrogram features with delta features and temporal pooling.
    
    Args:
        y (np.ndarray): Audio signal
        sr (int): Sampling rate
        n_mels (int): Number of mel frequency bins
        n_fft (int): FFT window size
        hop (int): Hop length for STFT
        
    Returns:
        np.ndarray: Feature vector of length 2 * 3 * n_mels (240)
    """
    # Trim silence
    y_trimmed, _ = librosa.effects.trim(y, top_db=25)
    
    # Return zero vector if audio is empty after trimming
    if len(y_trimmed) == 0:
        return np.zeros(2 * 3 * n_mels, dtype=np.float32)
    
    # Compute log-mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=y_trimmed, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop
    )
    log_mel = np.log(mel_spec + 1e-8)
    
    # Compute delta and delta-delta features
    delta = librosa.feature.delta(log_mel)
    delta_delta = librosa.feature.delta(log_mel, order=2)
    
    # Concatenate features: [log-mel; delta; delta-delta]
    features = np.concatenate([log_mel, delta, delta_delta], axis=0)
    
    # Pool by mean and std over time
    mean_feat = np.mean(features, axis=1)
    std_feat = np.std(features, axis=1)
    
    # Concatenate mean and std features
    result = np.concatenate([mean_feat, std_feat])
    
    return result.astype(np.float32)


if __name__ == "__main__":
    # Test with a simple sine wave
    sr = 8000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    y = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    
    feat = extract_logmel_feat(y, sr)
    print(f"Feature vector shape: {feat.shape}")
    print(f"Feature vector dtype: {feat.dtype}")
    print(f"First 10 values: {feat[:10]}")
