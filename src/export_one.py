from datasets import load_dataset, Audio
import soundfile as sf


def main():
    """Export first example from FSDD dataset to WAV file."""
    # Load dataset and cast audio to 8kHz
    dataset = load_dataset("mteb/free-spoken-digit-dataset", split="train")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=8000))
    
    # Get first example
    example = dataset[0]
    audio_array = example["audio"]["array"]
    label = example["label"]
    
    # Write to WAV file
    sf.write("sample.wav", audio_array, 8000)
    
    print(f"Exported first example to sample.wav")
    print(f"True label: {label}")


if __name__ == "__main__":
    main()
