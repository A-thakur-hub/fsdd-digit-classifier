from datasets import load_dataset, Audio


def load_fsdd(split="train", sr=8000):
    """
    Load the free-spoken-digit-dataset from Hugging Face.
    
    Args:
        split (str): Dataset split to load ("train", "test", or "validation")
        sr (int): Target sampling rate for audio conversion
        
    Returns:
        Dataset: Processed dataset with audio cast to target sampling rate
    """
    # Load the dataset from Hugging Face
    dataset = load_dataset("mteb/free-spoken-digit-dataset", split=split)
    
    # Cast audio column to target sampling rate
    dataset = dataset.cast_column("audio", Audio(sampling_rate=sr))
    
    return dataset


if __name__ == "__main__":
    ds = load_fsdd()
    ex = ds[0]
    print("rows:", len(ds))
    print("label:", int(ex["label"]))
    print("audio sr:", ex["audio"]["sampling_rate"], "shape:", ex["audio"]["array"].shape)
