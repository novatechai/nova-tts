import datasets
import itertools
from datasets import load_dataset
import time

# Assuming config.py is in the src directory, adjust path if needed
import sys
sys.path.append('./src')
import config

print("--- Starting Data Loading Test Script ---")

# Ensure HF token is set if using private/gated datasets
if not config.HUGGING_FACE_TOKEN:
    print("Warning: Hugging Face token not found in config. Might fail for gated datasets.")

# --- Load Dataset Sample ---
try:
    print(f"Loading dataset: {config.DATASET_NAME}")
    # Use streaming=False if you want to download for faster local iteration
    # but True is better for large datasets on machines with less disk space.
    # Set trust_remote_code=True if prompted by the library for this dataset.
    dataset = load_dataset(
        config.DATASET_NAME,
        streaming=True, # Keep streaming for now, can change if needed
        split="train", # Explicitly request train split
        # trust_remote_code=True # May be needed for some datasets
    )
    print("Dataset object created (streaming mode). Accessing 'train' split...")

    # --- Take Sample (Local Testing) ---
    if config.LOAD_LOCAL_TEST_SAMPLE:
        print(f"Taking first {config.LOCAL_TEST_SAMPLE_SIZE} samples from the stream...")

        start_time = time.time()
        # Use itertools.islice for streaming datasets
        test_samples = list(itertools.islice(dataset, config.LOCAL_TEST_SAMPLE_SIZE))
        load_time = time.time() - start_time

        if not test_samples:
            print(f"ERROR: Could not load any samples after {load_time:.2f} seconds.")
        else:
            print(f"Successfully loaded {len(test_samples)} samples in {load_time:.2f} seconds.")
            print("\n--- Sample 0 --- ")
            sample_0 = test_samples[0]
            print(f"Available keys: {sample_0.keys()}")

            # Access data using column names from config
            speaker_id = sample_0.get(config.DATASET_SPEAKER_COLUMN)
            text = sample_0.get(config.DATASET_TEXT_COLUMN, "")
            audio_info = sample_0.get(config.DATASET_AUDIO_COLUMN)

            print(f"Speaker: {speaker_id}")
            print(f"Text: {text[:100]}...")
            print(f"Language: {sample_0.get('language')}") # Check if language column still exists

            # Check audio info (usually decoded automatically by datasets)
            if isinstance(audio_info, dict):
                 print(f"Audio Keys: {audio_info.keys()}")
                 if 'sampling_rate' in audio_info:
                     print(f"Audio Sampling Rate: {audio_info['sampling_rate']}")
                 if 'array' in audio_info:
                     # With parquet, array might be large, just print shape
                     print(f"Audio Array Shape: {getattr(audio_info['array'], 'shape', 'N/A')}")
                 elif 'bytes' in audio_info:
                     print(f"Audio Bytes Length: {len(audio_info['bytes'])}")
            elif audio_info is not None:
                 print(f"Audio Info Type: {type(audio_info)}")
            else:
                 print("Audio information not found or is None.")

    else:
        print("LOAD_LOCAL_TEST_SAMPLE is False in config. Script finished without loading sample.")

except Exception as e:
    import traceback
    print(f"ERROR during data loading: {e}")
    traceback.print_exc()

print("\n--- Data Loading Test Script Finished ---") 