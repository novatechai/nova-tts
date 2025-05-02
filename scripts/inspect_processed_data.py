from datasets import load_from_disk
import os
import sys
sys.path.append('.')
import src.config as config # To get the path easily

# Path where the dataset was saved
dataset_path = config.PROCESSED_DATA_DIR

print(f"Loading processed dataset from: {dataset_path}")

if not os.path.exists(dataset_path):
    print(f"ERROR: Dataset directory not found at {dataset_path}")
else:
    try:
        processed_dataset = load_from_disk(dataset_path)
        print("Dataset loaded successfully!")
        print(f"Number of samples: {len(processed_dataset)}")
        print(f"Features: {processed_dataset.features}")

        # Print the first sample
        if len(processed_dataset) > 0:
            print("\n--- First Sample ---")
            sample_0 = processed_dataset[0]
            print(f"Keys: {sample_0.keys()}")
            print(f"Speaker ID: {sample_0['speaker_id']}")
            # input_ids is a list
            print(f"Input IDs (len={len(sample_0['input_ids'])}): {sample_0['input_ids'][:20]}...")
            # mimi_codes is a list of lists
            num_codebooks = len(sample_0['mimi_codes'])
            num_frames = len(sample_0['mimi_codes'][0]) if num_codebooks > 0 else 0
            print(f"Mimi Codes Shape (Codebooks x Frames): ({num_codebooks} x {num_frames})")
            # Print first 5 codes from first codebook
            if num_codebooks > 0 and num_frames > 0:
                print(f"Mimi Codes (first book, first 5 codes): {sample_0['mimi_codes'][0][:5]}...")
        else:
            print("Loaded dataset is empty.")

    except Exception as e:
        print(f"ERROR loading dataset from disk: {e}") 