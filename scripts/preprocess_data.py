# scripts/preprocess_data.py

import torch
import torchaudio
import torchaudio.functional as F
import itertools
from datasets import load_dataset, Dataset, Features, Value, Sequence, Array2D
from huggingface_hub import hf_hub_download
from moshi.models import loaders
import time
import os
import warnings

# Assuming config.py and text_processing.py are in the src directory
import sys
sys.path.append('.') # To import config from the root directory run
import src.config as config
from src.text_processing import process_text # Import our text processing function

# Suppress specific warnings if needed (e.g., from datasets/arrow)
warnings.simplefilter(action='ignore', category=FutureWarning)

print("--- Starting Data Preprocessing Script ---")

# --- Configuration & Setup ---
DEVICE = config.DEVICE
SAMPLE_RATE = config.MIMI_SAMPLE_RATE
# Use a limit for local testing, set to None for full run on H100
PROCESSING_LIMIT = config.LOCAL_TEST_SAMPLE_SIZE if config.LOAD_LOCAL_TEST_SAMPLE else None 
OUTPUT_DATASET_PATH = config.PROCESSED_DATA_DIR

print(f"Using device: {DEVICE}")
print(f"Target sample rate: {SAMPLE_RATE} Hz")
print(f"Processing limit: {PROCESSING_LIMIT if PROCESSING_LIMIT else 'None (Full Dataset)'}")
print(f"Output path: {OUTPUT_DATASET_PATH}")

os.makedirs(OUTPUT_DATASET_PATH, exist_ok=True)

# --- 1. Load Mimi Encoder Model ---
mimi = None
try:
    print("Loading Mimi model for encoding...")
    mimi_weight_path = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
    # Load on CPU first if memory is constrained, then move if needed
    # Or directly to DEVICE if confident
    mimi = loaders.get_mimi(mimi_weight_path, device=DEVICE)
    # Ensure sample rate matches config
    if mimi.sample_rate != SAMPLE_RATE:
        print(f"Warning: Mimi model sample rate ({mimi.sample_rate}) differs from config ({SAMPLE_RATE}). Using model's rate.")
        SAMPLE_RATE = mimi.sample_rate
    print("Mimi model loaded successfully.")
except Exception as e:
    print(f"ERROR loading Mimi model: {e}")
    exit()

# --- 2. Audio Normalization Function ---
def normalize_audio(waveform):
    """Peak normalize audio waveform to [-1.0, 1.0]."""
    if config.PREPROCESS_TRAIN_AUDIO_NORMALIZE:
        max_abs = torch.max(torch.abs(waveform))
        if max_abs > 1e-6: # Avoid division by zero
            return waveform / max_abs
    return waveform

# --- 3. Preprocessing Function for a Single Example ---
def preprocess_example(example):
    """Processes a single example from the dataset."""
    try:
        # Extract data using column names from config
        speaker_id = example.get(config.DATASET_SPEAKER_COLUMN)
        text = example.get(config.DATASET_TEXT_COLUMN)
        audio_info = example.get(config.DATASET_AUDIO_COLUMN)

        if not all([speaker_id, text, audio_info]):
            print(f"Warning: Missing required data in example: {example.keys()}. Skipping.")
            return None

        # --- Text Processing ---
        # Uses the pipeline: normalize -> phonemize -> tokenize
        token_ids = process_text(text)
        if token_ids is None:
            print(f"Warning: Text processing failed for text: '{text[:50]}...'. Skipping.")
            return None

        # --- Audio Processing ---
        # Ensure audio is loaded as array
        if not isinstance(audio_info, dict) or 'array' not in audio_info or 'sampling_rate' not in audio_info:
            print(f"Warning: Audio data is not in expected format {{ 'array': ..., 'sampling_rate': ... }}. Got: {audio_info}. Skipping.")
            return None

        waveform = torch.tensor(audio_info['array']).float()
        sr = audio_info['sampling_rate']

        # Ensure mono
        if waveform.ndim > 1 and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0) # Average channels
        elif waveform.ndim == 2 and waveform.shape[0] == 1:
            waveform = waveform.squeeze(0) # Remove channel dim if already mono [1, T]
        
        # Ensure correct shape [T]
        if waveform.ndim != 1:
             print(f"Warning: Unexpected waveform dimension after mono conversion: {waveform.ndim}. Skipping.")
             return None

        # Resample
        if sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
            waveform = resampler(waveform)

        # Normalize
        waveform = normalize_audio(waveform)

        # Encode to Mimi codes
        # mimi.encode expects [B=1, C=1, T]
        waveform_batch = waveform.unsqueeze(0).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            mimi_codes = mimi.encode(waveform_batch)
            # Result shape: [B=1, K=num_codebooks, T_codes]
            # Squeeze batch dim and convert to list of lists for saving
            mimi_codes_list = mimi_codes.squeeze(0).cpu().tolist()

        # Return processed data
        return {
            "input_ids": token_ids, # Return as list
            "speaker_id": speaker_id,
            "mimi_codes": mimi_codes_list # Save as list of lists
        }

    except Exception as e:
        import traceback
        print(f"ERROR processing example: {e}")
        # print(f"Problematic example keys: {example.keys()}")
        traceback.print_exc()
        return None

# --- 4. Load, Process, and Save Dataset --- 
def process_and_save():
    print(f"\n--- Loading and Processing Dataset: {config.DATASET_NAME} ---")
    dataset = load_dataset(
        config.DATASET_NAME,
        streaming=True,
        split="train",
        # trust_remote_code=True # May be needed
    )
    print("Dataset stream opened.")

    processed_data = []
    processed_count = 0
    skipped_count = 0
    start_time = time.time()

    print("Starting processing loop...")
    for i, example in enumerate(dataset):
        if PROCESSING_LIMIT is not None and processed_count >= PROCESSING_LIMIT:
            print(f"Reached processing limit: {PROCESSING_LIMIT}. Stopping.")
            break

        processed_example = preprocess_example(example)

        if processed_example is not None:
            processed_data.append(processed_example)
            processed_count += 1
        else:
            skipped_count += 1

        if (i + 1) % 100 == 0:
             current_time = time.time()
             elapsed = current_time - start_time
             rate = (i + 1) / elapsed if elapsed > 0 else 0
             print(f"Processed {i+1} raw samples. Kept: {processed_count}, Skipped: {skipped_count}. Rate: {rate:.2f} samples/sec.")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nFinished processing.")
    print(f"Total raw samples iterated: {i+1 if 'i' in locals() else 0}")
    print(f"Successfully processed samples: {processed_count}")
    print(f"Skipped samples due to errors/missing data: {skipped_count}")
    print(f"Total processing time: {total_time:.2f} seconds")

    if not processed_data:
        print("No data was processed successfully. Check dataset or preprocessing logic.")
        return

    # --- 5. Convert to Hugging Face Dataset and Save ---
    print("\nConverting processed data to Hugging Face Dataset...")
    try:
        # Define features: Sequence of Sequences for mimi_codes
        features = Features({
            'input_ids': Sequence(feature=Value(dtype='int64')),
            'speaker_id': Value(dtype='string'),
            'mimi_codes': Sequence(feature=Sequence(feature=Value(dtype='int64')))
            # Using nested Sequence instead of Array2D
            # 'mimi_codes': Array2D(shape=(mimi.num_codebooks, None), dtype='int64') # This caused error
        })

        # Create dataset from list of dictionaries
        hf_dataset = Dataset.from_list(processed_data, features=features)
        print(f"Dataset created with {len(hf_dataset)} rows.")
        print("Features:", hf_dataset.features)

        print(f"Saving dataset to disk at: {OUTPUT_DATASET_PATH}")
        hf_dataset.save_to_disk(OUTPUT_DATASET_PATH)
        print("Dataset saved successfully.")

    except Exception as e:
        import traceback
        print(f"ERROR during dataset conversion or saving: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    # Run the processing and saving
    process_and_save()
    print("\n--- Data Preprocessing Script Finished --- ") 