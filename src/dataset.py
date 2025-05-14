# src/dataset.py

import torch
from torch.utils.data import Dataset, DataLoader
import datasets
import librosa
# import soundfile as sf
import numpy as np
import os
import sys
from moshi.models import loaders as moshiloaders # Use moshi again
from huggingface_hub import hf_hub_download # Use downloader from verify_mimi
# from transformers import MimiModel, AutoFeatureExtractor # Remove transformers import
from torch.nn.utils.rnn import pad_sequence

# Ensure project root is in path for imports
if '.' not in sys.path:
    sys.path.append('.')

import src.config as config
from src.text_processing import TextProcessor

print("Loading Mimi for dataset processing using moshi...")
# --- Load Mimi using method from verify_mimi.py --- #
try:
    # Use constants from moshi.models.loaders
    mimi_repo = moshiloaders.DEFAULT_REPO
    mimi_filename = moshiloaders.MIMI_NAME # Should be 'mimi.pth'
    print(f"Attempting to load/download {mimi_filename} from {mimi_repo}...")
    mimi_weights_path = hf_hub_download(mimi_repo, mimi_filename)
    print(f"Mimi weights path: {mimi_weights_path}")

    mimi_device = "cuda:0" # Preprocessing on GPU
    mimi = moshiloaders.get_mimi(filename=mimi_weights_path, device=mimi_device)
    # Update config sample rate based on loaded model
    config.MIMI_SAMPLE_RATE = mimi.sample_rate
    print(f"Mimi loaded via moshi. Sample rate: {config.MIMI_SAMPLE_RATE}")
except Exception as e:
    print(f"ERROR loading Mimi via moshi/hf_hub_download: {e}")
    raise RuntimeError(f"Failed to load Mimi model: {e}")
# -------------------------------------------------- #

class MimiTTSDataset(Dataset):
    """Custom PyTorch Dataset for preparing Mimi TTS training data."""
    def __init__(self, split="train", config=config):
        super().__init__()
        self.config = config
        self.split = split
        self.text_processor = TextProcessor(config)
        # Keep reference to the global mimi model for encoding
        self.mimi_model = mimi 

        print(f"Loading dataset '{self.config.DATASET_NAME}'...") # Reverted print
        # Load the dataset (potentially streaming if very large) - REVERTED TO NON-STREAMING
        
        effective_split = self.split
        if self.config.LOAD_LOCAL_TEST_SAMPLE:
            sample_size = self.config.LOCAL_TEST_SAMPLE_SIZE
            # effective_split = f"{self.split}[:{sample_size}]" # Slice notation doesn't work well with load_dataset
            print(f"Attempting to load only first {sample_size} samples (will slice AFTER load)")
        else:
             print(f"Loading full split: '{self.split}'")

        try:
            # Load the specified split (non-streaming)
            hf_dataset = datasets.load_dataset(
                 self.config.DATASET_NAME, 
                 split=self.split, 
                 trust_remote_code=True, 
                 # streaming=False # Default
            )

            # Apply slicing *after* loading the split, if requested
            if self.config.LOAD_LOCAL_TEST_SAMPLE:
                 sample_size = min(self.config.LOCAL_TEST_SAMPLE_SIZE, len(hf_dataset))
                 print(f"Slicing dataset '{self.split}' to {sample_size} samples using .select()")
                 self.dataset = hf_dataset.select(range(sample_size))
            else:
                 print(f"Using full loaded split: '{self.split}'")
                 self.dataset = hf_dataset # Use the full loaded split

        except Exception as e:
            # Try fallback without split specification (for datasets without predefined splits)
            print(f"ERROR loading dataset with split '{self.split}': {e}. Trying without split...")
            try:
                hf_dataset = datasets.load_dataset(
                     self.config.DATASET_NAME, 
                     trust_remote_code=True,
                     # streaming=False # Default
                )
                # If it loaded a dict, try selecting the train split
                if isinstance(hf_dataset, datasets.dataset_dict.DatasetDict) and 'train' in hf_dataset:
                     print("Selecting 'train' split from loaded DatasetDict.")
                     hf_dataset = hf_dataset['train']
                elif isinstance(hf_dataset, datasets.dataset_dict.DatasetDict):
                     first_split = next(iter(hf_dataset))
                     print(f"Warning: 'train' split not found. Using first available split: '{first_split}'")
                     hf_dataset = hf_dataset[first_split]
                
                # Apply slicing after fallback load
                if self.config.LOAD_LOCAL_TEST_SAMPLE:
                     sample_size = min(self.config.LOCAL_TEST_SAMPLE_SIZE, len(hf_dataset))
                     print(f"Slicing dataset (after fallback) to {sample_size} samples using .select()")
                     self.dataset = hf_dataset.select(range(sample_size))
                else:
                     self.dataset = hf_dataset
                     
            except Exception as e2:
                print(f"FATAL: Could not load dataset split '{self.split}' even after fallback: {e2}")
                raise # Re-raise the exception

        print(f"Dataset loaded. Number of samples: {len(self.dataset)}")

    def __len__(self):
        # Restore original __len__
        return len(self.dataset)

    def __getitem__(self, idx):
        # Restore original __getitem__ logic
        item = self.dataset[idx]
        
        # Apply the same processing as before
        # 1. Process Text
        text = item[self.config.DATASET_TEXT_COLUMN]
        input_ids, attention_mask = self.text_processor.process(text, phonemes=True)

        # 2. Load and Process Audio
        audio_data = item[self.config.DATASET_AUDIO_COLUMN]
        waveform = audio_data['array'].astype(np.float32)
        original_sr = audio_data['sampling_rate']

        # Ensure waveform is mono
        if waveform.ndim > 1:
            if waveform.shape[1] == 1: # Already mono but shape (T, 1)
                waveform = waveform.squeeze(1)
            else: # Convert stereo to mono
                waveform = librosa.to_mono(waveform.T)

        # 3. Resample Audio for Speaker Encoder
        if original_sr != self.config.SPEAKER_ENCODER_SAMPLE_RATE:
            ref_waveform = librosa.resample(
                waveform, orig_sr=original_sr, target_sr=self.config.SPEAKER_ENCODER_SAMPLE_RATE
            )
        else:
            ref_waveform = waveform

        # 4. Resample Audio for Mimi and Generate Target Codes
        if original_sr != self.config.MIMI_SAMPLE_RATE:
            mimi_waveform = librosa.resample(
                waveform, orig_sr=original_sr, target_sr=self.config.MIMI_SAMPLE_RATE
            )
        else:
            mimi_waveform = waveform # Corrected variable name

        # Use moshi mimi model for encoding
        mimi_waveform_tensor = torch.tensor(mimi_waveform).to(mimi_device)
        with torch.no_grad():
            if mimi_waveform_tensor.ndim > 1:
                mimi_waveform_tensor = mimi_waveform_tensor.squeeze()
            
            # Add channel and batch dim for moshi encode [B=1, C=1, T]
            mimi_input = mimi_waveform_tensor.unsqueeze(0).unsqueeze(0) 
            mimi_codes = self.mimi_model.encode(mimi_input) # Shape [B=1, N, T_codes]

        # Remove the batch dimension added by encode
        if mimi_codes.shape[0] == 1:
            mimi_codes = mimi_codes.squeeze(0) # Shape [N, T_codes]
        else:
            print(f"Warning: Expected batch size 1 from Mimi encode, got {mimi_codes.shape[0]}")

        # --- Append EOS Token --- #
        # Create an EOS frame: [N, 1] filled with the EOS token ID
        eos_frame = torch.full((mimi_codes.shape[0], 1), 
                               self.config.MIMI_EOS_TOKEN_ID, 
                               dtype=torch.long, 
                               device=mimi_codes.device)
        # Concatenate along the time dimension (dim=1)
        mimi_codes_with_eos = torch.cat([mimi_codes, eos_frame], dim=1)
        # --- End Append EOS --- #

        # Convert numpy arrays / lists from text processing to tensors
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        ref_waveform = torch.tensor(ref_waveform, dtype=torch.float32)

        return {
            "text_ids": input_ids,
            "attention_mask": attention_mask,
            "ref_audio_waveform": ref_waveform,
            "target_mimi_codes": mimi_codes_with_eos.long()
        }

    # Remove __iter__ or comment out if preferred
    # def __iter__(self):
    #     ...

def collate_batch(batch):
    """Pads sequences within a batch and returns collated tensors and padding masks."""
    # Determine padding values
    text_pad_id = config.TEXT_PAD_TOKEN_ID if hasattr(config, 'TEXT_PAD_TOKEN_ID') else 0 
    mimi_code_pad_id = config.MIMI_PAD_TOKEN_ID if hasattr(config, 'MIMI_PAD_TOKEN_ID') else 0 
    audio_pad_value = 0.0
    
    # Separate items
    text_ids = [item['text_ids'] for item in batch]
    attn_masks = [item['attention_mask'] for item in batch]
    ref_waveforms = [item['ref_audio_waveform'] for item in batch]
    target_codes = [item['target_mimi_codes'] for item in batch] # Shape [N, T_codes]

    # --- Pad sequences --- 
    text_ids_padded = pad_sequence(text_ids, batch_first=True, padding_value=text_pad_id)
    attn_masks_padded = pad_sequence(attn_masks, batch_first=True, padding_value=0)
    ref_waveforms_padded = pad_sequence(ref_waveforms, batch_first=True, padding_value=audio_pad_value)
    
    # Pad target codes (transposed approach)
    target_codes_transposed = [item['target_mimi_codes'].transpose(0, 1) for item in batch] # List of [T, N]
    target_codes_padded_transposed = pad_sequence(target_codes_transposed, batch_first=True, padding_value=mimi_code_pad_id)
    target_codes_padded = target_codes_padded_transposed.transpose(1, 2) # [B, N, T_padded]

    # --- Create Padding Masks --- 
    # True where value is NOT the pad_id, False where it IS the pad_id
    text_padding_mask = (text_ids_padded != text_pad_id)
    
    # Target mask needs to be based on the time dimension (T_padded)
    # Check if *any* codebook at a given time step is padding. Since we pad all N codebooks
    # simultaneously with the same value, we can just check the first codebook channel.
    # Shape [B, T_padded]
    target_padding_mask = (target_codes_padded[:, 0, :] != mimi_code_pad_id)
    
    # NOTE: The original 'attention_mask' from the text processor is ignored here.
    # We derive the text mask purely from the padding added during collation.
    # This assumes the text processor does not add its own special padding tokens that
    # should be masked differently than the collation padding.

    return {
        "text_ids": text_ids_padded,          # [B, T_text_padded]
        "attention_mask": attn_masks_padded,    # [B, T_text_padded]
        "text_padding_mask": text_padding_mask, # [B, T_text_padded]
        "ref_audio_waveform": ref_waveforms_padded, # [B, T_audio_padded]
        "target_mimi_codes": target_codes_padded, # [B, N, T_code_padded]
        "target_padding_mask": target_padding_mask # [B, T_code_padded]
    }

def create_dataloaders(config):
    """Creates train and validation dataloaders."""
    
    train_dataset = MimiTTSDataset(split="train", config=config)
    print(f"Train dataset size: {len(train_dataset)}")

    # --- Validation Dataset --- 
    val_dataset = None
    val_loader = None
    potential_val_splits = ["validation", "test"]
    loaded_val_split = None

    for split_name in potential_val_splits:
        try:
            print(f"Attempting to load validation split: '{split_name}'")
            val_dataset = MimiTTSDataset(split=split_name, config=config)
            loaded_val_split = split_name
            print(f"Successfully loaded validation split: '{loaded_val_split}'")
            break
        except Exception as e:
            print(f"Could not load split '{split_name}': {e}. Trying next potential split.")
            val_dataset = None
            
    # --- If no val split loaded, try splitting train set --- #
    if val_dataset is None:
        print("No dedicated validation split found. Attempting to split training set.")
        if len(train_dataset) < 20: # Don't split very small datasets
             print("Training dataset too small to split for validation.")
        else:
             try:
                 # Use train_test_split for a 95/5 split (test_size=0.05)
                 # Use shuffle=False to take the *last* 5% deterministically
                 split_result = train_dataset.dataset.train_test_split(test_size=0.05, shuffle=False, seed=config.SEED)
                 # IMPORTANT: We need to re-wrap these splits in our MimiTTSDataset class
                 # This assumes train_dataset has access to the underlying HF dataset object via `.dataset`
                 train_split_data = split_result['train']
                 val_split_data = split_result['test']
                 
                 # Re-create the datasets using the split data
                 # Need to pass the underlying dataset object to the constructor
                 # Let's modify MimiTTSDataset to optionally accept a dataset object
                 print(f"Splitting train set: {len(train_split_data)} train / {len(val_split_data)} validation samples.")
                 train_dataset.dataset = train_split_data # Update the dataset object within the existing train_dataset instance
                 val_dataset = MimiTTSDataset(split="validation_from_train", config=config) # Create new instance for validation
                 val_dataset.dataset = val_split_data # Assign the split data to the new instance
                 loaded_val_split = "train_split(5%)"
                 print(f"Successfully created validation set from training data.")
                 # Update train dataset size print
                 print(f"New train dataset size: {len(train_dataset)}") 

             except AttributeError:
                 print("Warning: Dataset object does not have `.train_test_split()` method. Cannot automatically split train set.")
             except Exception as e_split:
                 print(f"Error attempting to split training set: {e_split}. Proceeding without validation set.")
                 val_dataset = None # Ensure it remains None on error

    # --- Collate Function --- 
    collate_fn = collate_batch 

    # --- Create DataLoaders --- 
    effective_batch_size = config.BATCH_SIZE 
    dataloader_shuffle = True # Shuffle training data
    num_workers = config.NUM_DATALOADER_WORKERS if hasattr(config, 'NUM_DATALOADER_WORKERS') else 0
    print(f"Using effective batch size: {effective_batch_size}, Shuffle: {dataloader_shuffle}, Workers: {num_workers}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=effective_batch_size,
        shuffle=dataloader_shuffle, 
        collate_fn=collate_fn, # Use custom collate_fn
        num_workers=num_workers
    )
    print(f"Train DataLoader created. Num batches: {len(train_loader)}")

    if val_dataset:
        print(f"Validation dataset size: {len(val_dataset)}")
        # Use same settings for val loader, but no shuffling
        val_loader = DataLoader(
            val_dataset,
            batch_size=effective_batch_size, 
            shuffle=False, 
            collate_fn=collate_fn, # Use custom collate_fn
            num_workers=num_workers
        )
        print(f"Validation DataLoader created. Num batches: {len(val_loader)}")
    else:
        print("Warning: No validation split found or loaded successfully. Skipping validation loader creation.")
        # Keep val_loader as None

    return train_loader, val_loader


# --- Basic Test --- #
if __name__ == "__main__":
    print("Testing MimiTTSDataset and DataLoader creation...")
    # Ensure local testing is enabled in config for this test
    if not config.LOAD_LOCAL_TEST_SAMPLE:
         print("WARNING: config.LOAD_LOCAL_TEST_SAMPLE is False. Set to True to run this test.")
         # Temporarily override for testing
         # config.LOAD_LOCAL_TEST_SAMPLE = True 
         # config.LOCAL_TEST_SAMPLE_SIZE = 2
         # raise SystemExit("Set config.LOAD_LOCAL_TEST_SAMPLE=True to test.")
    
    try:
        train_loader, _ = create_dataloaders(config)
        print("DataLoader created successfully.")

        print("\nFetching one batch...")
        for i, batch in enumerate(train_loader):
            print(f"Batch {i+1} loaded.")
            print("Keys:", batch.keys())
            print("Shapes:")
            for key, value in batch.items():
                # Add check for None type before accessing shape
                if value is not None:
                    print(f"  {key}: {value.shape}")
                else:
                    print(f"  {key}: None")
            
            # Check dtypes
            print("Dtypes:")
            for key, value in batch.items():
                 if value is not None:
                     print(f"  {key}: {value.dtype}")
                 else:
                     print(f"  {key}: None")

            if i == 0: # Only check the first batch
                break 
        
        print("\nDataset and DataLoader test finished.")

    except Exception as e:
        print(f"\nERROR during dataset/dataloader test: {e}")
        import traceback
        traceback.print_exc() 