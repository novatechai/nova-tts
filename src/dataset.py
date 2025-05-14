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
    def __init__(self, config=config, split="train", hf_dataset_obj=None):
        super().__init__()
        self.config = config
        self.split = split # Retain for logging/context if needed, but hf_dataset_obj takes precedence
        self.text_processor = TextProcessor(config)
        # Keep reference to the global mimi model for encoding
        self.mimi_model = mimi 

        if hf_dataset_obj is not None:
            print(f"Initializing MimiTTSDataset with pre-loaded Hugging Face dataset object ({split} context). Samples: {len(hf_dataset_obj)}")
            self.dataset = hf_dataset_obj
        else:
            # Existing loading logic based on self.split and self.config.DATASET_NAME
            print(f"Loading dataset '{self.config.DATASET_NAME}' for split '{self.split}'...")
            effective_split = self.split
            if self.config.LOAD_LOCAL_TEST_SAMPLE:
                sample_size = self.config.LOCAL_TEST_SAMPLE_SIZE
                print(f"Attempting to load only first {sample_size} samples for split '{self.split}' (will slice AFTER load)")
            else:
                print(f"Loading full split: '{self.split}'")

            try:
                loaded_single_split = datasets.load_dataset(
                    self.config.DATASET_NAME, 
                    split=effective_split,
                    keep_in_memory=True
                )
                if self.config.LOAD_LOCAL_TEST_SAMPLE:
                    sample_size = min(self.config.LOCAL_TEST_SAMPLE_SIZE, len(loaded_single_split))
                    print(f"Slicing dataset '{effective_split}' to {sample_size} samples using .select()")
                    self.dataset = loaded_single_split.select(range(sample_size))
                else:
                    self.dataset = loaded_single_split

            except Exception as e:
                print(f"ERROR loading dataset with specific split '{effective_split}': {e}. This should ideally be handled by create_dataloaders now.")
                # Fallback logic in __init__ becomes less critical if create_dataloaders handles splits robustly.
                # For safety, one might keep a simplified fallback or raise an error if hf_dataset_obj is None and loading fails.
                raise RuntimeError(f"Failed to load dataset for split '{effective_split}' in MimiTTSDataset constructor and no hf_dataset_obj was provided: {e}")

            print(f"Dataset for split '{self.split}' loaded. Number of samples: {len(self.dataset)}")

    def __len__(self):
        # Restore original __len__
        return len(self.dataset)

    def __getitem__(self, idx):
        # Restore original __getitem__ logic
        item = self.dataset[idx]
        
        # Apply the same processing as before
        # 1. Process Text
        text_to_process = None
        is_phoneme = False

        # Check for pre-computed phonemes first
        if self.config.DATASET_PHONEMES_COLUMN and self.config.DATASET_PHONEMES_COLUMN in item:
            phonemes_data = item[self.config.DATASET_PHONEMES_COLUMN]
            if phonemes_data and isinstance(phonemes_data, str):
                # print(f"DBG: Using pre-computed phonemes: {phonemes_data[:100]}...")
                text_to_process = phonemes_data
                is_phoneme = True
            else:
                print(f"Warning: Column '{self.config.DATASET_PHONEMES_COLUMN}' found but is empty or not a string. Falling back to raw text.")
        
        # If no valid pre-computed phonemes, use raw text
        if text_to_process is None:
            if self.config.DATASET_TEXT_COLUMN in item:
                text_to_process = item[self.config.DATASET_TEXT_COLUMN]
                # print(f"DBG: Using raw text: {text_to_process[:100]}...")
                is_phoneme = False # Ensure this is false if we fall back to text
            else:
                raise ValueError(f"Neither '{self.config.DATASET_PHONEMES_COLUMN}' (valid) nor '{self.config.DATASET_TEXT_COLUMN}' found in dataset item: {item.keys()}")

        if text_to_process is None:
             raise ValueError("Text input for processor is None, this should not happen.")

        input_ids, attention_mask = self.text_processor.process(text_to_process, is_phoneme_input=is_phoneme)

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
    
    hf_train_dataset = None
    hf_val_dataset = None

    print(f"Loading Hugging Face dataset: {config.DATASET_NAME}")
    try:
        # Attempt to load all splits first
        loaded_data = datasets.load_dataset(config.DATASET_NAME, keep_in_memory=True)
        
        if isinstance(loaded_data, datasets.DatasetDict):
            if 'train' not in loaded_data:
                raise ValueError(f"Dataset {config.DATASET_NAME} loaded as a DatasetDict but does not contain a 'train' split.")
            hf_train_dataset = loaded_data['train']
            
            if 'validation' in loaded_data:
                print("Found 'validation' split in the dataset.")
                hf_val_dataset = loaded_data['validation']
            elif 'test' in loaded_data: # Use 'test' as validation if 'validation' is not present
                print("Found 'test' split in the dataset, using as validation.")
                hf_val_dataset = loaded_data['test']
            else:
                print("No 'validation' or 'test' split found. Creating validation set from 'train' split.")
                if len(hf_train_dataset) < 2: # Need at least 2 samples to split
                    raise ValueError("Training dataset is too small to create a validation split.")
                # Ensure test_size is less than 1.0 and greater than 0.0 if dataset is small
                val_split_percentage = config.VALIDATION_SPLIT_PERCENTAGE
                if len(hf_train_dataset) * val_split_percentage < 1:
                    val_split_percentage = 1 / len(hf_train_dataset) # ensure at least 1 sample for validation
                    print(f"Adjusted validation split percentage to {val_split_percentage:.4f} to get at least one sample.")
                
                split_datasets = hf_train_dataset.train_test_split(
                    test_size=val_split_percentage, 
                    shuffle=True, 
                    seed=42 # for reproducibility
                )
                hf_train_dataset = split_datasets['train']
                hf_val_dataset = split_datasets['test'] # train_test_split names the validation part 'test'
                print(f"Created validation set with {len(hf_val_dataset)} samples.")
        elif isinstance(loaded_data, datasets.Dataset):
            # Loaded data is a single split, assume it's the training set
            print("Loaded dataset as a single split. Assuming it is the training set and creating validation split.")
            hf_train_dataset = loaded_data
            if len(hf_train_dataset) < 2:
                raise ValueError("Dataset is too small to create a validation split.")
            val_split_percentage = config.VALIDATION_SPLIT_PERCENTAGE
            if len(hf_train_dataset) * val_split_percentage < 1:
                val_split_percentage = 1 / len(hf_train_dataset)
                print(f"Adjusted validation split percentage to {val_split_percentage:.4f} to get at least one sample.")

            split_datasets = hf_train_dataset.train_test_split(
                test_size=val_split_percentage, 
                shuffle=True, 
                seed=42
            )
            hf_train_dataset = split_datasets['train']
            hf_val_dataset = split_datasets['test']
            print(f"Created validation set with {len(hf_val_dataset)} samples.")
        else:
            raise TypeError(f"Loaded dataset {config.DATASET_NAME} is of unexpected type: {type(loaded_data)}")

    except Exception as e:
        print(f"Error loading or splitting dataset {config.DATASET_NAME}: {e}")
        raise

    # Apply local test sample slicing if configured, *after* train/val split
    if config.LOAD_LOCAL_TEST_SAMPLE:
        train_sample_size = min(config.LOCAL_TEST_SAMPLE_SIZE, len(hf_train_dataset))
        val_sample_size = min(config.LOCAL_TEST_SAMPLE_SIZE, len(hf_val_dataset)) # Apply to val set too
        
        print(f"Slicing training data to {train_sample_size} samples for local testing.")
        hf_train_dataset = hf_train_dataset.select(range(train_sample_size))
        if hf_val_dataset and len(hf_val_dataset) > 0:
            print(f"Slicing validation data to {val_sample_size} samples for local testing.")
            hf_val_dataset = hf_val_dataset.select(range(val_sample_size))
        elif not hf_val_dataset:
             print("Warning: No validation dataset to slice for local testing.")


    train_dataset = MimiTTSDataset(config=config, split="train", hf_dataset_obj=hf_train_dataset)
    print(f"Train dataset size: {len(train_dataset)}")

    val_dataset = None
    if hf_val_dataset and len(hf_val_dataset) > 0:
        val_dataset = MimiTTSDataset(config=config, split="validation", hf_dataset_obj=hf_val_dataset)
        print(f"Validation dataset size: {len(val_dataset)}")
    else:
        print("No validation dataset created or loaded.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        collate_fn=collate_batch,
        shuffle=True,
        num_workers=config.NUM_WORKERS if hasattr(config, 'NUM_WORKERS') else 0,
        pin_memory=config.PIN_MEMORY if hasattr(config, 'PIN_MEMORY') else False
    )

    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.BATCH_SIZE, # Can use same or different batch size
            collate_fn=collate_batch,
            shuffle=False, # No need to shuffle validation data
            num_workers=config.NUM_WORKERS if hasattr(config, 'NUM_WORKERS') else 0,
            pin_memory=config.PIN_MEMORY if hasattr(config, 'PIN_MEMORY') else False
        )
        print("Train and Validation DataLoaders created.")
    else:
        print("Train DataLoader created. No Validation DataLoader (validation set was empty or not created).")

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