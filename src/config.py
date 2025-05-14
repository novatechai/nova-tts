# Configuration settings for the project

import torch
import os

# --- Hugging Face --- 
# WARNING: Hardcoding tokens is insecure. Use environment variables or secrets management.

HUGGING_FACE_TOKEN = os.getenv("HF_TOKEN")
# --- Hardware ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Model Identifiers ---
# Text Encoder (Using Qwen3-4B)
# TEXT_ENCODER_MODEL = "Qwen/Qwen3-4B"  # ~8GB, might be slow locally
TEXT_ENCODER_MODEL = "Qwen/Qwen3-0.6B" # Smaller Qwen model for local dev

# Speaker Encoder (Using SpeechBrain ECAPA-TDNN)
SPEAKER_ENCODER_MODEL = "speechbrain/spkrec-ecapa-voxceleb"

# Mimi (Accessed via moshi library, using default weights)
MIMI_REPO = "kyutai/mimi"
MIMI_WEIGHTS_FILENAME = "model.safetensors" # Default name used by moshi loader - Changed based on HF repo

# --- Audio Processing ---
# Determined by Mimi model
MIMI_SAMPLE_RATE = 24000
# Speaker Encoder might expect 16kHz, needs checking during implementation
SPEAKER_ENCODER_SAMPLE_RATE = 16000 # Typical for SpeechBrain models, verify

# Parameters for reference audio preprocessing (inference)
PREPROCESS_REF_AUDIO_NORMALIZE = True
PREPROCESS_REF_AUDIO_REMOVE_SILENCE = True
# Silence removal parameters (adjust as needed)
REF_SILENCE_THRESHOLD = 0.015 # Energy threshold
REF_MIN_SILENCE_DURATION = 0.15 # Seconds

# Parameters for training audio preprocessing
PREPROCESS_TRAIN_AUDIO_NORMALIZE = True
# We decided NOT to remove silence from training data

# --- Dataset ---
# DATASET_NAME = "MrDragonFox/EN_Emilia_Yodas_616h"
DATASET_NAME = "Jinsaryko/Elise"
# DATASET_SUBSET = "Emilia-YODAS" # No longer needed, dataset IS the subset
# DATASET_LANGUAGE = "en" # No longer needed, dataset IS English
DATASET_TEXT_COLUMN = "text" # Column name in Elise dataset
DATASET_SPEAKER_COLUMN = "speaker_name" # Elise dataset doesn't have this column
DATASET_AUDIO_COLUMN = "audio"
DATASET_PHONEMES_COLUMN = "phonemes" # Column name for pre-computed phonemes in Elise dataset

# For local testing
LOAD_LOCAL_TEST_SAMPLE = False # Set to False on H100
LOCAL_TEST_SAMPLE_SIZE = 5 # Number of rows to load for local testing
VALIDATION_SPLIT_PERCENTAGE = 0.1 # Percentage of train data to use for validation if no val split exists

# --- Training Hyperparameters (Placeholders) ---
# BATCH_SIZE = 4 # New batch size - REMOVED DUPLICATE HERE
LEARNING_RATE = 1e-4
NUM_EPOCHS = 30 # Adjust as needed

# Training Configuration
FREEZE_TEXT_ENCODER = True # Start with frozen text encoder
# TRAIN_BATCH_SIZE = 16 # Using unified BATCH_SIZE now
# VALID_BATCH_SIZE = 16 # Using unified BATCH_SIZE now
BATCH_SIZE = 4 # KEEP THIS ONE
EPOCHS = 2 # Keep low for testing

# --- Output Directories ---
OUTPUT_DIR = "outputs"
MODEL_SAVE_DIR = "models"
PROCESSED_DATA_DIR = "data/processed"

# Set the HF token environment variable (optional but good practice)
if HUGGING_FACE_TOKEN:
    os.environ["HF_TOKEN"] = HUGGING_FACE_TOKEN

print(f"Config loaded. Using device: {DEVICE}")

# Mimi Configuration
MIMI_VOCAB_SIZE = 2049 # Typically 2048 codes + 1 for padding/special?
MIMI_NUM_CODEBOOKS = 8 # Standard for Mimi

# Mimi Code Generator (Transformer Decoder) Configuration
GENERATOR_D_MODEL = 512      # Internal dimension of the Transformer
GENERATOR_NHEAD = 8          # Number of attention heads
GENERATOR_NUM_LAYERS = 6   # Number of decoder layers
GENERATOR_DIM_FF = 2048    # Dimension of feedforward network
GENERATOR_DROPOUT = 0.1      # Dropout rate

# --- Data Configuration ---
# DATASET_NAME = "MrDragonFox/EN_Emilia_Yodas_616h" 

# Added MIMI_EOS_TOKEN_ID = 0 to the configuration
MIMI_EOS_TOKEN_ID = 0 