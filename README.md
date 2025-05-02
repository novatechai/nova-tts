# Custom Mimi-Based TTS System (NovaTTS)

## Overview

This project implements a custom text-to-speech (TTS) system designed for voice cloning using the Kyutai Mimi audio codec. Inspired by architectures like SesameAI's CSM, it takes text and a short reference audio clip as input and generates speech in the reference speaker's voice.

The core idea is to train a custom Transformer-based model (`NovaTTS`) to generate intermediate Mimi audio codes conditioned on both text content and speaker characteristics. These codes are then decoded into an audio waveform using the pre-trained Mimi vocoder provided by the `moshi` library.

## Features

*   **Voice Cloning:** Generates speech in the voice of a speaker from a short reference audio clip.
*   **Text-to-Speech:** Converts input text into audible speech.
*   **Custom Model:** Trains a dedicated `TransformerDecoder` (`MimiCodeGenerator`) to predict Mimi codes.
*   **Pre-trained Components:** Leverages pre-trained models for text representation (`Qwen/Qwen3-0.6B`), speaker encoding (`speechbrain/spkrec-ecapa-voxceleb`), and audio decoding (`moshi`).
*   **Streaming Architecture:** The code generation component (`model.generate_codes`) is designed to yield code chunks, enabling potential low-latency inference (though the current `infer.py` script decodes the full sequence due to library constraints).
*   **Experiment Tracking:** Integrated with Weights & Biases (`wandb`) for monitoring training progress.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd mimi-tts 
    ```
2.  **Create Environment & Install Dependencies:** This project uses `uv` for environment management.
    ```bash
    # Create a virtual environment (e.g., using Python's venv)
    python -m venv .venv
    
    # Activate the environment
    source .venv/bin/activate 
    
    # Install dependencies using uv
    uv pip install -r requirements.txt
    # Or using pip:
    # pip install -r requirements.txt
    ```
3.  **Install Espeak-NG:** The `phonemizer` library relies on `espeak-ng` as its backend.
    *   **macOS:** `brew install espeak-ng`
    *   **Ubuntu/Debian:** `sudo apt update && sudo apt install espeak-ng`
    *   **Other:** Follow instructions for your OS.
        *Note: The code attempts to find the library path automatically on macOS, adjust `src/text_processing.py` if needed for other systems.*

4.  **(Optional) Weights & Biases Setup:**
    *   Sign up for a free account at [wandb.ai](https://wandb.ai/).
    *   Log in via your terminal:
        ```bash
        wandb login
        ```
        Follow the prompts (you may need to paste an API key).

## Usage

### Training

1.  **Configuration:** Review and adjust parameters in `src/config.py` (e.g., `DATASET_NAME`, `EPOCHS`, `BATCH_SIZE`, `LEARNING_RATE`).
2.  **Prepare Data:** Ensure the dataset specified in `config.py` is accessible (e.g., on Hugging Face Hub).
3.  **Delete Old Checkpoints (Important on Retraining):** If restarting training, remove previous checkpoints:
    ```bash
    rm models/*.pth
    ```
4.  **Run Training:** Activate the environment and run the training script:
    ```bash
    source .venv/bin/activate
    python src/train.py
    ```
    *   Training progress (loss) will be printed to the console.
    *   If `wandb` is enabled, metrics will be logged to your WandB dashboard (default project: `mimi-tts-nova`).
    *   The script saves the best model checkpoint based on validation loss (if available) to `models/novatts_best.pth`.
    *   If no validation set is found (or created from the train set), checkpoints are saved after every epoch (`models/novatts_epoch_N.pth`).

### Inference

1.  **Ensure Checkpoint Exists:** You need a trained model checkpoint (e.g., `models/novatts_best.pth`).
2.  **Run Inference:** Activate the environment and run the inference script:
    ```bash
    source .venv/bin/activate
    python src/infer.py --text "Your desired text to synthesize." \
                        --ref_audio_path path/to/reference_audio.wav \
                        --checkpoint_path models/novatts_best.pth \
                        --output_path output/generated_speech.wav \
                        # Optional generation parameters:
                        # --temperature 0.8 \
                        # --top_k 50 \
                        # --max_gen_len 1500
    ```
    *   `--text`: The input text.
    *   `--ref_audio_path`: Path to a short audio file (.wav recommended) of the target speaker.
    *   `--checkpoint_path`: Path to the trained model checkpoint.
    *   `--output_path`: Path where the generated .wav file will be saved.
    *   `--temperature`, `--top_k`, `--max_gen_len`: Control the sampling process during code generation.
    *   `--chunk_size`: (Currently affects internal generation but not end-to-end latency due to decode limitations).

    *Note: For models in early training stages, the script applies necessary clamping to generated codes to prevent errors. This may result in glitchy audio until the model is better trained.*

## Code Structure

*   `src/`
    *   `config.py`: Central configuration for hyperparameters, paths, etc.
    *   `model.py`: Defines the `NovaTTS` model, including `MimiCodeGenerator`.
    *   `dataset.py`: Handles dataset loading, preprocessing, collation, and dataloader creation.
    *   `text_processing.py`: Contains `TextProcessor` for normalization, phonemization, and tokenization.
    *   `train.py`: Script for training the `NovaTTS` model.
    *   `infer.py`: Script for running inference with a trained model.
*   `memory-bank/`: Documentation tracking project context, progress, decisions.
*   `models/`: Default directory for saving trained model checkpoints.
*   `output/`: Default directory for saving generated audio files.
*   `scripts/`: Utility scripts (e.g., reference audio samples).
*   `requirements.txt`: Python dependencies.
*   `README.md`: This file.

## Key Dependencies

*   PyTorch
*   Transformers (Hugging Face)
*   Datasets (Hugging Face)
*   SpeechBrain
*   Moshi (Kyutai)
*   Librosa
*   Phonemizer (with espeak-ng backend)
*   WandB (Weights & Biases)
*   uv (for environment/package management)

## Known Issues & Future Work

*   **Inference Clamping:** Requires clamping generated codes during inference until the model is sufficiently trained.
*   **Phonemizer Warning:** A benign "words count mismatch" warning from `phonemizer` occurs but is filtered from logs.
*   **True Streaming Inference:** The `moshi` decoder currently seems to require the full code sequence, preventing true low-latency streaming in `infer.py`. Further investigation into `moshi`'s API or alternative decoding might be needed.
*   **Expressive Speech:** Generating non-speech sounds (giggles, sighs) based on text tags requires appropriately tagged training data, which is currently unavailable.
*   **Model Evaluation:** More rigorous evaluation metrics (e.g., WER, MOS, FAD) could be added.
