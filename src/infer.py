# src/infer.py

import torch
import librosa
import soundfile as sf
import argparse
import os
import logging

# Ensure project root is in path for imports
import sys
if '.' not in sys.path:
    sys.path.append('.')

# Import project components
import src.config as config
from src.model import NovaTTS
from src.text_processing import TextProcessor
from src.dataset import mimi # Import the globally loaded mimi model

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def preprocess_audio(audio_path, target_sr):
    """Loads and resamples audio to the target sample rate."""
    try:
        waveform, sr = librosa.load(audio_path, sr=None, mono=True)
        if sr != target_sr:
            logger.info(f"Resampling audio from {sr} Hz to {target_sr} Hz...")
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)
        # Convert to torch tensor and add batch dimension
        return torch.tensor(waveform).unsqueeze(0)
    except Exception as e:
        logger.error(f"Error loading or processing audio file {audio_path}: {e}")
        raise

def main(args):
    logger.info(f"Using device: {config.DEVICE}")

    # --- 1. Load Model --- #
    logger.info("Loading NovaTTS model structure...")
    model = NovaTTS(config).to(config.DEVICE)
    
    logger.info(f"Loading checkpoint from: {args.checkpoint_path}")
    try:
        checkpoint = torch.load(args.checkpoint_path, map_location=config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Checkpoint loaded successfully (Epoch {checkpoint.get('epoch', 'N/A')})")
    except FileNotFoundError:
        logger.error(f"Checkpoint file not found: {args.checkpoint_path}")
        return
    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}")
        return

    model.eval() # Set model to evaluation mode
    logger.info("Model set to evaluation mode.")

    # --- 2. Initialize Text Processor --- #
    logger.info("Initializing Text Processor...")
    text_processor = TextProcessor(config)
    logger.info("Text Processor initialized.")

    # --- 3. Prepare Inputs --- #
    logger.info("Preparing inputs...")
    # Load and preprocess reference audio
    try:
        ref_waveform = preprocess_audio(args.ref_audio_path, config.SPEAKER_ENCODER_SAMPLE_RATE)
        ref_waveform = ref_waveform.to(config.DEVICE)
        logger.info(f"Reference audio loaded and processed: {ref_waveform.shape}")
    except Exception as e:
        logger.error(f"Failed to prepare reference audio: {e}")
        return
        
    # Process input text
    try:
        # Use the process method directly
        input_ids_list, attn_mask_list = text_processor.process(args.text, phonemes=True)
        
        # Convert lists to tensors and add batch dimension
        text_ids = torch.tensor(input_ids_list).unsqueeze(0).to(config.DEVICE) 
        attn_mask = torch.tensor(attn_mask_list).unsqueeze(0).to(config.DEVICE) 
        logger.info(f"Text processed: IDs shape {text_ids.shape}, Mask shape {attn_mask.shape}")
    except Exception as e:
        logger.error(f"Failed to process text: {e}")
        return

    # --- 4. Generate Mimi Codes (Streaming) --- #
    logger.info("Generating Mimi code stream...")
    try:
        code_chunk_generator = model.generate_codes(
            text_input_ids=text_ids,
            text_attention_mask=attn_mask,
            ref_audio_waveform=ref_waveform,
            max_length=args.max_gen_len, # Use argument for max length
            chunk_size=args.chunk_size, # Pass chunk size
            temperature=args.temperature, # Pass generation params
            top_k=args.top_k
        )
        # We don't have the full shape upfront anymore
        logger.info("Code generator created.")
    except Exception as e:
        logger.error(f"Error during code generator initialization: {e}", exc_info=True)
        return

    # --- 5. Decode Codes to Audio (Corrected Streaming) --- #
    logger.info("Decoding code stream using Mimi streaming context...")
    if mimi is None:
        logger.error("Global Mimi model not loaded. Cannot decode.")
        return
        
    output_audio_chunks = []
    try:
        # Use the mimi streaming context manager
        with mimi.streaming(batch_size=1):
            logger.info("Mimi streaming context entered.")
            # Iterate through the code chunks yielded by our generator
            for i, code_chunk in enumerate(code_chunk_generator):
                logger.info(f"Processing code chunk {i+1}, shape: {code_chunk.shape}")
                
                # Prepare chunk for decoding (CPU, long)
                codes_to_decode = code_chunk.cpu().long()
                # Note: Clamping is removed, assuming model behaves
                
                # Decode this specific chunk
                audio_chunk = mimi.decode(codes_to_decode)
                logger.info(f"Decoded audio chunk {i+1}, shape: {audio_chunk.shape}")
                
                # Append the resulting audio chunk (move to CPU if not already)
                if audio_chunk.numel() > 0:
                    output_audio_chunks.append(audio_chunk.cpu())
                else:
                    logger.warning(f"Received empty audio chunk {i+1}.")
                    
        # --- Concatenate Audio Chunks --- # 
        if not output_audio_chunks:
            logger.error("No audio chunks were generated after decoding.")
            output_waveform = torch.tensor([]) # Empty tensor
        else:
            output_waveform = torch.cat(output_audio_chunks, dim=-1) # Concatenate along time axis
            # Ensure waveform is 1D 
            if output_waveform.ndim > 1:
                 output_waveform = output_waveform.squeeze()
            logger.info(f"Streaming decoding complete. Final waveform shape: {output_waveform.shape}")

    except Exception as e:
        logger.error(f"Error decoding Mimi code stream: {e}", exc_info=True)
        return

    # --- 6. Save Output Audio --- #
    try:
        output_dir = os.path.dirname(args.output_path)
        if output_dir:
             os.makedirs(output_dir, exist_ok=True)
        
        # Detach tensor from graph before converting to numpy
        sf.write(args.output_path, output_waveform.detach().numpy(), config.MIMI_SAMPLE_RATE)
        logger.info(f"Generated audio saved to: {args.output_path}")
    except Exception as e:
        logger.error(f"Error saving output audio: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate audio using a trained NovaTTS model.")
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize.")
    parser.add_argument("--ref_audio_path", type=str, required=True, help="Path to reference audio file for speaker voice.")
    parser.add_argument("--checkpoint_path", type=str, default="models/novatts_best.pth", help="Path to the model checkpoint.")
    parser.add_argument("--output_path", type=str, default="output/generated_audio.wav", help="Path to save the generated audio.")
    parser.add_argument("--max_gen_len", type=int, default=300, help="Maximum number of code frames to generate.")
    parser.add_argument("--chunk_size", type=int, default=40, help="Number of code frames per generation chunk (for streaming).")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature for generation.")
    parser.add_argument("--top_k", type=int, default=None, help="Top-K sampling parameter for generation.")
    
    args = parser.parse_args()
    main(args) 