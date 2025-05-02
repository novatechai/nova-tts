import torch
import torchaudio
from huggingface_hub import hf_hub_download
from moshi.models import loaders
import time
import os

print("--- Starting Mimi Verification Script ---.")

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_WAV_PATH = "scripts/urek.wav"
OUTPUT_DIR = "outputs"
OUTPUT_NON_STREAMING = os.path.join(OUTPUT_DIR, "urek_reconstructed_non_streaming.wav")
OUTPUT_STREAMING = os.path.join(OUTPUT_DIR, "urek_reconstructed_streaming.wav")

print(f"Using device: {DEVICE}")
print(f"Input WAV: {INPUT_WAV_PATH}")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 1. Load Mimi Model ---
try:
    print("Loading Mimi model using moshi.models.loaders.get_mimi...")
    mimi_weight_path = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(mimi_weight_path, device=DEVICE)
    NUM_CODEBOOKS = mimi.num_codebooks
    print(f"Successfully loaded Mimi model.")
    print(f"Default number of codebooks: {NUM_CODEBOOKS}")
except Exception as e:
    print(f"ERROR loading Mimi model: {e}")
    exit()

# --- 2. Get Sampling Rate ---
try:
    SAMPLE_RATE = mimi.sample_rate
    print(f"Mimi sampling rate: {SAMPLE_RATE} Hz")
except Exception as e:
    print(f"ERROR getting sampling rate: {e}")
    exit()

# --- 3. Load and Prepare Input Audio ---
try:
    print(f"\nLoading audio from: {INPUT_WAV_PATH}")
    input_waveform, input_sr = torchaudio.load(INPUT_WAV_PATH)
    input_waveform = input_waveform.to(DEVICE)

    # Resample if necessary
    if input_sr != SAMPLE_RATE:
        print(f"Resampling input audio from {input_sr} Hz to {SAMPLE_RATE} Hz...")
        resampler = torchaudio.transforms.Resample(orig_freq=input_sr, new_freq=SAMPLE_RATE).to(DEVICE)
        input_waveform = resampler(input_waveform)

    # Ensure mono (take first channel if stereo)
    if input_waveform.shape[0] > 1:
        print("Input audio is stereo, taking first channel.")
        input_waveform = input_waveform[0, :].unsqueeze(0)

    print(f"Prepared input waveform shape: {input_waveform.shape}") # Expect [1, T]

except Exception as e:
    print(f"ERROR loading or preparing input audio: {e}")
    exit()

# --- 4. Test Basic Encode/Decode ---
decoded_waveform = None # Initialize in case of error
try:
    print("\n--- Testing basic encode/decode ---")
    # Encode
    with torch.no_grad(): # Disable gradients for encode/decode
        # Expected input shape for mimi.encode: [B, 1, T]
        codes = mimi.encode(input_waveform.unsqueeze(1)) # Add channel dim
        print(f"Encoded codes shape: {codes.shape}") # Expected: [B, K, T_codes] B=1, K=num_codebooks

        # Decode (non-streaming)
        decoded_waveform = mimi.decode(codes)
        print(f"Decoded waveform shape (non-streaming): {decoded_waveform.shape}") # Expected: [B, 1, T_audio]

    # Save non-streaming output
    if decoded_waveform is not None and decoded_waveform.numel() > 0:
        # Ensure gradients are detached before saving
        torchaudio.save(OUTPUT_NON_STREAMING, decoded_waveform.detach().squeeze(1).cpu(), SAMPLE_RATE)
        print(f"Saved non-streaming reconstructed audio to: {OUTPUT_NON_STREAMING}")
    else:
        print("Skipping save for non-streaming audio (decode failed or empty).")

except Exception as e:
    print(f"ERROR during basic encode/decode test: {e}")

# --- 5. Test Streaming Decode ---
full_streamed_audio = None # Initialize
try:
    print("\n--- Testing streaming decode ---")
    num_frames = codes.shape[2]
    chunk_size = max(1, 2) # Use smaller chunks for streaming simulation
    decoded_stream_chunks = []

    print(f"Decoding {num_frames} code frames in chunks of ~{chunk_size}...")

    start_time = time.time()
    # Disable gradients for the whole streaming loop
    with torch.no_grad(), mimi.streaming(batch_size=1):
        for i in range(0, num_frames, chunk_size):
            chunk_start = i
            chunk_end = min(i + chunk_size, num_frames)
            # Ensure code_chunk doesn't require grad if codes does (it shouldn't after encode)
            code_chunk = codes[:, :, chunk_start:chunk_end].detach()

            if code_chunk.shape[2] == 0: continue

            audio_chunk = mimi.decode(code_chunk)
            if audio_chunk is not None and audio_chunk.numel() > 0:
                # Detach chunk immediately after decoding
                decoded_stream_chunks.append(audio_chunk.detach())
            else:
                print(f"Warning: decode() returned empty or None for chunk {chunk_start}-{chunk_end}")

    end_time = time.time()
    print(f"Streaming decode finished in {end_time - start_time:.3f} seconds.")

    if decoded_stream_chunks:
        processed_chunks = []
        for ch in decoded_stream_chunks:
            # Already detached, just move to CPU if needed for concat/save
            ch_cpu = ch.cpu()
            if ch_cpu.dim() == 2: # Should be [B, T] or [1, T]
                processed_chunks.append(ch_cpu)
            elif ch_cpu.dim() == 3 and ch_cpu.shape[1] == 1: # Should be [B, 1, T]
                processed_chunks.append(ch_cpu.squeeze(1))
            else:
                print(f"Warning: Unexpected chunk shape {ch_cpu.shape}, skipping.")

        if processed_chunks:
            full_streamed_audio = torch.cat(processed_chunks, dim=1) # Concat along time T
            full_streamed_audio = full_streamed_audio.unsqueeze(1) # Add channel dim back [B, 1, T]
            print(f"Concatenated streamed audio shape: {full_streamed_audio.shape}")

            # Save streaming output (already detached and on CPU)
            torchaudio.save(OUTPUT_STREAMING, full_streamed_audio.squeeze(1), SAMPLE_RATE)
            print(f"Saved streaming reconstructed audio to: {OUTPUT_STREAMING}")

            # Compare length to non-streamed version
            if decoded_waveform is not None and abs(full_streamed_audio.shape[2] - decoded_waveform.detach().shape[2]) < 5:
                 print("Streamed audio length is consistent with non-streamed.")
            elif decoded_waveform is not None:
                 print(f"WARNING: Streamed audio length ({full_streamed_audio.shape[2]}) differs from non-streamed ({decoded_waveform.detach().shape[2]}).")
            print("Streaming decode test completed.")
        else:
            print("Warning: No valid chunks were collected for streaming concatenation.")
    else:
        print("WARNING: No chunks were decoded in streaming test.")

except Exception as e:
    print(f"ERROR during streaming decode test: {e}")

print("\n--- Mimi Verification Script Finished ---") 