# src/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from speechbrain.inference.speaker import EncoderClassifier
from transformers import AutoModel, AutoConfig # For Qwen3 later
import os
import time # For generate_codes placeholder
import math
import logging # Import logging
from typing import Optional, Tuple, Generator # Import necessary types and Generator

# Assuming config.py is in the src directory, adjust path if needed
import sys
if '.' not in sys.path:
    sys.path.append('.')
import src.config as config

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants for Fallbacks (Example) ---
# Consider moving these to config or defining more formally
DEFAULT_SPEAKER_DIM = 192
DEFAULT_TEXT_DIM_SMALL = 1024 # For 0.6B model
DEFAULT_TEXT_DIM_LARGE = 3072 # For 4B model

print("Defining NovaTTS model structure...")

# --- Positional Encoding --- Helper class
class PositionalEncoding(nn.Module):
    """Injects positional information into sequence embeddings."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """
        Args:
            d_model (int): The embedding dimension.
            dropout (float): Dropout value.
            max_len (int): Maximum sequence length.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # Add batch dimension [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Adds positional encoding to the input tensor.

        Args:
            x (torch.Tensor): Input tensor, expects shape [Batch, SeqLen, Dim].

        Returns:
            torch.Tensor: Output tensor with added positional encoding [Batch, SeqLen, Dim].
        """
        # Ensure pe batch dim matches x batch dim if needed (usually not necessary with broadcasting)
        # self.pe shape: [1, max_len, d_model]
        # x shape: [B, T, D]
        # Slicing pe: [1, T, D]
        # Broadcasting handles the addition along the batch dimension.
        x = x + self.pe[:, :x.size(1), :] # Assumes batch_first=True
        return self.dropout(x)

# --- Mimi Code Generator --- #
class MimiCodeGenerator(nn.Module):
    """Transformer Decoder model to generate Mimi codes based on text and speaker conditioning."""
    def __init__(self, config, speaker_embedding_dim: int, text_embedding_dim: int):
        """
        Args:
            config: Configuration object containing model hyperparameters 
                    (GENERATOR_D_MODEL, MIMI_NUM_CODEBOOKS, MIMI_VOCAB_SIZE, etc.).
            speaker_embedding_dim (int): Dimension of the input speaker embedding.
            text_embedding_dim (int): Dimension of the input text embeddings.
        """
        super().__init__()
        self.config = config
        self.d_model = config.GENERATOR_D_MODEL
        self.num_codebooks = config.MIMI_NUM_CODEBOOKS
        self.vocab_size = config.MIMI_VOCAB_SIZE
        self.speaker_embedding_dim = speaker_embedding_dim
        self.text_embedding_dim = text_embedding_dim

        # --- Input Conditioning Layers ---
        self.speaker_proj = nn.Linear(self.speaker_embedding_dim, self.d_model)
        self.text_proj = nn.Linear(self.text_embedding_dim, self.d_model) if self.text_embedding_dim != self.d_model else nn.Identity()

        # --- Mimi Code Embeddings (Input to Decoder) ---
        self.code_embeddings = nn.ModuleList(
            [nn.Embedding(self.vocab_size, self.d_model) 
             for _ in range(self.num_codebooks)]
        )
        # TODO: Revisit vocab size +1 decision based on actual data/padding used.

        # --- Positional Encoding ---
        self.pos_encoder = PositionalEncoding(self.d_model, config.GENERATOR_DROPOUT)

        # --- Transformer Decoder ---
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=config.GENERATOR_NHEAD,
            dim_feedforward=config.GENERATOR_DIM_FF,
            dropout=config.GENERATOR_DROPOUT,
            batch_first=True,
            activation=F.gelu 
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.GENERATOR_NUM_LAYERS
        )

        # --- Output Layer ---
        self.output_projection = nn.Linear(self.d_model, self.vocab_size * self.num_codebooks)

        logger.info("MimiCodeGenerator initialized.")

    def forward(self,
                text_embeddings: torch.Tensor,
                speaker_embedding: torch.Tensor,
                target_codes: torch.Tensor,
                # Accept padding masks
                memory_key_padding_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        """
        Forward pass for training (Teacher Forcing).

        Args:
            text_embeddings (torch.Tensor): Text encoder outputs [B, T_text, D_text].
            speaker_embedding (torch.Tensor): Speaker encoder output [B, D_speaker].
            target_codes (torch.Tensor): Ground truth codes [B, N_codebooks, T_code].
            memory_key_padding_mask (Optional[torch.Tensor]): Mask for text encoder memory [B, 1 + T_text].
                                                            False for valid tokens, True for padding.
            tgt_key_padding_mask (Optional[torch.Tensor]): Mask for target codes [B, T_code].
                                                         False for valid tokens, True for padding.

        Returns:
            torch.Tensor: Logits for each codebook [B, N_codebooks, T_code, Vocab_Size].
        """
        B, _, T_code = target_codes.shape

        # 1. Prepare Memory (from text and speaker)
        projected_text_embeddings = self.text_proj(text_embeddings)         # [B, T_text, D_model]
        # Project speaker embedding: [B, 1, D_speaker] -> [B, 1, D_model]
        speaker_cond = self.speaker_proj(speaker_embedding)
        
        # --- DEBUG SHAPES --- 
        # print(f"DBG Generator Fwd: speaker_cond shape: {speaker_cond.shape}")
        # print(f"DBG Generator Fwd: projected_text_embeddings shape: {projected_text_embeddings.shape}")
        # --- END DEBUG --- 

        # Combine by prepending speaker condition token
        memory = torch.cat([speaker_cond, projected_text_embeddings], dim=1) # [B, 1 + T_text, D_model]

        # 2. Prepare Target Embeddings (Decoder Input)
        # Embed each codebook and sum
        # target_codes shape: [B, N, T] -> [B, T, N]
        target_codes_permuted = target_codes.permute(0, 2, 1)
        
        # Embed each codebook and sum along the codebook dimension
        # We need to embed each codebook separately using its dedicated embedding layer
        target_embeddings_sum = torch.zeros(B, T_code, self.d_model, device=target_codes.device)
        for i in range(self.num_codebooks):
            codes_i = target_codes_permuted[:, :, i] # Shape: [B, T_code]
            # Apply the i-th embedding layer
            embedded_codes_i = self.code_embeddings[i](codes_i) # Shape: [B, T_code, D_model]
            target_embeddings_sum += embedded_codes_i
            
        # target_embeddings_sum is now [B, T_code, D_model]

        # Add positional encoding
        # The PositionalEncoding expects [SeqLen, Batch, Dim] or [Batch, SeqLen, Dim]
        # Our PE implementation assumes [SeqLen, Batch, Dim], but takes [Batch, SeqLen, Dim] if batch_first=True (default)
        # Let's ensure our input matches the expected format (check PE implementation or adjust here)
        # Assuming PE expects [Batch, SeqLen, Dim]
        tgt = self.pos_encoder(target_embeddings_sum) # [B, T_code, D_model]

        # 3. Create Masks
        # Target mask: prevent attending to future positions
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(T_code, device=tgt.device)

        # 4. Transformer Decoder Pass
        output = self.transformer_decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask, # Pass masks
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        # Output shape: [B, T_code, D_model]

        # 5. Project to Logits
        logits = self.output_projection(output) # [B, T_code, Vocab_Size * N_codebooks]

        # Reshape logits to [B, N_codebooks, T_code, Vocab_Size]
        logits = logits.view(B, T_code, self.num_codebooks, self.vocab_size)
        logits = logits.permute(0, 2, 1, 3)

        return logits

# --- NovaTTS Main Model --- #
class NovaTTS(nn.Module):
    """Main Text-to-Speech model combining text encoder, speaker encoder, and Mimi code generator."""
    
    def __init__(self, model_config=config):
        """
        Initializes the NovaTTS model components by loading prerequisite models.

        Args:
            model_config: Configuration object (e.g., from src.config)
        """
        super().__init__()
        self.config = model_config
        logger.info("Initializing NovaTTS model...")

        # --- Load Prerequisite Encoders ---
        self.speaker_encoder, self.speaker_embedding_dim = self._load_speaker_encoder()
        self.text_encoder_model, self.text_embedding_dim = self._load_text_encoder()

        # --- 3. Mimi Code Generator (Decoder) ---
        logger.info("Initializing Mimi Code Generator...")
        if self.speaker_embedding_dim is None or self.text_embedding_dim is None:
             logger.error("Speaker or Text embedding dimension not determined. Cannot initialize Generator.")
             raise ValueError("Speaker or Text embedding dimension not determined. Cannot initialize Generator.")

        self.mimi_code_generator = MimiCodeGenerator(
            config=self.config,
            speaker_embedding_dim=self.speaker_embedding_dim,
            text_embedding_dim=self.text_embedding_dim
        )
        logger.info("Mimi Code Generator initialized successfully.")
        logger.info("NovaTTS initialization complete.")

    def _load_speaker_encoder(self) -> Tuple[Optional[EncoderClassifier], Optional[int]]:
        """Loads the speaker encoder model and determines its embedding dimension."""
        logger.info(f"Loading Speaker Encoder: {self.config.SPEAKER_ENCODER_MODEL}")
        speaker_embedding_dim: Optional[int] = None
        speaker_encoder: Optional[EncoderClassifier] = None
        try:
            # Use a temporary directory within the project for downloaded models
            speaker_encoder_savedir = os.path.join(self.config.MODEL_SAVE_DIR, "speechbrain_cache")
            os.makedirs(speaker_encoder_savedir, exist_ok=True)

            speaker_encoder = EncoderClassifier.from_hparams(
                source=self.config.SPEAKER_ENCODER_MODEL,
                savedir=speaker_encoder_savedir,
                run_opts={"device": self.config.DEVICE} # Load directly to target device
            )
            speaker_encoder.eval() # Set to eval mode
            # Freeze speaker encoder weights
            for param in speaker_encoder.parameters():
                 param.requires_grad = False
            logger.info("Speaker Encoder loaded successfully and frozen.")

            # Get embedding dimension
            try:
                # Use actual expected sample rate from config
                dummy_audio = torch.randn(1, self.config.SPEAKER_ENCODER_SAMPLE_RATE).to(self.config.DEVICE)
                with torch.no_grad():
                    dummy_embedding = speaker_encoder.encode_batch(dummy_audio)
                speaker_embedding_dim = dummy_embedding.shape[-1]
                logger.info(f"Determined Speaker Embedding Dimension: {speaker_embedding_dim}")
            except Exception as e_dim:
                logger.warning(f"Could not automatically determine speaker embedding dim: {e_dim}")
                # Use default/fallback if auto-detection fails
                speaker_embedding_dim = DEFAULT_SPEAKER_DIM 
                logger.warning(f"Assuming Speaker Embedding Dimension: {speaker_embedding_dim}")

        except Exception as e:
            logger.error(f"Failed to load Speaker Encoder: {e}", exc_info=True) # Log traceback
            speaker_encoder = None
            # Use default/fallback if loading fails
            speaker_embedding_dim = DEFAULT_SPEAKER_DIM 
            logger.warning(f"Using fallback Speaker Embedding Dimension: {speaker_embedding_dim}")
            
        return speaker_encoder, speaker_embedding_dim

    def _load_text_encoder(self) -> Tuple[Optional[AutoModel], Optional[int]]:
        """Loads the text encoder model and determines its embedding dimension."""
        logger.info(f"Loading Text Encoder: {self.config.TEXT_ENCODER_MODEL}")
        text_embedding_dim: Optional[int] = None
        text_encoder_model: Optional[AutoModel] = None
        try:
             text_encoder_model = AutoModel.from_pretrained(
                 self.config.TEXT_ENCODER_MODEL,
                 trust_remote_code=True,
                 # torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
             ).to(self.config.DEVICE)

             # Store config for convenience if needed later
             # self.text_encoder_config = text_encoder_model.config 
             text_embedding_dim = text_encoder_model.config.hidden_size
             logger.info(f"Text Encoder loaded successfully. Embedding dim: {text_embedding_dim}")

             if self.config.FREEZE_TEXT_ENCODER:
                 logger.info("Freezing Text Encoder parameters.")
                 for param in text_encoder_model.parameters():
                    param.requires_grad = False
             else:
                 logger.info("Text Encoder parameters will be trainable.")

        except Exception as e:
             logger.error(f"Failed to load Text Encoder: {e}", exc_info=True) # Log traceback
             text_encoder_model = None
             # Try to get dim from config as fallback
             try:
                 logger.info("Attempting to load text encoder config for fallback dimension...")
                 temp_config = AutoConfig.from_pretrained(self.config.TEXT_ENCODER_MODEL, trust_remote_code=True)
                 text_embedding_dim = temp_config.hidden_size
                 logger.warning(f"Using fallback text embedding dim from config: {text_embedding_dim}")
             except Exception as e_cfg:
                 logger.warning(f"Could not load config for fallback dimension: {e_cfg}")
                 # Use hardcoded fallback based on model name as last resort
                 if "0.6B" in self.config.TEXT_ENCODER_MODEL:
                    text_embedding_dim = DEFAULT_TEXT_DIM_SMALL
                 else:
                     # Assume larger model if name doesn't contain known small size indicator
                     text_embedding_dim = DEFAULT_TEXT_DIM_LARGE 
                 logger.warning(f"Using hardcoded fallback text embedding dim: {text_embedding_dim}")
                 
        return text_encoder_model, text_embedding_dim
        
    def forward(self, 
                text_input_ids: torch.Tensor, 
                text_attention_mask: torch.Tensor, 
                ref_audio_waveform: torch.Tensor, 
                target_mimi_codes: torch.Tensor,
                # Add masks from collate_fn
                text_padding_mask: Optional[torch.Tensor] = None,
                target_padding_mask: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        """
        Defines the forward pass for TRAINING.

        Args:
            text_input_ids (torch.Tensor): Batch of tokenized text IDs [B, T_text].
            text_attention_mask (torch.Tensor): Attention mask for text inputs [B, T_text].
            ref_audio_waveform (torch.Tensor): Batch of raw audio waveforms for speaker encoding 
                                            [B, N_samples]. (Should be resampled to SPEAKER_ENCODER_SAMPLE_RATE).
            target_mimi_codes (torch.Tensor): Ground truth Mimi codes [B, N_codebooks, T_code].
            text_padding_mask (Optional[torch.Tensor]): Padding mask for text_input_ids [B, T_text]. 
                                                      True for valid, False for padding.
            target_padding_mask (Optional[torch.Tensor]): Padding mask for target_mimi_codes [B, T_code].
                                                       True for valid, False for padding.

        Returns:
            torch.Tensor: Output logits from the Mimi Code Generator [B, N_codebooks, T_code, Vocab_Size].
        
        Raises:
            RuntimeError: If speaker or text encoder were not loaded successfully during init.
        """
        # 1. Get Speaker Embedding
        if self.speaker_encoder is None:
            logger.error("Forward pass failed: Speaker encoder is not loaded.")
            raise RuntimeError("Speaker encoder not loaded.")
        # Ensure waveform is on the correct device
        ref_audio_waveform = ref_audio_waveform.to(self.config.DEVICE)

        # Speaker encoder is frozen, so no gradients needed
        with torch.no_grad(): 
            # encode_batch expects [B, N_samples]
            speaker_embeddings = self.speaker_encoder.encode_batch(ref_audio_waveform)
            # print(f"DBG NovaTTS Fwd: speaker_embeddings raw shape: {speaker_embeddings.shape}") # REMOVE DEBUG

        # 2. Get Text Embeddings
        if self.text_encoder_model is None:
            logger.error("Forward pass failed: Text encoder is not loaded.")
            raise RuntimeError("Text encoder not loaded.")
        # Ensure inputs are on correct device
        text_input_ids = text_input_ids.to(self.config.DEVICE)
        text_attention_mask = text_attention_mask.to(self.config.DEVICE)

        # Run text encoder (potentially with grads if not frozen)
        grad_context = torch.no_grad() if self.config.FREEZE_TEXT_ENCODER else torch.enable_grad()
        with grad_context:
            text_outputs = self.text_encoder_model(
                input_ids=text_input_ids,
                attention_mask=text_attention_mask
            )
            text_embeddings = text_outputs.last_hidden_state # [B, T_text, D_text]

        # 3. Generate Mimi Code Logits using the Generator
        # Ensure target codes are on the correct device
        target_mimi_codes = target_mimi_codes.to(self.config.DEVICE)

        # --- Create Padding Masks for Transformer --- 
        # Determine batch size
        B = text_input_ids.shape[0]
        
        # Transformer expects True for padded positions, False for valid positions.
        # Our collate_fn masks are True for valid, False for padding. Invert them.
        
        mem_pad_mask = None
        if text_padding_mask is not None:
            # Mask for memory: [B, 1 + T_text]
            # Speaker token is never padded (False)
            speaker_pad_mask = torch.zeros(B, 1, dtype=torch.bool, device=self.config.DEVICE)
            # Invert text mask (False becomes True for padding)
            inverted_text_mask = ~text_padding_mask.to(self.config.DEVICE)
            mem_pad_mask = torch.cat([speaker_pad_mask, inverted_text_mask], dim=1)
            
        tgt_pad_mask = None
        if target_padding_mask is not None:
            # Invert target mask (False becomes True for padding)
            tgt_pad_mask = ~target_padding_mask.to(self.config.DEVICE)

        output_logits = self.mimi_code_generator(
            text_embeddings=text_embeddings,
            speaker_embedding=speaker_embeddings,
            target_codes=target_mimi_codes, 
            memory_key_padding_mask=mem_pad_mask, # Pass constructed masks
            tgt_key_padding_mask=tgt_pad_mask
        )

        # Output shape: [B, N_codebooks, T_code, Vocab_Size]
        return output_logits

    @torch.no_grad() # Disable gradients for inference
    def generate_codes(self, 
                       text_input_ids: torch.Tensor, 
                       text_attention_mask: torch.Tensor, 
                       ref_audio_waveform: torch.Tensor, 
                       max_length: int = 1000, 
                       start_token_id: int = 0, 
                       # Pass the actual EOS ID from config
                       eos_token_id: Optional[int] = config.MIMI_EOS_TOKEN_ID, 
                       temperature: float = 1.0, 
                       top_k: Optional[int] = None,
                       chunk_size: int = 40 
                       ) -> Generator[torch.Tensor, None, None]: 
        """
        Defines the autoregressive generation for INFERENCE.
        Generates Mimi codes frame by frame and yields them in chunks.

        Args:
            text_input_ids (torch.Tensor): Tokenized text IDs [1, T_text]. (Batch size must be 1 for now).
            text_attention_mask (torch.Tensor): Attention mask [1, T_text].
            ref_audio_waveform (torch.Tensor): Raw audio waveform [1, N_samples].
            max_length (int): Maximum number of code frames to generate.
            start_token_id (int): ID used as the initial input for each codebook.
            eos_token_id (Optional[int]): ID indicating end of sequence. Generation stops if this is predicted.
                                         (Currently unused, relies on max_length).
            temperature (float): Softmax temperature for sampling. 1.0 = standard softmax.
            top_k (Optional[int]): If set, limits sampling to the top K most likely tokens.
            chunk_size (int): Number of code frames to generate before yielding.

        Returns:
            Generator[torch.Tensor, None, None]: A generator yielding chunks of codes 
                                                 with shape [1, N_codebooks, chunk_size].
            
        Raises:
            RuntimeError: If speaker or text encoder were not loaded successfully during init.
            NotImplementedError: If batch size > 1 is attempted.
        """
        B = text_input_ids.shape[0]
        if B != 1:
            # TODO: Implement batch generation if needed
            raise NotImplementedError("Batch size > 1 not yet supported for generate_codes")

        device = self.config.DEVICE
        num_codebooks = self.config.MIMI_NUM_CODEBOOKS
        d_model = self.mimi_code_generator.d_model
        vocab_size = self.config.MIMI_VOCAB_SIZE

        # --- Prepare Inputs (Done once) ---
        # 1. Speaker Embedding
        if self.speaker_encoder is None: raise RuntimeError("Speaker encoder not loaded.")
        ref_audio_waveform = ref_audio_waveform.to(device)
        speaker_embeddings = self.speaker_encoder.encode_batch(ref_audio_waveform) # [1, D_speaker]

        # 2. Text Embeddings
        if self.text_encoder_model is None: raise RuntimeError("Text encoder not loaded.")
        text_input_ids = text_input_ids.to(device)
        text_attention_mask = text_attention_mask.to(device)
        text_outputs = self.text_encoder_model(input_ids=text_input_ids, attention_mask=text_attention_mask)
        text_embeddings = text_outputs.last_hidden_state # [1, T_text, D_text]

        # 3. Prepare Memory for Decoder
        projected_text_embeddings = self.mimi_code_generator.text_proj(text_embeddings) # [1, T_text, D_model]
        speaker_cond = self.mimi_code_generator.speaker_proj(speaker_embeddings)
        memory = torch.cat([speaker_cond, projected_text_embeddings], dim=1) # [1, 1 + T_text, D_model]
        
        # Create memory padding mask based on input text_attention_mask
        # Assume text_attention_mask is 1 for valid, 0 for padding
        # Transformer expects True for padding, False for valid.
        mem_pad_mask = None
        if text_attention_mask is not None:
            speaker_pad_mask = torch.zeros(1, 1, dtype=torch.bool, device=device) # Speaker token is not padded
            # Invert the attention mask (0 becomes True -> padding)
            inverted_text_mask = (text_attention_mask == 0) 
            mem_pad_mask = torch.cat([speaker_pad_mask, inverted_text_mask], dim=1)
            

        # --- Autoregressive Generation Loop ---
        generated_codes_list = [] # Store generated codes temporarily
        # Start with the initial token, will be sliced off before returning first chunk
        current_generated_codes = torch.full((1, num_codebooks, 1), start_token_id, dtype=torch.long, device=device)

        for i in range(max_length):
            current_seq_len = current_generated_codes.shape[-1]

            # Embed the currently generated codes
            target_embeddings_sum = torch.zeros(1, current_seq_len, d_model, device=device)
            target_codes_permuted = current_generated_codes.permute(0, 2, 1) # [1, current_seq_len, N]
            for codebook_idx in range(num_codebooks):
                codes_i = target_codes_permuted[:, :, codebook_idx] # [1, current_seq_len]
                target_embeddings_sum += self.mimi_code_generator.code_embeddings[codebook_idx](codes_i)

            # Add positional encoding
            tgt = self.mimi_code_generator.pos_encoder(target_embeddings_sum) # [1, current_seq_len, D_model]

            # Create masks for the current length
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(current_seq_len, device=device)
            # Target padding mask not needed in generation as we build the sequence one by one

            # Decoder pass
            output = self.mimi_code_generator.transformer_decoder(
                tgt=tgt, 
                memory=memory, 
                tgt_mask=tgt_mask,
                memory_key_padding_mask=mem_pad_mask
            ) # [1, current_seq_len, D_model]

            # Project to logits for the *last* time step only
            last_step_output = output[:, -1:, :] # [1, 1, D_model]
            logits = self.mimi_code_generator.output_projection(last_step_output) # [1, 1, Vocab_Size * N]

            # Reshape logits: [1, 1, N, Vocab_Size] -> [1, N, Vocab_Size]
            logits = logits.view(1, 1, num_codebooks, vocab_size).squeeze(1) 

            # --- Sampling --- # 
            if temperature != 1.0:
                logits = logits / temperature
            if top_k is not None and top_k > 0:
                top_k = min(top_k, vocab_size)
                indices_to_remove = logits < torch.topk(logits, top_k, dim=-1)[0][..., -1, None]
                logits = logits.masked_fill(indices_to_remove, float('-inf'))

            probs = F.softmax(logits, dim=-1) # [1, N, V]
            next_code_ids = torch.multinomial(probs.squeeze(0), num_samples=1) # Input: [N, V] -> Output [N, 1]
            next_code_ids = next_code_ids.unsqueeze(0) # Add back batch dim -> [1, N, 1]

            # Store the newly generated frame (code ids)
            generated_codes_list.append(next_code_ids)
            
            # Update the input for the next step
            current_generated_codes = torch.cat([current_generated_codes, next_code_ids], dim=-1)

            # --- Yield Chunk --- #
            if len(generated_codes_list) >= chunk_size:
                chunk_to_yield = torch.cat(generated_codes_list, dim=-1) 
                yield chunk_to_yield
                generated_codes_list = [] 

            # --- Check EOS (Using first codebook as indicator) --- #
            if eos_token_id is not None and next_code_ids[0, 0, 0].item() == eos_token_id:
                logger.info(f"EOS token ({eos_token_id}) detected in codebook 0. Stopping generation.")
                break
            # --- End EOS Check --- #

        # Yield any remaining codes after the loop finishes
        if generated_codes_list:
            chunk_to_yield = torch.cat(generated_codes_list, dim=-1) 
            yield chunk_to_yield

        # Note: The original function returned generated_codes[:, :, 1:] 
        # which excluded the start token. The generator yields chunks as they are generated,
        # so the start token is effectively handled implicitly.


# --- Example Instantiation & Basic Tests ---
if __name__ == "__main__":
    print("\n--- Testing NovaTTS Instantiation & Forward/Generate ---")
    try:
        print("Instantiating model...")
        model = NovaTTS() # Uses defaults from config.py
        print("\nNovaTTS model instantiated successfully.")

        if model.speaker_encoder:
            print("Speaker Encoder loaded.")
        if model.text_encoder_model:
            print("Text Encoder loaded.")
        if model.mimi_code_generator:
            print("Mimi Code Generator initialized.")

        # --- Test Forward Pass --- #
        print("\n--- Testing Forward Pass ---")
        # Create dummy inputs (adjust shapes as needed)
        dummy_text_ids = torch.randint(0, 1000, (2, 50)).long() # Batch=2, SeqLen=50
        dummy_attn_mask = torch.ones((2, 50)).long()
        # Waveform length needs to match speaker encoder sample rate
        dummy_audio_len = config.SPEAKER_ENCODER_SAMPLE_RATE * 3 # 3 seconds
        dummy_ref_wave = torch.randn(2, dummy_audio_len) # Batch=2
        dummy_target_codes = torch.randint(0, config.MIMI_VOCAB_SIZE, (2, config.MIMI_NUM_CODEBOOKS, 100)).long() # B, N, T_code=100

        print("Running forward pass...")
        output_logits = model.forward(
            text_input_ids=dummy_text_ids,
            text_attention_mask=dummy_attn_mask,
            ref_audio_waveform=dummy_ref_wave,
            target_mimi_codes=dummy_target_codes,
            text_padding_mask=None,
            target_padding_mask=None
        )
        print(f"Forward pass successful. Output logits shape: {output_logits.shape}")
        # Expected: [2, 8, 100, 2049]
        expected_shape = (dummy_target_codes.shape[0], config.MIMI_NUM_CODEBOOKS, dummy_target_codes.shape[2], config.MIMI_VOCAB_SIZE)
        assert output_logits.shape == expected_shape, f"Expected shape {expected_shape}, got {output_logits.shape}"
        print("Forward pass output shape is correct.")


        # --- Test Generation Pass --- #
        print("\n--- Testing Generation Pass ---")
        # Use smaller inputs for generation test (Batch size = 1)
        dummy_gen_text_ids = torch.randint(0, 1000, (1, 30)).long()
        dummy_gen_attn_mask = torch.ones((1, 30)).long()
        dummy_gen_ref_wave = torch.randn(1, dummy_audio_len) # 3 seconds

        print("Running generate_codes...")
        # Use a small max_length for testing
        generated_output = model.generate_codes(
            text_input_ids=dummy_gen_text_ids,
            text_attention_mask=dummy_gen_attn_mask,
            ref_audio_waveform=dummy_gen_ref_wave,
            max_length=10 # Generate 10 frames
            # eos_token_id=config.MIMI_VOCAB_SIZE -1 # Example if last ID is EOS
        )
        print(f"Generation pass successful. Output codes shape: {generated_output.shape}")
        # Expected: [1, 8, 10]
        expected_gen_shape = (1, config.MIMI_NUM_CODEBOOKS, 10)
        assert generated_output.shape == expected_gen_shape, f"Expected shape {expected_gen_shape}, got {generated_output.shape}"
        print("Generated codes shape is correct.")

    except Exception as e:
        print(f"\nERROR during instantiation or testing: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- Model Implementation Test Finished ---") 