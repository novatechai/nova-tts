# src/train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau # Import scheduler
import os
import time
import numpy as np # For infinity
import logging # Import logging
import wandb # Import wandb

# --- Suppress Specific Phonemizer Warning --- #
class PhonemizerFilter(logging.Filter):
    def filter(self, record):
        # Ignore the specific warning message from phonemizer
        return 'words count mismatch' not in record.getMessage()

# Get the root logger or a specific logger if phonemizer uses one (check its code)
# Adding to root logger for simplicity, might need adjustment if too broad
logging.getLogger().addFilter(PhonemizerFilter())
# --- End Warning Suppression --- #

# Ensure project root is in path for imports
import sys
if '.' not in sys.path:
    sys.path.append('.')

import src.config as config
from src.model import NovaTTS
# Need to import the dataset/dataloader later
from src.dataset import create_dataloaders # Example name

print("Starting Training Setup...")

def calculate_loss(pred_logits, target_codes, criterion):
    """
    Calculates the CrossEntropyLoss, reshaping inputs as needed.

    Args:
        pred_logits (torch.Tensor): Model predictions [B, N_Codebooks, T_Code, Vocab_Size].
        target_codes (torch.Tensor): Ground truth codes [B, N_Codebooks, T_Code].
        criterion: The loss function (e.g., nn.CrossEntropyLoss).

    Returns:
        torch.Tensor: The calculated loss value.
    """
    B, N, T, V = pred_logits.shape
    
    # Reshape for CrossEntropyLoss:
    # Input: [B * N * T, V] 
    # Target: [B * N * T]
    # Permute pred_logits to put Vocab_Size last: [B, N, T, V] -> [B, N, V, T] is WRONG
    # Need: [B, N, T, V] -> contiguous -> view -> [B*N*T, V]
    pred_logits_flat = pred_logits.contiguous().view(B * N * T, V)
    
    # Reshape target_codes: [B, N, T] -> contiguous -> view -> [B * N * T]
    target_codes_flat = target_codes.contiguous().view(B * N * T)
    
    loss = criterion(pred_logits_flat, target_codes_flat)
    
    return loss

def evaluate(model, dataloader, criterion, device):
    """Evaluates the model on the given dataloader."""
    model.eval() # Set model to evaluation mode
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad(): # Disable gradient calculations
        for i, batch_dict in enumerate(dataloader): # Dataloader yields the dict directly
            try:
                text_ids = batch_dict["text_ids"].to(device)
                attn_mask = batch_dict["attention_mask"].to(device)
                ref_wave = batch_dict["ref_audio_waveform"].to(device)
                target_codes = batch_dict["target_mimi_codes"].to(device)
            except (KeyError, IndexError, AttributeError, ValueError) as e:
                print(f"Error processing validation batch {i}: {repr(e)}")
                print(f"Validation Batch content: {repr(batch_dict)}") # Print the dict
                continue

            pred_logits = model(
                text_input_ids=text_ids,
                text_attention_mask=attn_mask,
                ref_audio_waveform=ref_wave,
                target_mimi_codes=target_codes
            )
            loss = calculate_loss(pred_logits, target_codes, criterion)
            total_loss += loss.item()
            num_batches += 1
            
            # Optional: Log validation batch progress
            # if (i + 1) % 10 == 0:
            #     print(f"  Validation Batch {i+1}/{len(dataloader)}")
            
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    return avg_loss

def main():
    print(f"Using device: {config.DEVICE}")

    # --- Initialize WandB --- #
    try:
        wandb.init(
            project="mimi-tts-nova", # Project name on WandB
            config={
                "learning_rate": config.LEARNING_RATE,
                "epochs": config.EPOCHS,
                "batch_size": config.BATCH_SIZE,
                "dataset": config.DATASET_NAME,
                "text_encoder": config.TEXT_ENCODER_MODEL,
                "speaker_encoder": config.SPEAKER_ENCODER_MODEL,
                "generator_d_model": config.GENERATOR_D_MODEL,
                "generator_nhead": config.GENERATOR_NHEAD,
                "generator_num_layers": config.GENERATOR_NUM_LAYERS,
                "generator_dim_ff": config.GENERATOR_DIM_FF,
                "mimi_codebooks": config.MIMI_NUM_CODEBOOKS,
                "mimi_vocab_size": config.MIMI_VOCAB_SIZE,
                # Add any other relevant config parameters
            }
        )
        print("WandB initialized.")
        wandb_enabled = True
    except Exception as e:
        print(f"Warning: Failed to initialize WandB: {e}. Training will proceed without WandB logging.")
        wandb_enabled = False

    # --- Setup Output Dirs ---
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    
    # --- 1. Model ---
    print("Loading NovaTTS model...")
    model = NovaTTS(config).to(config.DEVICE)
    print("Model loaded.")

    # --- 2. Data ---
    print("Setting up Dataloaders...")
    train_loader, val_loader = create_dataloaders(config)
    # print(f"Train dataset size: {len(train_loader.dataset)}, Val dataset size: {len(val_loader.dataset)}")
    print("Dataloaders setup complete.")
    # print("WARNING: Dataloader setup is a placeholder.")
    # Dummy loader for structure testing
    # Format: (text_ids, attn_mask, ref_wave, target_codes) - tensors *without* batch dim
    # dummy_train_data = [(
    #     torch.randint(0,1000,(50,)),                 # text_ids [T_text]
    #     torch.ones(50,),                             # attn_mask [T_text]
    #     torch.randn(config.SPEAKER_ENCODER_SAMPLE_RATE*3), # ref_wave [T_audio]
    #     torch.randint(0, config.MIMI_VOCAB_SIZE,
    #                   (config.MIMI_NUM_CODEBOOKS, 100)) # target_codes [N, T_codes]
    # )] * 10 # Repeat the same single-item batch 10 times
    # # DataLoader with batch_size=1 will add the batch dimension
    # train_loader = DataLoader(dummy_train_data, batch_size=1) # Set batch_size=1

    # --- 3. Loss Function ---
    # Ignore padding index if applicable (needs coordination with data prep)
    # criterion = nn.CrossEntropyLoss(ignore_index=config.PAD_TOKEN_ID) 
    criterion = nn.CrossEntropyLoss() 
    print("Loss function defined (CrossEntropyLoss).")

    # --- 4. Optimizer ---
    # Only optimize the MimiCodeGenerator parameters
    trainable_params = model.mimi_code_generator.parameters()
    optimizer = optim.AdamW(
        trainable_params, 
        lr=config.LEARNING_RATE, 
        # weight_decay=config.WEIGHT_DECAY # Add if defined in config
    )
    print(f"Optimizer defined (AdamW, lr={config.LEARNING_RATE}) for MimiCodeGenerator.")
    
    # --- 5. Scheduler (Optional) ---
    # TODO: Implement learning rate scheduler if desired -> IMPLEMENTED
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3)
    print(f"Learning rate scheduler: {scheduler.__class__.__name__}")
    
    # --- 6. Training Loop ---
    print("\nStarting Training Loop...")
    start_time = time.time()
    best_val_loss = np.inf # Track best validation loss

    for epoch in range(config.EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{config.EPOCHS} ---")
        model.train() # Set model *except* frozen parts to training mode
        # Ensure frozen models remain in eval mode even after model.train()
        if config.FREEZE_TEXT_ENCODER and model.text_encoder_model:
            model.text_encoder_model.eval()
        if model.speaker_encoder:
            model.speaker_encoder.eval()
            
        epoch_loss = 0.0
        num_batches = 0 # Keep track for averaging loss

        # Simple progress tracking
        batch_start_time = time.time()
        
        for i, batch_list in enumerate(train_loader):
            optimizer.zero_grad()

            # --- Data Preparation ---
            # Ensure data is on the correct device
            try:
                # print(f"DBG: Processing batch {i}")
                # print(f"DBG: Type of batch_list: {type(batch_list)}")
                # print(f"DBG: Value of batch_list: {repr(batch_list)}")
                
                # # Check if batch_list is a list and non-empty before indexing
                # if isinstance(batch_list, list) and len(batch_list) > 0:
                #     print(f"DBG: Type of batch_list[0]: {type(batch_list[0])}")
                #     print(f"DBG: Value of batch_list[0]: {repr(batch_list[0])}")
                #     batch_dict = batch_list[0] # Get the dictionary from the list
                # else:
                #     print(f"Error: batch_list is not a non-empty list for batch {i}. Skipping.")
                #     continue
                
                # With standard Dataset/DataLoader, the yielded item *is* the batch dict
                batch_dict = batch_list 
                    
                # Now access items from batch_dict
                text_ids = batch_dict["text_ids"].to(config.DEVICE)
                # Load the original attention mask from tokenizer/dataset
                attn_mask = batch_dict["attention_mask"].to(config.DEVICE) 
                ref_wave = batch_dict["ref_audio_waveform"].to(config.DEVICE)
                target_codes = batch_dict["target_mimi_codes"].to(config.DEVICE)
                # Get the new padding masks
                text_pad_mask = batch_dict["text_padding_mask"].to(config.DEVICE)
                target_pad_mask = batch_dict["target_padding_mask"].to(config.DEVICE)
                
            except (KeyError, IndexError, AttributeError, ValueError, TypeError) as e: 
                 print(f"Error processing batch {i}: {repr(e)}") 
                 print(f"Problematic Batch content: {repr(batch_list)}") 
                 continue # Skip batch if unpacking fails


            # --- Forward Pass ---
            pred_logits = model(
                text_input_ids=text_ids,
                text_attention_mask=attn_mask, # Pass the original tokenizer mask
                ref_audio_waveform=ref_wave,
                target_mimi_codes=target_codes, 
                # Pass the padding masks for decoder
                text_padding_mask=text_pad_mask,
                target_padding_mask=target_pad_mask
            )

            # --- Loss Calculation ---
            loss = calculate_loss(pred_logits, target_codes, criterion)

            # --- Backward Pass & Optimization ---
            loss.backward()
            # Optional: Gradient clipping
            # torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            # --- Logging (Basic) ---
            if (i + 1) % 1 == 0: # Log every batch for dummy data
                 batch_time = time.time() - batch_start_time
                 print(f"  Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}, Time: {batch_time:.2f}s")
                 batch_start_time = time.time() # Reset timer for next batch

        # --- End of Epoch ---
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
        print(f"Epoch {epoch+1} finished. Average Training Loss: {avg_epoch_loss:.4f}")
        
        # Log epoch metrics to WandB
        log_dict = {"epoch": epoch + 1, "train_loss": avg_epoch_loss}
        
        # --- Validation Loop ---
        if val_loader:
            print(f"Running validation for Epoch {epoch+1}...")
            val_start_time = time.time()
            avg_val_loss = evaluate(model, val_loader, criterion, config.DEVICE)
            val_time = time.time() - val_start_time
            print(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}, Time: {val_time:.2f}s")
            
            # Log validation loss to WandB
            log_dict["val_loss"] = avg_val_loss
            log_dict["lr"] = optimizer.param_groups[0]['lr'] # Log current learning rate
            
            # --- Model Saving (Best Only) --- 
            # TODO: Implement saving based on best validation loss -> IMPLEMENTED
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                checkpoint_path = os.path.join(config.MODEL_SAVE_DIR, f"novatts_best.pth") # Save only best
                save_payload = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': avg_epoch_loss,
                    'val_loss': avg_val_loss, 
                    'scheduler_state_dict': scheduler.state_dict() # Save scheduler state
                }
                try:
                    torch.save(save_payload, checkpoint_path)
                    print(f"** New best validation loss ({best_val_loss:.4f}). Checkpoint saved to {checkpoint_path} **")
                except Exception as e:
                    print(f"Error saving checkpoint: {e}")
            else:
                print(f"Validation loss ({avg_val_loss:.4f}) did not improve from best ({best_val_loss:.4f}). Not saving checkpoint.")
            
            # --- LR Scheduler Step --- 
            # TODO: Update learning rate scheduler if used -> IMPLEMENTED
            scheduler.step(avg_val_loss) # Step scheduler based on validation loss
            
        else:
            print("No validation loader available, skipping validation, scheduler step, and best model saving.")
            avg_val_loss = None
            # Save checkpoint every epoch if no validation is available
            checkpoint_path = os.path.join(config.MODEL_SAVE_DIR, f"novatts_epoch_{epoch+1}.pth")
            save_payload = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_epoch_loss,
                'val_loss': avg_val_loss 
            }
            try:
                torch.save(save_payload, checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path} (no validation)")
            except Exception as e:
                print(f"Error saving checkpoint: {e}")
                
        # Log metrics for the epoch
        if wandb_enabled:
            wandb.log(log_dict)

    # --- End of Training ---
    total_training_time = time.time() - start_time
    print(f"\nTraining finished in {total_training_time:.2f} seconds.")
    
    # --- Finish WandB Run --- #
    if wandb_enabled:
        wandb.finish()
        print("WandB run finished.")

if __name__ == "__main__":
    main() 