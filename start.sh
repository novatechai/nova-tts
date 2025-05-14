#!/bin/bash

echo "-------------------------------------------------------------------"
echo "Mimi-TTS Nova Training Start Script"
echo "-------------------------------------------------------------------"
echo ""
echo "IMPORTANT: Make sure your Python virtual environment (.venv) is"
echo "activated before running this script, or source it here:"
echo "  source .venv/bin/activate"
echo ""
echo "Please also ensure your training parameters in 'src/config.py'"
echo "(e.g., EPOCHS, BATCH_SIZE, DATASET_NAME) are set as desired."
echo ""
echo "If using Weights & Biases, ensure you are logged in: wandb login"
echo ""
echo "Starting training..."
echo "-------------------------------------------------------------------"

# Activate the environment (optional, if not already active)
# If you always activate your environment manually before running the script,
# you can comment out or remove the next two lines.
# if [ -f ".venv/bin/activate" ]; then
#   source .venv/bin/activate
# else
#   echo "Warning: .venv/bin/activate not found. Please ensure your environment is active."
# fi

python src/train.py

echo "-------------------------------------------------------------------"
echo "Training script finished."
echo "-------------------------------------------------------------------" 