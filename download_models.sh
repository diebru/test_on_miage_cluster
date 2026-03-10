#!/bin/bash

echo "========================================================"
echo " STARTING MODELS AND ADAPTERS DOWNLOAD "
echo "========================================================"

# Check if huggingface-cli is installed, otherwise install it in the current environment
if ! command -v huggingface-cli &> /dev/null; then
    echo "Installing huggingface_hub[cli] in the Conda environment..."
    pip install -U "huggingface_hub[cli]"
fi

# Configurations: the 3 models used in the TokenSkip experiments
MODELS=("3B" "7B" "14B")
BASE_DIR="models"

# Create the models folder if it does not exist
mkdir -p $BASE_DIR

for SIZE in "${MODELS[@]}"; do
    echo "--------------------------------------------------------"
    echo " Downloading size: ${SIZE}"
    echo "--------------------------------------------------------"

    # 1. Download BASE Model (Qwen2.5-Instruct)
    BASE_MODEL_NAME="Qwen/Qwen2.5-${SIZE}-Instruct"
    BASE_LOCAL_DIR="${BASE_DIR}/Qwen2.5-${SIZE}-Instruct"
    
    echo " >> Downloading Base Model: ${BASE_MODEL_NAME}..."
    # Exclude .pth and .bin to download only safetensors (faster for vLLM)
    huggingface-cli download ${BASE_MODEL_NAME} --local-dir ${BASE_LOCAL_DIR} --exclude "*.pth" "*.bin"

    # 2. Download ADAPTER Model (TokenSkip-GSM8K)
    ADAPTER_MODEL_NAME="hemingkx/TokenSkip-Qwen2.5-${SIZE}-Instruct-GSM8K"
    ADAPTER_LOCAL_DIR="${BASE_DIR}/TokenSkip-Qwen2.5-${SIZE}-Instruct-GSM8K"
    
    echo " >> Downloading the Adapter: ${ADAPTER_MODEL_NAME}..."
    huggingface-cli download ${ADAPTER_MODEL_NAME} --local-dir ${ADAPTER_LOCAL_DIR}

    echo " >> Completed for ${SIZE}."
done

echo "========================================================"
echo " ALL DOWNLOADS HAVE BEEN COMPLETED."
echo " The models are located in the 'models/' folder"
echo "========================================================"