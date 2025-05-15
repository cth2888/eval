#!/bin/bash

set -x

export VLLM_ATTENTION_BACKEND=XFORMERS
export CUDA_VISIBLE_DEVICES=0,1,2,3
# Default values
BASE_PATHS="/root/cth/cth/eval"
MODEL_PATHS=("/root/cth/cth/models/deepscaler_high_entropy")
DATATYPES=("deepscaler_rest") #"amc" "minerva")
OUTPUT_DIR="${BASE_PATHS}/output"
N_PASSES=2
MAX_LENGTH=16384
TP_SIZE=1

# Create output directory if it doesn't exist
mkdir -p "${OUTPUT_DIR}"

# Loop through all model paths
for DATATYPE in "${DATATYPES[@]}"; do
    for MODEL_PATH in "${MODEL_PATHS[@]}"; do
        # Set output path for this specific model and datatype
        MODEL_NAME=$(basename "${MODEL_PATH}")
        OUTPUT_PATH="${OUTPUT_DIR}/${MODEL_NAME}_${DATATYPE}"

        # Check if output path already exists
        if [ -d "${OUTPUT_PATH}" ]; then
            echo "Output path ${OUTPUT_PATH} already exists, skipping..."
            continue
        fi

        # Echo the values for verification
        echo "Model Path: ${MODEL_PATH}"
        echo "Dataset: ${DATATYPE}"
        echo "Output Directory: ${OUTPUT_PATH}"
        echo "Number of Passes: ${N_PASSES}"
        echo "Max Response Length: ${MAX_LENGTH}"
        echo "Tensor Parallel Size: ${TP_SIZE}"

        # Run the evaluation
        python3 -m verl.trainer.main_generation \
            trainer.nnodes=1 \
            trainer.n_gpus_per_node=4 \
            data.path=data/${DATATYPE}.parquet \
            data.output_path=${OUTPUT_PATH}/dataset_${DATATYPE}.parquet \
            data.temp_output_path=${OUTPUT_PATH}/dataset_${DATATYPE}_temp.json \
            data.n_samples=${N_PASSES} \
            data.batch_size=1024 \
            model.path=${MODEL_PATH} \
            rollout.temperature=0.6 \
            rollout.response_length=${MAX_LENGTH} \
            rollout.prompt_length=1024 \
            rollout.max_num_batched_tokens=64000 \
            rollout.top_k=-1 \
            rollout.top_p=0.95 \
            rollout.gpu_memory_utilization=0.85 \
            rollout.tensor_model_parallel_size=${TP_SIZE}
        if [ -f "${OUTPUT_PATH}/pass.csv" ]; then
            python3 -m merge "${OUTPUT_PATH}/pass.csv" "${OUTPUT_DIR}/merge_result.csv"
        else
            echo "Warning: pass.csv not found at ${OUTPUT_PATH}, skipping merge."
        fi
    done
done