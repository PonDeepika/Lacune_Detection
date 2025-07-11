#!/bin/bash

# Default values
DEVICE=""
SAM_CKPT=""
ROOT_TEST=""
MODEL_PATH=""
SAM_OUTPUT_PATH=""
MARS_ATLAS=""
REGISTERED_MARS_FILES=""

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --device) DEVICE="$2"; shift ;;
        --sam_path) SAM_PATH="$2"; shift ;;
        --sam_checkpoint) SAM_CKPT="$2"; shift ;;
        --root_test) ROOT_TEST="$2"; shift ;;
        --model_path) MODEL_PATH="$2"; shift ;;
        --CPGSAM_output_path) CPGSAM_OUTPUT_PATH="$2"; shift ;;
        --ART_output_path) ART_OUTPUT_PATH="$2"; shift ;;
        --Registered_MARS_files) REGISTERED_MARS_FILES="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# export PYTHONPATH="/mnt/2dbe8447-21b5-4374-a5eb-ada69dcbb95a/Lacune_Detection:$PYTHONPATH"
export PYTHONPATH="$SAM_PATH:$PYTHONPATH"

# Check mandatory args for CPG+SAM
if [[ -z "$DEVICE" || -z "$SAM_CKPT" || -z "$ROOT_TEST" || -z "$MODEL_PATH" || -z "$CPGSAM_OUTPUT_PATH" ]]; then
    echo "Missing arguments for CPG_SAM_Inference.py"
    exit 1
fi


# ----------------------------
# Run CPG + SAM Inference
# ----------------------------
echo "Running CPG + SAM Inference..."
python CPG_SAM_inference.py \
    --device "$DEVICE" \
    --sam_checkpoint "$SAM_CKPT" \
    --root_test "$ROOT_TEST" \
    --model_path "$MODEL_PATH" \
    --CPGSAM_output_path "$CPGSAM_OUTPUT_PATH"

# ----------------------------
# Run ART Refinement
# ----------------------------

# Check mandatory args for ART
if [[ -z "$MARS_ATLAS" || -z "$CPGSAM_OUTPUT_PATH" || -z "$REGISTERED_MARS_FILES" || -z "$ART_OUTPUT_PATH" ]]; then
    echo "Missing arguments for ART_Inference.py"
    exit 1
fi

echo "Running ART Inference..."
python ART_module.py \
    --root_test "$ROOT_TEST" \
    --CPGSAM_output_path "$CPGSAM_OUTPUT_PATH" \
    --Registered_MARS_files "$REGISTERED_MARS_FILES" \
    --ART_output_path "$ART_OUTPUT_PATH"

echo "Done!"