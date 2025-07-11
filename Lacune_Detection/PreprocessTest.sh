#!/bin/bash

# Initialize empty variables
INPUT_DIR=""
OUTPUT_DIR=""

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --input_dir) INPUT_DIR="$2"; shift ;;
        --output_dir) OUTPUT_DIR="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "REceived"
# Check required arguments
if [[ -z "$INPUT_DIR" || -z "$OUTPUT_DIR" ]]; then
    echo "Usage: bash PreprocessTest.sh --input_dir <INPUT_DIR> --output_dir <OUTPUT_DIR>"
    exit 1
fi
echo "INPUT_DIR: $INPUT_DIR"
echo "OUTPUT_DIR: $OUTPUT_DIR"
# Create output directory if not exists
mkdir -p "$OUTPUT_DIR"

# Process all FLAIR images
find "$INPUT_DIR" -name "*FLAIR.nii.gz" | while read -r flair; do
    base=$(basename "$flair" .nii.gz)
    parent=$(dirname "$flair")
    bet_output="${parent}/${base}_brain"
    restore_output="${OUTPUT_DIR}/${base}_preprocessed.nii.gz"
    echo "Processing: $flair"

    # Step 1: Brain extraction
    bet "$flair" "$bet_output" -f 0.5 
    # Step 2: Bias field correction
    fast -B -n 2 "${bet_output}.nii.gz"

    base_no_ext="${bet_output%.nii.gz}"  # remove .nii.gz extension if present
    rm -f "${base_no_ext}_pveseg.nii.gz" "${base_no_ext}_seg.nii.gz"\
      "${base_no_ext}_pve_0.nii.gz" "${base_no_ext}_pve_1.nii.gz" "${base_no_ext}_pve_2.nii.gz" \
      "${base_no_ext}_mixeltype.nii.gz" "${base_no_ext}_bias.nii.gz"

    # Step 3: Move the restored image to output
    if [[ -f "${bet_output}_restore.nii.gz" ]]; then
        mv "${bet_output}_restore.nii.gz" "$restore_output"
        echo "Saved: $restore_output"
    else
        echo "Warning: Bias-corrected image not found for $flair"
    fi
done