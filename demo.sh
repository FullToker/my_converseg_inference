#!/bin/bash
# demo.sh — run ConverSeg inference
#
# Usage:
#   bash demo.sh                          # batch on default ./images folder (model loads once)
#   bash demo.sh --dir /path/to/folder    # batch on custom folder
#   bash demo.sh --single /path/img.jpg   # single image via demo.py

# ---- Checkpoint & model config ----
FINAL_CKPT="./checkpoints/ConverSeg-Net-3B/ConverSeg-Net_sam2_90000.torch"
PLM_CKPT="./checkpoints/ConverSeg-Net-3B/ConverSeg-Net_plm_90000.torch"
LORA_CKPT="./checkpoints/ConverSeg-Net-3B/lora_plm_adapter_90000"
MODEL_CFG="sam2_hiera_l.yaml"
BASE_CKPT="./checkpoints/sam2_hiera_large.pt"

# ---- Inference settings ----
PROMPT="thing can be used to heat water"
DEVICE="cuda"
OUT_DIR="./demo_outputs"

# ---- Parse arguments ----
MODE="batch"          # default: batch folder mode
IMAGE_DIR="./images"  # default input folder
SINGLE_IMAGE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dir)
            IMAGE_DIR="$2"; shift 2 ;;
        --single)
            MODE="single"; SINGLE_IMAGE="$2"; shift 2 ;;
        --prompt)
            PROMPT="$2"; shift 2 ;;
        --out_dir)
            OUT_DIR="$2"; shift 2 ;;
        --device)
            DEVICE="$2"; shift 2 ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: bash demo.sh [--dir FOLDER] [--single IMAGE] [--prompt TEXT] [--out_dir DIR] [--device cuda|cpu]"
            exit 1 ;;
    esac
done

# ---- Run ----
if [[ "$MODE" == "single" ]]; then
    echo "Mode: single image -> demo.py"
    echo "Image:  $SINGLE_IMAGE"
    echo "Prompt: $PROMPT"
    echo "Output: $OUT_DIR"
    echo "----------------------------------------"
    python demo.py \
        --final_ckpt "$FINAL_CKPT" \
        --plm_ckpt   "$PLM_CKPT" \
        --lora_ckpt  "$LORA_CKPT" \
        --model_cfg  "$MODEL_CFG" \
        --base_ckpt  "$BASE_CKPT" \
        --image      "$SINGLE_IMAGE" \
        --prompt     "$PROMPT" \
        --device     "$DEVICE" \
        --out_dir    "$OUT_DIR"
else
    echo "Mode: batch folder -> demo_batch.py"
    echo "Folder: $IMAGE_DIR"
    echo "Prompt: $PROMPT"
    echo "Output: $OUT_DIR"
    echo "----------------------------------------"
    python demo_batch.py \
        --final_ckpt "$FINAL_CKPT" \
        --plm_ckpt   "$PLM_CKPT" \
        --lora_ckpt  "$LORA_CKPT" \
        --model_cfg  "$MODEL_CFG" \
        --base_ckpt  "$BASE_CKPT" \
        --image_dir  "$IMAGE_DIR" \
        --prompt     "$PROMPT" \
        --device     "$DEVICE" \
        --out_dir    "$OUT_DIR"
fi
