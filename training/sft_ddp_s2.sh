# UPDATE: Replace placeholder paths with your actual paths.
#!/usr/bin/env bash
set -euo pipefail
# # GPU binding: Slurm uses allocated GPUs; local runs use 6,7
# if [[ -n "${SLURM_JOB_ID:-}" ]]; then
#   [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]] && { echo "[ERROR] SLURM_JOB_ID set but CUDA_VISIBLE_DEVICES empty." 1>&2; exit 1; }
# else
#   export CUDA_VISIBLE_DEVICES="6,7"
# fi
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG=TRACE
NPROC_PER_NODE=$RESOURCE_GPU
nnodes=$WORLD_SIZE
NNODES=$WORLD_SIZE
NODE_RANK=$RANK
MASTER_ADDR=$MASTER_ADDR
MASTER_PORT=$MASTER_PORT
export VGGT_DTYPE="${VGGT_DTYPE:-bfloat16}"
export SWIFT_SINGLE_DEVICE_MODE=1
export DISABLE_MP_DDP=1

DATA_STAGE2="
data/zy_utils/lingoqa_action_train_qwen_v0.json#150000
data/zy_utils/merged_surds_vqas_v0.jsonl#25000
data/preprocess_data/MapLMv2/maplm_v2_train_ms_swift.jsonl#35000
data/preprocess_data/MapLMv2/maplm_v2_train_ms_swift_multi.jsonl#5000
data/Filter/RefCOCO/raw_data/refcoco_train_qwen_v1.jsonl#10000
data/Filter/Flickr/filter_data/flickr_filter_06.jsonl#18000
data/Filter/VQAv2/filter_data/vqa_filter_06.jsonl#90000
data/Filter/Gqa/filter_data/gqa_filter_06.jsonl#95000
data/zy_utils/visual_trace_rel_qwen_v0.jsonl#1200
data/zy_utils/where2place_train_rel_qwen_v0.jsonl#100
data/workflow_pipeline/OpenSpaces/dataset/openspaces_train.jsonl#3500
data/zy_utils/convert_tools/SPAR-7m/scannet_merged_data_converted.jsonl#1100000
data/zy_utils/convert_tools/SPAR-7m/structured3d_merged_data_converted.jsonl#450000
data/Filter/Drivevqa/filter_data/drivevqa_rel_filter_06.jsonl#30000
data/Filter/DriveLMMo1/filter_data/filtered_raw_data_drivelmm_06_v2.jsonl#20000
data/Filter/Omnidrive/filter_data/omnidrive_vqa_filtered_06.jsonl#15000
data/Filter/Omnidrive/filter_data/omnidrive_vqa_filtered_multi_06.jsonl#5000
data/zy_utils/bdd100k_labels_images_train_weather_qwen_v1.json#50000
data/preprocess_data/vsi-590k/train_images.jsonl
data/preprocess_data/vsi-590k/train_videos.jsonl
"

swift sft \
    --model checkpoints/fm_qwen3_vl_moe_vggt_xattnv2_s1/v2-20260404-233757/checkpoint-8762/ \
    --custom_register_path plugins/qwen3_vl_moe_vggt_register.py \
    --model_type qwen3_vl_moe_vggt \
    --dataset $DATA_STAGE2 \
    --train_type lora \
    --target_modules 'all-linear' \
    --freeze_llm false \
    --freeze_vit true \
    --freeze_aligner true \
    --modules_to_save 'connector' \
    --lora_rank 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --gradient_checkpointing true \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 8e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --output_dir results/sft_checkpoint/fm_qwen3_vl_moe_vggt_xattnv2_s2 \
    --logging_steps 2 \
    --save_steps 500 \
    --eval_steps 1000 \
    --save_total_limit 3 \
    --max_length 15000 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --dataloader_pin_memory true \
    --deepspeed zero2 \
    --split_dataset_ratio 0.0 \
    --run_name stage2_lora_aligner_llm_2M \
    --save_safetensors true \
    2>&1 | tee logs/post_training_s2/train_stage2.log
