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
swift sft \
  --model checkpoints/fm_qwen3_vl_moe_vggt_xattn_full_s3_CL3/v0-20260228-104121/checkpoint-448-merged \
  --model_type qwen3_vl_moe_vggt \
  --custom_register_path plugins/qwen3_vl_moe_vggt_register.py \
  --dataset \
    "data/zy_utils/lingoqa_scenery_train_qwen_v0.json#5000" \
    "data/zy_utils/lingoqa_action_train_qwen_v0.json#5000" \
    "data/zy_utils/merged_surds_vqas_rel_qwen_v0.jsonl#10000" \
    "data/Filter/Drivevqa/filter_data/drivevqa_rel_filter_06.jsonl#2000" \
    "data/Filter/RefCOCO/raw_data/refcoco_train_qwen_v1.jsonl#5000" \
    "data/Filter/VQAv2/filter_data/vqa_filter_06.jsonl#10000" \
    "data/Filter/Gqa/filter_data/gqa_filter_06.jsonl#10000" \
    "data/zy_utils/bdd100k_labels_images_train_weather_qwen_v1.json#5000" \
    "data/Filter/Roborefit/filter_data/roborefit_filter_06.jsonl#5000" \
    "data/Filter/Cosmosr1/filter_data/cosmosr1_filter_06.jsonl#1000" \
    "data/zy_utils/vladbench_train_image_rel_qwen_v1.json#5000" \
    "data/zy_utils/robovqa_understanding.jsonl#10000" \
    "data/zy_utils/visual_trace_rel_qwen_v0.jsonl#1000" \
    "data/zy_utils/embodied-r1_train_rel_qwen_v0.jsonl#1000" \
    "data/zy_utils/part_affordance_train_rel_qwen_v0.jsonl#1000" \
    "data/zy_utils/RoboAfford_train_rel_qwen_v0.jsonl#1000" \
    "data/zy_utils/vabench_bbox_no_reasoning_train_rel_qwen_v0.jsonl#1000" \
    "data/zy_utils/where2place_train_rel_qwen_v0.jsonl#1000" \
    "data/Filter/DriveLMMo1/filter_data/filtered_raw_data_drivelmm_06_v2.jsonl#5000" \
    "data/Filter/MapLM/filter_data/maplm_filter_06.jsonl#10000" \
    "data/all_eval_data/ego3d-bench/ego3d_bench_full.jsonl#3000" \
    "data/all_eval_data/strideqa/strideqa_bench_t0_bench_image.jsonl#500" \
    "data/all_eval_data/strideqa/strideqa_bench_t0_bench_video.jsonl#500" \
    "data/all_eval_data/strideqa/strideqa_bench_t1_bench_image.jsonl#500" \
    "data/all_eval_data/strideqa/strideqa_bench_t1_bench_image.jsonl#500" \
    "data/all_eval_data/strideqa/strideqa_bench_t2_bench_image.jsonl#500" \
    "data/all_eval_data/strideqa/strideqa_bench_t2_bench_video.jsonl#500" \
    "data/all_eval_data/strideqa/strideqa_bench_t3_bench_image.jsonl#500" \
    "data/all_eval_data/strideqa/strideqa_bench_t3_bench_video.jsonl#500" \
  --torch_dtype bfloat16 \
  --remove_unused_columns true \
  --learning_rate 1e-4 \
  --train_type lora \
  --lora_rank 16 \
  --lora_alpha 32 \
  --freeze_vit true \
  --freeze_llm false \
  --freeze_aligner false \
  --modules_to_save connector \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --logging_steps 2 \
  --save_steps 20 \
  --eval_steps 50 \
  --save_total_limit 5 \
  --max_length 32096 \
  --output_dir results/sft_checkpoint/fm_qwen3_vl_moe_vggt_xattn_full_s4_mix \
  --run_name verify_image_tchw \
  --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
  --deepspeed zero2 \
  --use_logits_to_keep true \
  --num_train_epochs 1 \
  --warmup_ratio 0.05 \
  --dataloader_num_workers 4 \
  --dataset_num_proc 4

  # "data/Filter/MapLM/filter_data/maplm_filter_06.jsonl#1000" \
