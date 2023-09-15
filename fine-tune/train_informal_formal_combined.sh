#!/bin/bash
export WANDB_PROJECT=st_ICHF_combined
export WANDB_RUN_ID=alpaca_combo
export  TRANSFORMERS_CACHE="/mnt/swordfish-pool2/models/transformers_cache"
MODEL_NAME_OR_PATH="/mnt/swordfish-pool2/models/alpaca/7B"
OUTPUT_DIR="/mnt/swordfish-pool2/asaakyan/style-transfer/alpaca-combo"

torchrun --nproc_per_node=4 --master_port=12346 train.py \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --data_path ../data/gyafc_w_ICHF_alpaca/combined_train.json \
    --fp16 True \
    --tf32 True \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer'