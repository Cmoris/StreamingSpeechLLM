#!/bin/bash

# ================================================================
# 多卡训练脚本
# 使用 DeepSpeed 或 torchrun 进行分布式训练
# ================================================================

# 设置基础参数
export CUDA_VISIBLE_DEVICES=0,1,2,3  # 使用的GPU编号
NUM_GPUS=4  # GPU数量

# 模型和数据路径
MODEL_PATH="Qwen/Qwen2.5-Omni-7B"
ASR_DATA_PATH="/n/work6/yizhang/Moris/zoom2025/pretrain_labels/l4_conv_train"
ASR_AUDIO_FOLDER_A="/n/work6/yizhang/Moris/zoom2025/audios/A_all"
ASR_AUDIO_FOLDER_B="/n/work6/yizhang/Moris/zoom2025/audios/B_all"

DATA_VERSION=0

# 训练超参数
BATCH_SIZE=1  # 每张卡的batch size
GRAD_ACCUM=8 # 梯度累积步数
LEARNING_RATE=2e-5
NUM_EPOCHS=4
MAX_LENGTH=2048

CHUNK_SECS=1

# LoRA参数
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05
TARGET_MODULES="q_proj,k_proj,v_proj,o_proj"

# 保存和日志
SAVE_STEPS=500
LOGGING_STEPS=10
SAVE_TOTAL_LIMIT=5
OUTPUT_DIR=/n/work6/yizhang/Moris/Models/StreamingSpeechLLM/ASR/qwen2_5Omni_asrl4_chunk${CHUNK_SECS}s_lora16


torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    train.py \
    --pretrained_model_name_or_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --data_version $DATA_VERSION \
    --annotation_dir $ASR_DATA_PATH \
    --audio_root_a $ASR_AUDIO_FOLDER_A \
    --audio_root_b $ASR_AUDIO_FOLDER_B\
    --chunk_secs $CHUNK_SECS \
    --dataloader_num_workers 8 \
    --model_max_length $MAX_LENGTH \
    --lora_enable True \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --target_modules $TARGET_MODULES \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate $LEARNING_RATE \
    --lr_scheduler_type "cosine" \
    --freeze_modules audio_tower visual \
    --warmup_step 1000 \
    --weight_decay 0 \
    --bits 16 \
    --bf16 True \
    --logging_steps $LOGGING_STEPS \
    --save_strategy "steps" \
    --save_steps $SAVE_STEPS \
    --save_total_limit $SAVE_TOTAL_LIMIT \
    --gradient_checkpointing True \
    --ddp_find_unused_parameters False \
    --report_to "tensorboard" \
    --overwrite_output_dir True