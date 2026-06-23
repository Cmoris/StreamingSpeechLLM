#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5  # 使用的GPU编号
GPUS_PER_NODE=6
MASTER_PORT=29502

MODEL_PATHS=(
    "Qwen/Qwen3-0.6B"
    "Qwen/Qwen3-1.7B"
)

AUDIO_TOWERS=(
        "reazon-research/japanese-zipformer-base-k2"
        "reazon-research/japanese-hubert-base-k2"
        "openai/whisper-large-v3"
    )

ANNOTATION_DIR="/home/muyun/project/Streaming_ASR_VAP/bin/data_processing/l2"
AUDIO_ROOT_A="/mnt/nvme/workspaces/muyun/Dataset/zoom2025/audios/A_all"
AUDIO_ROOT_B="/mnt/nvme/workspaces/muyun/Dataset/zoom2025/audios/B_all"

# 训练超参数
BATCH_SIZE=1  # 每张卡的batch size
GRAD_ACCUM=8  # 梯度累积步数
LEARNING_RATE=2e-5
NUM_EPOCHS=4
MAX_LENGTH=2048

# LoRA参数
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05
TARGET_MODULES="q_proj,k_proj,v_proj,o_proj"

SAVE_STEPS=5000
LOGGING_STEPS=10
SAVE_TOTAL_LIMIT=10

for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    for AUDIO_TOWER_PATH in "${AUDIO_TOWERS[@]}"; do
        TOWER_NAME=$(basename ${AUDIO_TOWER_PATH})
        MODEL_NAME=$(basename ${MODEL_PATH})
        OUTPUT_DIR=/mnt/nvme/workspaces/muyun/Models/ASR_TS/base_${MODEL_NAME}_${TOWER_NAME}_c2s_chunk0.5s_lora16_stage2

        torchrun \
            --nproc_per_node=${GPUS_PER_NODE} \
            --master_port=${MASTER_PORT} \
            train.py \
            --model_type SpeechQwen3 \
            --model_path ${MODEL_PATH} \
            --audio_tower ${AUDIO_TOWER_PATH} \
            --freeze_backbone False \
            --tune_mm_mlp_adapter False \
            --pretrain_mm_mlp_adapter \
            --mm_projector_a_type conv \
            --pretrain_mm_mlp_adapter \
            --pretrain_qformer  \
            --use_qformer True \
            --qformer_model bert-large-uncased \
            --qformer_layers 2 \
            --qformer_dim 1024 \
            --annotation_dir ${ANNOTATION_DIR} \
            --audio_root_a ${AUDIO_ROOT_A} \
            --audio_root_b ${AUDIO_ROOT_B} \
            --output_dir ${OUTPUT_DIR} \
            --lora_enable True \
            --lora_r $LORA_R \
            --lora_alpha $LORA_ALPHA \
            --lora_dropout $LORA_DROPOUT \
            --target_modules $TARGET_MODULES \
            --num_train_epochs $NUM_EPOCHS \
            --per_device_train_batch_size $BATCH_SIZE \
            --gradient_accumulation_steps $GRAD_ACCUM \
            --learning_rate 2e-5 \
            --lr_scheduler_type cosine \
            --warmup_ratio 0.03 \
            --logging_steps $LOGGING_STEPS \
            --save_strategy "steps" \
            --save_steps $SAVE_STEPS \
            --save_total_limit $SAVE_TOTAL_LIMIT \
            --bits 16 \
            --bf16 True \
            --gradient_checkpointing True \
            --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
            --dataloader_num_workers 8 \
            --report_to tensorboard 
    done
done