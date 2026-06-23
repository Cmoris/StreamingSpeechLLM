#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # 使用的GPU编号
export TORCH_DISTRIBUTED_DEBUG=DETAIL
GPUS_PER_NODE=8
MASTER_PORT=29500

# Model
MODEL_PATHS=(
    "Qwen/Qwen3-1.7B"
)
AUDIO_TOWER="reazon-research/japanese-zipformer-base-k2"

# Dataset
ANNOTATION_DIR="/n/work6/yizhang/Moris/reazonspeech/shards_train"
DATA_VERSION="single_channel_asr"

# 训练超参数
BATCH_SIZE=16  # 每张卡的batch size
GRAD_ACCUM=4  # 梯度累积步数
LEARNING_RATE=2e-5
NUM_EPOCHS=1

# LoRA参数
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05
TARGET_MODULES="q_proj,k_proj,v_proj,o_proj"

SAVE_STEPS=1000
LOGGING_STEPS=10
SAVE_TOTAL_LIMIT=10

for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    TOWER_NAME=$(basename ${AUDIO_TOWER})
    MODEL_NAME=$(basename ${MODEL_PATH})
    OUTPUT_DIR=/n/work6/yizhang/Moris/Models/StreamingSpeechLLM/ASR/base_${MODEL_NAME}_${TOWER_NAME}_stage1

    torchrun \
        --nproc_per_node=${GPUS_PER_NODE} \
        --master_port=${MASTER_PORT} \
        train.py \
        --model_type SpeechQwen3 \
        --model_path $MODEL_PATH \
        --audio_tower $AUDIO_TOWER \
        --freeze_backbone False \
        --tune_mm_mlp_adapter True \
        --mm_projector_a_type conv \
        --use_qformer True \
        --tune_qformer True \
        --qformer_model bert-large-uncased \
        --qformer_layers 2 \
        --qformer_dim 1024 \
        --queries_per_sec 8 \
        --audio_frame_rate 25 \
        --annotation_dir $ANNOTATION_DIR \
        --data_version $DATA_VERSION \
        --output_dir $OUTPUT_DIR \
        --num_train_epochs $NUM_EPOCHS \
        --per_device_train_batch_size $BATCH_SIZE \
        --gradient_accumulation_steps $GRAD_ACCUM \
        --learning_rate $LEARNING_RATE \
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