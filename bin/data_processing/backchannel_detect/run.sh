export CUDA_VISIBLE_DEVICES=0,1,2,3  # 使用的GPU编号
NUM_GPUS=4  # GPU数量
SPLIT="train"

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29501 \
    back_channel_detect_llm.py \
    --model_name Qwen/Qwen3-8B \
    --annotation_dir /n/work6/yizhang/Moris/zoom2025/finetune_labels/${SPLIT} \
    --output_dir /home/yizhang/Moris/StreamingSpeechLLM/bin/data_processing/backchannel_detect/finetune_${SPLIT} \
    --barch_size 8