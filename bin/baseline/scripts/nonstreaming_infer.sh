export CUDA_VISIBLE_DEVICES=4  # 使用的GPU编号
NUM_GPUS=1  # GPU数量

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29501 \
    nonstreaming_infer.py \
    --annotation_dir /n/work6/yizhang/Moris/zoom2025/pretrain_labels/l3_conv_test_with_backchannel \
    --audio_root_a /n/work6/yizhang/Moris/zoom2025/audios/A_all \
    --audio_root_b /n/work6/yizhang/Moris/zoom2025/audios/B_all \
    --model_base Qwen/Qwen3-1.7B \
    --model_path /n/work6/yizhang/Moris/Models/StreamingSpeechLLM/ASR_CONV_pre/base_Qwen3-1.7B_japanese-zipformer-base-k2_stage1/checkpoint-10360 \
    --output_dir /home/yizhang/Moris/StreamingSpeechLLM/bin/baseline/results/base_Qwen3-1.7B_japanese-zipformer-base-k2_stage1