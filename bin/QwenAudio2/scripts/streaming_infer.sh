export CUDA_VISIBLE_DEVICES=4  # 使用的GPU编号
NUM_GPUS=1  # GPU数量

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    streaming_infer.py \
    --annotation_dir /n/work6/yizhang/Moris/zoom2025/finetune_labels/l3_conv_test_with_backchannel \
    --audio_root_a /n/work6/yizhang/Moris/zoom2025/audios/A_gd \
    --audio_root_b /n/work6/yizhang/Moris/zoom2025/audios/B_gd \
    --initial_chunk_sec 0 \
    --streaming_chunk_sec 4 \
    --pretrained_model_name_or_path Qwen/Qwen2-Audio-7B-Instruct \
    --model_path /n/work6/yizhang/Moris/Models/StreamingSpeechLLM/ASR_CONV_pre/qwen2audio_pretrainl3_bc_lora16/checkpoint-8500 \
    --output_dir /home/yizhang/Moris/StreamingSpeechLLM/bin/QwenAudio2/result_nonstreaming/qwen2audio_asrl2_chunk2s/