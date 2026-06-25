export CUDA_VISIBLE_DEVICES=0,1  # 使用的GPU编号
NUM_GPUS=2  # GPU数量

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29501 \
    eval/eval_nonstreaming_conv.py \
    --annotation_dir /ctd/Works/m-wu/Datasets/zoom2025/finetune_labels/l3_conv_test_with_backchannel \
    --audio_root_a /ctd/Works/m-wu/Datasets/zoom2025/audios/A_gd \
    --audio_root_b /ctd/Works/m-wu/Datasets/zoom2025/audios/B_gd \
    --pretrained_model_name_or_path Qwen/Qwen2-Audio-7B-Instruct \
    --lora_path /ctd/Works/m-wu/Models/StreamingSpeechLLM/ASR_CONV_finetune/qwen2audio_l5_lora16 \
    --output_path nonstreaming_results/qwen2audio_l5_lora16.jsonl \
    --max_new_tokens 512