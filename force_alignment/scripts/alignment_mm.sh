export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPUS=8

torchrun --nproc_per_node=$NUM_GPUS \
  --master_port=29502 \
  alignment/generate_alignment_mm.py \
  --annotation-dir /n/work6/yizhang/Moris/zoom2025/asr_labels \
  --language ja \
  --batch-size 8 \
  --output-dir ./results_gd