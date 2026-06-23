python \
    alignment/generate_alignment.py \
    --annotation-dir /n/work6/yizhang/Moris/zoom2025/audios/B_gd \
    --batch-size 1 \
    --target-sr 16000 \
    --language ja \
    --device cuda:1 \
    --output-dir ./results \