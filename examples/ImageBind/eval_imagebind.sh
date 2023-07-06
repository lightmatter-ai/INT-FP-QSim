python -u eval_imagebind.py \
    --eval-batch-size 32 \
    --data-dir /data/datasets/imagenet \
    --load-checkpoint /data/checkpoints/imagebind_huge.pth \
    --seed 42 \
    --do-FP8-eval