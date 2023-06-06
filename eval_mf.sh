MODEL_DIR=facebook/maskformer-swin-base-ade
#MODEL_DIR=facebook/maskformer-swin-large-ade

python eval_mf.py \
    --model_name $MODEL_DIR \
    --batch_size 1 \
    --do_FP8_eval