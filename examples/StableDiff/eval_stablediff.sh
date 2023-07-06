MODEL_DIR=CompVis/stable-diffusion-v1-4
CLIP_MODEL=openai/clip-vit-base-patch16

python -u eval_stablediff.py \
    --model-name-or-path $MODEL_DIR \
    --clip-model-name-or-path $CLIP_MODEL \
    --dataset-name-or-path conceptual_captions \
    --pipeline-module-parts vae unet text_encoder \
    --num-images-to-save 5 \
    --output-dir ./sd_images \
    --do_FP8_eval
