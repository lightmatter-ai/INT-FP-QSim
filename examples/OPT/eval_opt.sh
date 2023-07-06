MODEL_NAME=lnair/opt-350m-wikitext2
#MODEL_NAME=lnair/opt-1.3b-wikitext2
#MODEL_NAME=lnair/opt-2.7b-wikitext2

python -u eval_opt.py \
    --model_name_or_path $MODEL_NAME \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 2 \
    --seed 42 \
    --output_dir ./tmp/test-clm \
    --do_eval \
    --do_FP8_eval
    #--block_size 256