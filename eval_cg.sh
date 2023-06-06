MODEL_DIR=Salesforce/codegen-2B-mono
#MODEL_DIR=Salesforce/codegen-6B-mono

python -u eval_cg.py \
    --model-name-or-path $MODEL_DIR \
    --dataset-name-or-path openai_humaneval \
    --do_FP8_eval