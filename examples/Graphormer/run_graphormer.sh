MODEL_DIR=lnair/graphormer-ogbg-molhiv

python -u eval_graphormer.py \
    --model-name-or-path $MODEL_DIR \
    --dataset-name-or-path OGB/ogbg-molhiv \
    --eval-in-fp8