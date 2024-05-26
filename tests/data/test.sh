mkdir -p logs

LOG_FILE=logs/test.log

xtrainers-preprocess \
    --task clm \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --model_name_or_path openai-community/gpt2 \
    --overwrite_cache \
    --output_name  wikitext2_gpt2 \
    --block_size 1024  2>&1 | tee $LOG_FILE
