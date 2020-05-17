cd ..

pipenv run python "examples/train_encoder_decoder.py" \
    "--train_data" "data/original/small_parallel_enja/train.ja" "data/original/small_parallel_enja/train.en" \
    "--valid_data" "data/original/small_parallel_enja/dev.ja" "data/original/small_parallel_enja/dev.en" \
    "--model" "global_attention" \
    "--lang" "en_to_ja" \
    "--output_dir_format" "data/model/encoder_decoder/{date}" \
    "--model_name_format" "epoch-{epoch}.h5" \
    "--optimizer" "sgd" \
    "-lr" "1e-1" \
    "--momentum" "0.7" "--nesterov" \
    "--lr_scheduler" "cosine_annealing" \
    "--lr_decay" "0.1" \
    "--lr_steps" "0.1" "0.5" "0.75" "0.9" \
    "--min_lr" "1e-5" \
    "--epochs" "100" \
    "--mb_size" "16" \
    "--seed" "2"

cd scripts
