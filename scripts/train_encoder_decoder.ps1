$curr_dir = ($pwd).Path
cd I:`Yuki`workspace`mltools

pipenv run python "examples/train_encoder_decoder.py" `
    "--train_data" "data/cache/japaneseenglish-bilingual-corpus/BDS/train" `
    "--valid_data" "data/cache/japaneseenglish-bilingual-corpus/BDS/valid" `
    "--model" "global_attention" `
    "--output_dir_format" "data/model/encoder_decoder/{date}" `
    "--model_name_format" "epoch-{epoch}.h5" `
    "--optimizer" "sgd" "-lr" "1e-1" "--momentum" "0.7" "--nesterov" "--clipnorm" "1e-1" `
    "--lr_decay_rate" "0.31" `
    "--lr_decay_epochs" "3" "6" "51" "52" "76" "77" "91" "92" `
    "--epochs" "100" `
    "--mb_size" "16" `
    "--seed" "2"

cd $curr_dir