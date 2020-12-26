# インストール手順
1. Python 3.7.6 64bit
1. MeCab v0.996
1. CUDA 10.2
1. pipenv

## MeCab
64 bit 版 Python を使うため、あらかじめ 64 bit 版 MeCab をインストールする必要がある。
- Windows  
以下の URL からダウンロードする。  
https://github.com/ikegami-yukino/mecab/releases/tag/v0.996
- Linux - apt  
以下のコマンドを実行する。  
`sudo apt install mecab`
- Linux - コンパイル  
以下の URL からソースコードをダウンロードしコンパイルする。  
https://taku910.github.io/mecab/

## CUDA
- CUDA 10.2

# 使い方
## スクリプト実行例
- Example: Cart Pole の Q 学習
    ```
    python examples/train_cart_pole.py \
        --alpha 0.1 \
        --discount 0.99 \
        --render
    ```

- Example: Gensim の Word2Vec モデル
    ```
    python examples/train_gensim_w2v.py \
        --input_dir data/original/wikipedia_ja \
        --cache_dir data/cache/wikipedia_ja \
        --model_name_to_save data/model/gensim_w2v.bin \
        --window 5 \
        --size 100 \
        --negative 5 \
        --ns_exponent 0.75 \
        --min_count 5 \
        --alpha 0.025 \
        --min_alpha 0.0001 \
        --epochs 20 \
        --workers 4 \
        --seed 0
    ```

- Example: 自作 Word2Vec モデル
    ```
    python examples/train_my_w2v.py \
        --input_dir data/original/wikipedia_ja \
        --cache_dir data/cache/wikipedia_ja \
        --model_name_to_save data/model/my_w2v.bin \
        --window 5 \
        --size 100 \
        --negative 5 \
        --ns_exponent 0.75 \
        --min_count 5 \
        --alpha 0.025 \
        --min_alpha 0.0001 \
        --epochs 20 \
        --mb_size 512 \
        --workers 1 \
        --seed 0
    ```

- Example: データセットの作成
    ```
    python examples/preprocess/bilingual_data_set.py \
        "--input_dir" "data/original/japaneseenglish-bilingual-corpus/wiki_corpus_2.01" \
        "--cache_dir" "data/cache/japaneseenglish-bilingual-corpus"
    ```

- Example: Encoder-Decoder モデルの学習
    ```
    python "examples/train_encoder_decoder.py" \
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
    ```

- Example: Cifar 10 画像分類モデルの学習
    ```
    python "examples/train_cifar10_classifier.py" \
        "--train_image_npy_path" "data/preprocess/CIFAR-10/train_image.npy" \
        "--train_label_npy_path" "data/preprocess/CIFAR-10/train_label.npy" \
        "--test_image_npy_path" "data/preprocess/CIFAR-10/test_image.npy" \
        "--test_label_npy_path" "data/preprocess/CIFAR-10/test_label.npy" \
        "--model" "res_net" \
        "--output_dir_format" "data/model/cifar10/{date}" \
        "--model_name_format" "epoch-{epoch}.h5" \
        "--optimizer" "sgd" \
        "-lr" "1e-1" \
        "--momentum" "0.7" "--nesterov" \
        "--lr_scheduler" "cosine_annealing" \
        "--lr_decay" "0.1" \
        "--lr_steps" "0.1" "0.5" "0.75" "0.9" \
        "--min_lr" "1e-5" \
        "--mb_size" "256" \
        "--epochs" "100" \
        "--seed" "2"
    ```
