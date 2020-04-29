# インストール手順
- Python 3.7.6 64bit
- Mecab v0.996

## MeCab
64 bit 版 Python を使うには、あらかじめ 64 bit 版 MeCab をインストールする必要がある。
- Windows  
以下の URL からダウンロードする。  
https://github.com/ikegami-yukino/mecab/releases/tag/v0.996
- Linux - apt  
以下のコマンドを実行する。  
`sudo apt install mecab`
- Linux - コンパイル  
以下の URL からソースコードをダウンロードしコンパイルする。  
https://taku910.github.io/mecab/

# 使い方
## 実行方法
- Seq2Seq モデル
    1. japaneseenglish-bilingual-corpus.zip を解凍する。
    1. bilingual_data_set.py を実行し、解凍データからトークナイズ済みのデータを生成する。
    1. run_encoder_decoder.py を実行し、トークナイズ済みのデータから Seq2Seq モデルを学習する。

## スクリプト
- Example: run_gensim_w2v.py
    ```
    python examples/run_gensim_w2v.py \
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

- Example: run_my_w2v.py
    ```
    python examples/run_my_w2v.py \
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

- Example: bilingual_data_set.py
    ```
    python examples/preprocess/bilingual_data_set.py \
        "--input_dir" "data/original/japaneseenglish-bilingual-corpus/wiki_corpus_2.01" \
        "--cache_dir" "data/cache/japaneseenglish-bilingual-corpus"
    ```

- Example: train_encoder_decoder.py
    ```
    python examples/train_encoder_decoder.py \
        "--train_data" "data/cache/japaneseenglish-bilingual-corpus/BDS/train" \
        "--valid_data" "data/cache/japaneseenglish-bilingual-corpus/BDS/valid" \
        "--output_dir_format" "data/model/encoder_decoder/{date}" \
        "--model_name_format" "epoch-{epoch}.hdf5" \
        "--model" "naive" \
        "-lr" "1e-1" \
        "--momentum" "0.9" \
        "--nesterov" \
        "--epochs" "30" \
        "--seed" "2"
    ```

- Example: train_cifar10_classifier.py
    ```
    python examples/train_cifar10_classifier.py",
        "--train_image_npy_path" "data/preprocess/CIFAR-10/train_image.npy" \
        "--train_label_npy_path" "data/preprocess/CIFAR-10/train_label.npy" \
        "--test_image_npy_path" "data/preprocess/CIFAR-10/test_image.npy" \
        "--test_label_npy_path" "data/preprocess/CIFAR-10/test_label.npy" \
        "--output_dir_format" "data/model/cifar10/{date}" \
        "--model_name_format" "epoch-{epoch}.h5" \
        "--optimizer" "adam" \
        "-lr" "1e-1" \
        "--epochs" "30" \
        "--seed" "2"
    ```

- Example: train_cart_pole.py
    ```
    python examples/train_cart_pole.py
    ```
