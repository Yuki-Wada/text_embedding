# How to Install
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

# How to Use
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
