# インストール手順
以下の順でインストールを行う
1. MeCab v0.996
1. CUDA 11.0
1. Python

## MeCab
64 bit 版 Python を使うため、あらかじめ 64 bit 版 MeCab をインストールする必要がある。
- Windows  
以下の URL からダウンロードする。  
https://github.com/ikegami-yukino/mecab/releases/tag/v0.996
- Linux
    - apt  
    以下のコマンドを実行する。  
    `sudo apt install mecab`
    - コンパイル  
    以下の URL からソースコードをダウンロードしコンパイルする。  
    https://taku910.github.io/mecab/

## CUDA
- CUDA 11.0 のドライバをインストール
    - `https://developer.nvidia.com/cuda-11.0-download-archive` で、OS・CPU アーキテクチャ・バージョンを適切に指定し、ドライバをダウンロード
    - ダウンロードしたファイルを実行し、ドライバをインストール

## Python
以下の順でインストールを行う
1. pyenv をインストール
1. pyenv で Python 3.8, 3.9 をインストール
1. Poetry をインストール
1. Poetry でパッケージをインストール

### PyEnv
1. インストール
    1. Linux では以下のコマンドを実行
        ```
        git clone https://github.com/pyenv/pyenv.git ~/.pyenv
        echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bash_profile
        echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bash_profile
        echo -e 'if command -v pyenv 1>/dev/null 2>&1; then\n  eval "$(pyenv init -)"\nfi' >> ~/.bash_profile
        source ~/.bash_profile
        ```
    1. windows では、`https://github.com/pyenv-win/pyenv-win` でダウンロードできる ZIP ファイルを展開し、Path に追加
1. 各バージョンの Python をインストール
    1. `pyenv update` でインストールできる Python のバージョンを更新
    1. `pyenv install --list` でインストールできる Python のバージョンを確認できる
    1. 以下のコマンドを実行し、Python 3.8 と Python 3.9 をインストール
        ```
        pyenv install 3.8.7
        pyenv install 3.9.1
        ```
    1. `pyenv version` でインストールした Python のバージョンを確認できる
    1. `pyenv local [バージョン名]` でカレントディレクトリで利用する Python のバージョンを指定
        1. venvs/gym ディレクトリで `pyenv local 3.8.7` を実行
        1. venvs/torch_cpu ディレクトリで `pyenv local 3.9.1` を実行
        1. venvs/torch_gpu ディレクトリで `pyenv local 3.9.1` を実行

### Poetry
1. インストール
    1. https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py から get-poetry.py をダウンロード
    1. `python get-poetry.py` を実行
    1. Poetry が不要になった場合は `python -m pip uninstall poetry` を実行
    1. Poetry があるディレクトリ (~/.poetry/bin) を Path に追加
1. ローカルディレクトリに仮想環境を作成
    1. `poetry config --list` で Poetry の設定を確認
    1. 仮想環境用ディレクトリの作成場所を設定
        - 仮想環境を現ディレクトリ下の .venv ディレクトリに作成するには、`poetry config virtualenvs.in-project true` を実行すればよい
    1. 仮想環境を作成し、パッケージをインストール
        1. OpenAI Gym を利用する場合は、venvs/gym ディレクトリで `poetry install` を実行
            - Windoes 上で train_pong.py を実行する場合、以下も行う
                1. 以下の URL から ale_c.dll をダウンロード  
                https://drive.google.com/u/0/uc?id=1WQrEBliYbASwNDyyVIlPFSZHRwAa7sPp&export=download
                1. venvs/gym/.venv/Lib/site-packages/atari_py/ale_interface/ ディレクトリ以下にダウンロードした ale_c.dll をコピー
        1. Torch を CPU で利用する場合は、venvs/torch_cpu ディレクトリで `poetry install` を実行
        1. Torch を GPU で利用する場合は、venvs/torch_gpu ディレクトリで `poetry install` を実行
1. ルートディレクトリを Python のモジュール検索パスに追加
    1. 以下を記述したテキストファイルを作成し、拡張子 pth で保存
    ```
    [インストールディレクトリ]
    ```
    1. 保存したファイルを [インストールディレクトリ]/.venv/Lib/site-packages ディレクトリ以下にコピー
1. その他、Poetry の使い方に関する Tips
    - パッケージを追加
    ```
    poetry add [package]
    ```
        - GPU 版 PyTorch を "torch_gpu" Extra グループの場合のみインストールするようにパッケージを追加
        ```
        poetry add "https://download.pytorch.org/whl/cu110/torch-1.7.1%2Bcu110-cp39-cp39-win_amd64.whl[torch_gpu]" --optional
        ``` 
    - パッケージを削除
    ```
    poetry remove [package]
    ```

# 使い方
インストールが適切に終わっていれば、以下のコマンド例を実行することで、スクリプトを試すことができる

## スクリプト実行例
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

- Example: Cart Pole タスクの強化学習
    ```
    python examples/train_cart_pole.py \
        --alpha 0.1 \
        --gamma 0.99 \
        --render
    ```

- Example: 迷路タスクの強化学習
    ```
    python examples/train_maze.py \
        --render
    ```
