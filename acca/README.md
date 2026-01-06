# Setup environment

We utilize Miniconda to create the virtual environment and Python=3.10.  
The required packages are installed by executing `setup.sh`.

```bash
curl -fsSL https://pixi.sh/install.sh | sh

pixi run -e pre-install build-hdf5
pixi run -e pre-install remove-cache
pixi install
```

$HOME/.bash_profileに以下を記入しておくと、次回から自動でロードされる
```bash
# .bash_profile

# Get the aliases and functions
if [ -f ~/.bashrc ]; then
	. ~/.bashrc
fi

# User specific environment and startup programs

PATH=$PATH:$HOME/.local/bin:$HOME/bin

export PATH
```

# 画像に対する処理の実行方法

`run_detection_for_image.py`を使用して画像に対する検出処理を実行します。

## 実行コマンド例

```bash
pixi run python acca/run_detection_for_image.py \
    image_dir=/home/aag16599bn/research/hanabi_videos/5min_freq_images \
    output_dir=demo \
    image_extension=jpg \
    task_index=1 \
    task_stepsize=1
```

## パラメータ説明

- `image_dir`: 画像ディレクトリのパス。`*/*`パターンでサブディレクトリを検索します
- `output_dir`: 出力先のルートディレクトリ
- `image_extension`: 画像ファイルの拡張子（例: `jpg`, `png`）
- `task_index`: 処理を開始するタスクのインデックス（1から開始）
- `task_stepsize`: 処理するタスクの数
- `resume`: 既存の状態ファイルから再開するかどうか（デフォルト: `False`）

## 環境変数による指定

`TASK_INDEX`と`TASK_STEPSIZE`を環境変数で指定することも可能です：

```bash
export TASK_INDEX=1
export TASK_STEPSIZE=1
pixi run python acca/run_detection_for_image.py \
    image_dir=/home/aag16599bn/research/hanabi_videos/5min_freq_images \
    output_dir=demo \
    image_extension=jpg
```

## 処理の流れ

1. `image_dir`配下の`*/*`パターンに一致するディレクトリを取得
2. `task_index`と`task_stepsize`に基づいて処理対象を選択
3. 各画像ディレクトリに対して：
   - `HanabiImageData`オブジェクトを作成
   - 既存の状態ファイルをチェック
   - `resume=True`かつ既存の状態ファイルがある場合、前回の状態を読み込み
   - パイプライン（検出器、グラフ動画作成など）を実行
   - 状態を保存