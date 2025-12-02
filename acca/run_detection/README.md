# run_detection

検出（Detection）のみを実行するモジュールです。複数のGPUを使用した並列処理に対応しています。

## 概要

このモジュールは、ビデオから人物の頭部を検出する処理を実行します。P2PNetを使用した検出処理を行い、複数のGPUで並列処理することで効率的に大量のビデオを処理できます。

## ディレクトリ構造

```
run_detection/
├── main.py              # メイン実行スクリプト
├── set.py               # 設定ファイルとrun.shを生成
├── detector.py           # 検出器の実装
├── io_setting.py         # 入出力設定
├── video_setting.py      # ビデオ設定
└── conf/                 # 設定ファイル
    ├── config.yaml
    ├── detector/
    │   └── p2pnet.yaml
    └── io_setting/
        └── io_setting.yaml
```

## 使用方法

### 1. 設定ファイルとrun.shの生成

まず、`set.py`を使用してビデオの分割設定と実行スクリプトを生成します。

```bash
python -m run_detection.set \
    --video_root_dir <ビデオディレクトリ> \
    --save_root_dir <出力ディレクトリ> \
    --node_type <ノードタイプ> \
    --pbs_walltime <実行時間>
```

#### パラメータ

- `--video_root_dir`: 処理するビデオファイルが格納されているディレクトリ
- `--save_root_dir`: 出力結果を保存するディレクトリ
- `--node_type`: PBSノードタイプ（`rt_HF`: 8GPU, `rt_HG`: 1GPU, `rt_HC`: CPUのみ）
- `--pbs_walltime`: PBSジョブの実行時間（例: `1:00:00`）

#### 実行例

```bash
python -m run_detection.set \
    --video_root_dir demo_videos \
    --save_root_dir demo_outputs/demo_videos \
    --node_type rt_HG \
    --pbs_walltime 1:00:00
```

このコマンドを実行すると、以下のファイルが生成されます：

- `<save_root_dir>/detection_run_info/input_video_path_{gpu_id}.json`: 各GPUに割り当てるビデオリスト
- `<save_root_dir>/detection_run_info/run.sh`: PBSジョブ実行スクリプト

### 2. 検出処理の実行
生成された`run.sh`を実行します：

#### インタラクティブノードで実行

```bash
bash <save_root_dir>/detection_run_info/run.sh
```

#### PBSジョブとして実行

```bash
qsub <save_root_dir>/detection_run_info/run.sh
```

## ノードタイプとGPU数

| ノードタイプ | GPU数 | 説明 |
|------------|-------|------|
| `rt_HF`    | 8     | 高性能GPUノード（8GPU並列処理） |
| `rt_HG`    | 1     | 標準GPUノード（1GPU処理） |
| `rt_HC`    | 0     | CPUノード（GPUなし） |

## 出力ディレクトリ構造

```
<save_root_dir>/
├── detection_run_info/          # 実行設定ファイル
│   ├── input_video_path_0.json
│   ├── input_video_path_1.json
│   └── run.sh
├── detection_hydra_log/         # Hydraログ
└── results/                      # 検出結果
    └── <video_name>/
        └── detection/
            ├── frame_000000.json
            ├── frame_000001.json
            └── ...
```

## 設定ファイル

### `conf/config.yaml`

メイン設定ファイル。以下のパラメータを設定できます：

- `input_video_path_dir`: 入力ビデオパスディレクトリ（JSONファイルのディレクトリ）
- `output_dir`: 出力ディレクトリ
- `gpu_id`: 使用するGPU ID

### `conf/detector/p2pnet.yaml`

検出器（P2PNet）の設定：

- `cfg_path`: P2PNet設定ファイルのパス
- `weight_path`: 学習済みモデルの重みファイルパス
- `image_size`: 入力画像サイズ
- `dtype`: データ型（`float32`など）

### `conf/io_setting/io_setting.yaml`

入出力設定：

- `input_video_path_dir`: 入力ビデオパスディレクトリ
- `gpu_id`: GPU ID
- `save_root_dir`: 保存先ルートディレクトリ

## 並列処理の仕組み

1. `set.py`がビデオリストをGPU数に応じて分割
2. 各GPUに割り当てられたビデオリストをJSONファイルとして保存
3. `run.sh`が各GPUで並列に`main.py`を実行
4. 各GPUが割り当てられたビデオを順次処理

