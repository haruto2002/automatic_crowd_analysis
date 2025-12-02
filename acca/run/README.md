# run

検出（Detection）、トラッキング（Tracking）、可視化（Visualization）を含む完全なパイプラインを実行するモジュールです。

## 概要

このモジュールは、単一のビデオに対して以下の処理を順次実行します：

1. **検出（Detection）**: P2PNetを使用した人物頭部の検出
2. **トラッキング（Tracking）**: ByteTrackを使用した人物の追跡
3. **可視化（Visualization）**: 検出結果とトラッキング結果の可視化

## ディレクトリ構造

```
run/
├── main.py              # メイン実行スクリプト
├── detector.py           # 検出器の実装
├── tracker.py            # トラッカーの実装
├── visualizer.py         # 可視化の実装
└── conf/                 # 設定ファイル
    ├── config.yaml
    ├── detector/
    │   └── p2pnet.yaml
    ├── tracker/
    │   └── bytetrack.yaml
    ├── detection_visualizer/
    │   └── display_detection.yaml
    └── tracking_visualizer/
        └── display_trajectory.yaml
```

## 使用方法

### 基本的な実行

```bash
python -m run.main \
    video_dir=<ビデオディレクトリ> \
    video_name=<ビデオファイル名> \
    output_dir=<出力ディレクトリ>
```

### 実行例

```bash
python -m run.main \
    video_dir=demo_videos \
    video_name=demo_1s.mov \
    output_dir=demo_outputs
```

### Hydraを使用した実行

```bash
python -m run.main \
    video_dir=demo_videos \
    video_name=demo_1s.mov \
    output_dir=demo_outputs \
    hydra.run.dir=demo_outputs/hydra_log
```

## パラメータ

### 必須パラメータ

- `video_dir`: ビデオファイルが格納されているディレクトリ
- `video_name`: 処理するビデオファイル名
- `output_dir`: 出力結果を保存するディレクトリ

### オプションパラメータ

- `hydra.run.dir`: Hydraのログ出力ディレクトリ（デフォルト: `${output_dir}/hydra_log`）

## 出力ディレクトリ構造

```
<output_dir>/
├── <video_name_without_ext>/     # ビデオ名（拡張子なし）
│   ├── detection/                # 検出結果
│   │   ├── frame_000000.json
│   │   ├── frame_000001.json
│   │   └── ...
│   ├── track/                    # トラッキング結果
│   │   ├── track_000000.json
│   │   ├── track_000001.json
│   │   └── ...
│   └── vis/                      # 可視化結果
│       ├── detection/            # 検出可視化
│       │   ├── frame_000000.jpg
│       │   ├── frame_000001.jpg
│       │   └── ...
│       └── track.mp4             # トラッキング可視化動画
└── hydra_log/                    # Hydraログ
    ├── config.yaml
    ├── overrides.yaml
    └── main.log
```

## 処理フロー

1. **検出（Detection）**
   - 入力ビデオから各フレームを読み込み
   - P2PNetを使用して人物の頭部を検出
   - 検出結果をJSON形式で保存

2. **トラッキング（Tracking）**
   - 検出結果を読み込み
   - ByteTrackを使用して人物を追跡
   - トラッキング結果をJSON形式で保存

3. **可視化（Visualization）**
   - 検出結果を画像として可視化
   - トラッキング結果を動画として可視化

## 設定ファイル

### `conf/config.yaml`

メイン設定ファイル。以下のコンポーネントを設定します：

- `detector`: 検出器の設定
- `tracker`: トラッカーの設定
- `detection_visualizer`: 検出可視化の設定
- `tracking_visualizer`: トラッキング可視化の設定

### `conf/detector/p2pnet.yaml`

検出器（P2PNet）の設定ファイルです。人物頭部の検出に使用するP2PNetモデルの設定を行います。

#### パラメータ説明

- `cfg_path` (str): P2PNetの設定ファイルのパス
  - 例: `p2pnet/conf/p2p.yaml`
  - P2PNetモデルのアーキテクチャや学習時の設定が記載されたファイル

- `weight_path` (str): 学習済みモデルの重みファイルのパス
  - 例: `cutout.pth`
  - 事前学習済みのP2PNetモデルの重みファイル

- `gpu_id` (int): 使用するGPUのID
  - 例: `0`
  - 複数GPUが利用可能な場合、どのGPUを使用するかを指定

- `image_size` (list[int]): 入力画像のサイズ `[高さ, 幅]`
  - 例: `[4320, 7680]`
  - ビデオフレームをこのサイズにリサイズして検出処理を行います
  - 大きいサイズほど精度が向上しますが、処理時間とメモリ使用量が増加します

- `dtype` (str): データ型
  - 例: `float16`, `float32`
  - モデルの推論時に使用するデータ型
  - `float16`はメモリ使用量を削減できますが、精度が若干低下する可能性があります

### `conf/tracker/bytetrack.yaml`

トラッカー（ByteTrack）の設定ファイルです。検出された人物をフレーム間で追跡するための設定を行います。

#### パラメータ説明

- `img_h_size` (int): 画像の高さ
  - 例: `4320`
  - トラッキング処理に使用する画像の高さ（ピクセル）

- `img_w_size` (int): 画像の幅
  - 例: `7680`
  - トラッキング処理に使用する画像の幅（ピクセル）

- `track_thresh` (float): トラッキングの閾値
  - 例: `0.6`
  - 検出信頼度がこの値以上の検出結果のみをトラッキング対象とします
  - 値が高いほど、信頼度の高い検出のみを追跡します

- `track_buffer` (int): トラッキングバッファサイズ
  - 例: `30`
  - 一時的に検出されなくなった人物を保持するフレーム数
  - 値が大きいほど、一時的な検出失敗に対して頑健になりますが、メモリ使用量が増加します

- `match_thresh` (float): マッチングの閾値
  - 例: `10.0`
  - フレーム間で人物をマッチングする際の距離閾値
  - 値が大きいほど、遠く離れた位置の人物も同じIDとして追跡できます

- `distance_metric` (str): 距離計量
  - 例: `euclidean`
  - 人物間の距離を計算する方法
  - `euclidean`: ユークリッド距離を使用

### `conf/detection_visualizer/display_detection.yaml`

検出結果の可視化設定ファイルです。検出された人物頭部を画像上に描画する際の設定を行います。

#### パラメータ説明

- `threshold` (float): 可視化する検出結果の信頼度閾値
  - 例: `0.5`
  - この値以上の信頼度を持つ検出結果のみを可視化します
  - 値が高いほど、信頼度の高い検出のみが表示されます

- `freq` (int): 可視化するフレームの間隔
  - 例: `10`
  - この値で指定した間隔のフレームのみを可視化します
  - `10`の場合、10フレームごとに1枚の画像を生成します
  - 値が大きいほど、生成される画像数が減り、処理時間が短縮されます

- `downscale_ratio` (float): 画像の縮小率
  - 例: `0.5`
  - 可視化画像を生成する際に、元画像をこの比率で縮小します
  - `0.5`の場合、元画像の50%のサイズで可視化画像を生成します
  - 値が小さいほど、生成される画像サイズが小さくなり、処理時間と保存容量が削減されます

### `conf/tracking_visualizer/display_trajectory.yaml`

トラッキング結果の可視化設定ファイルです。追跡された人物の軌跡を動画として可視化する際の設定を行います。

#### パラメータ説明

- `freq` (int): 可視化するフレームの間隔
  - 例: `5`
  - この値で指定した間隔のフレームのみを可視化動画に含めます
  - `5`の場合、5フレームごとに1フレームを動画に含めます
  - 値が大きいほど、生成される動画が短くなり、処理時間が短縮されます

- `start_frame` (int or None): 可視化を開始するフレーム番号
  - 例: `None`（全フレームを可視化）
  - 指定したフレーム番号から可視化を開始します
  - `None`の場合は最初のフレームから開始します

- `end_frame` (int or None): 可視化を終了するフレーム番号
  - 例: `None`（最後のフレームまで可視化）
  - 指定したフレーム番号まで可視化します
  - `None`の場合は最後のフレームまで可視化します

- `downscale_ratio` (float): 画像の縮小率
  - 例: `0.5`
  - 可視化動画を生成する際に、元画像をこの比率で縮小します
  - `0.5`の場合、元画像の50%のサイズで可視化動画を生成します
  - 値が小さいほど、生成される動画サイズが小さくなり、処理時間と保存容量が削減されます