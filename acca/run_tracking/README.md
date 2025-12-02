## run_tracking

Hydra ベースのトラッキング一括実行モジュールです。`result_dir` 配下の各シーン/動画ディレクトリを走査し、`detection/` にある検出結果を入力としてトラッキングを実行し、`track/` に結果を保存します。

### ディレクトリ構成（想定）
```
<project_root>/
  demo_outputs/demo_videos/
    results/
      <scene_or_video_A>/
        detection/           # 入力（検出結果：ByteTrack互換のフォーマットを想定）
        track/               # 出力（トラッキング結果；実行時に自動生成）
      <scene_or_video_B>/
        detection/
        track/
  run_tracking/
    conf/
      config.yaml
      tracker/
        bytetrack.yaml
    main.py
    tracker.py
```

- 入力: 各 `<scene_or_video_*>/detection/`
- 出力: 各 `<scene_or_video_*>/track/`（存在しない場合は自動作成）

### 実行方法
プロジェクトルートで以下を実行します。

```bash
python -m run_tracking.main
```

Hydra を用いているため、設定はコマンドラインから上書き可能です（後述）。

### 設定

- `run_tracking/conf/config.yaml`
  - `defaults`: 使用するトラッカー設定を選択（デフォルトは `tracker: bytetrack`）。
  - `node_type`: ノード種別の任意ラベル（処理ロジックには未使用）。
  - `result_dir`: 入出力のベースディレクトリ（各サブディレクトリ下の `detection/` を読み、`track/` に保存）。
  - `hydra_log_dir`: Hydra の出力（ログ、`tracking_time.txt` など）の保存先。
  - `hydra.run.dir`: 実行ごとの出力先（既定で `hydra_log_dir`）。

- `run_tracking/conf/tracker/bytetrack.yaml`
  - `_target_`: 使用するトラッカー実装。デフォルトは `run_tracking.tracker.Tracker`。
  - `img_h_size` / `img_w_size`: 画像解像度（高さ/幅）。
  - `track_thresh`: トラック生成/継続の閾値。
  - `track_buffer`: ロスト許容フレーム数。
  - `match_thresh`: マッチング距離の閾値。
  - `distance_metric`: マッチング距離の種類（`euclidean` など）。

- `run_tracking/main.py`
  - `Pool` による並列実行を行います。プールサイズはデフォルトで `32` に固定されています（必要に応じてソースを編集してください）。
  - 実行時間の計測結果は `hydra_log_dir/tracking_time.txt` に出力されます。
