# 仮想環境の作り方

```
curl -fsSL https://pixi.sh/install.sh | sh

echo 'export PATH="$HOME/.pixi/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

# 動画からフレームの切り出し
```
pixi run python frame_cutter.py --path2video <PATH2VIDEO> --img_extension jpg --save_dir images
```

# DetectionとTracking

画像に対して人物検出（Detection）と追跡（Tracking）を実行します。

## 設定ファイル

`acca/conf/image_conf/conf.yaml`を編集して、以下のパラメータを設定します：

- `image_dir`: 処理する画像が格納されているディレクトリのパス（例: `"images/DSC_6628"`）
- `output_dir`: 結果を保存するルートディレクトリ（例: `"results"`）
- `image_extension`: 画像ファイルの拡張子（例: `"jpg"`）
- `resume`: 既存の状態ファイルから処理を再開するかどうか（`True`/`False`）
- `pipeline`: 実行する処理の順序（デフォルト: `detector` → `tracker`）

設定ファイルでは、`detector`と`tracker`の詳細設定も指定できます（デフォルトでは`p2pnet`と`bytetrack`が使用されます）。

## 実行方法

```
pixi run python acca/run_for_image.py
```

## 処理の流れ

1. **初期化**: `HanabiImageData`オブジェクトを作成し、画像ディレクトリから画像パスを取得します
2. **状態管理**: 出力ディレクトリに状態ファイル（JSON形式）を保存し、`resume=True`の場合は既存の状態ファイルから処理を再開できます
3. **Detectionフェーズ**: 検出器P2PNetを使用して、各画像から人物を検出します。検出結果は`{output_dir}/{event_name}/{place_name}/detection/`に保存されます
4. **Trackingフェーズ**: 検出結果を基に、ByteTrackアルゴリズムを使用して人物の追跡を行います。追跡結果は`{output_dir}/{event_name}/{place_name}/tracking/`に保存されます
5. **状態保存**: 各フェーズの実行後、処理状態がJSONファイルとして保存されます

出力ディレクトリの構造は以下の通りです：
```
{output_dir}/
  └── {event_name}/          # 画像ディレクトリの親ディレクトリ名
      └── {place_name}/       # 画像ディレクトリ名
          ├── detection/      # 検出結果
          ├── tracking/       # 追跡結果
          └── state/          # 状態ファイル（JSON）
```