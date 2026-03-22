# Subtitle Library UI - Step 6

Python + Gradio で作る字幕管理アプリの Step 6 実装です。  
この段階では保存・再読込・書き出しを追加し、セッションをまたいで編集を継続できるようにしています。

## 実装範囲

- 作品一覧ページ
- 作品詳細ページ
- 話数編集ページ
- 話者一覧ページ
- 前処理済み wav からの文字起こし
- ローカル speaker diarization
- JSON ベースの作品 / 話数保存
- 保存済みライブラリの再読込
- 編集継続
- TXT / CSV 書き出し

## 保存される内容

### Work

- work_id
- title
- character_names
- created_at
- updated_at

### Episode

- episode_id
- title
- status
- updated_at
- file_path / wav_path
- range_start / range_end
- enhance_audio
- subtitle_segments
- speakers
- merge_map
- speaker_label_map

### SubtitleSegment

- id
- start
- end
- speaker_id
- raw_label
- display_name
- original_text
- edited_text

## 保存先ディレクトリ

```text
project/
  data/
    works/
    episodes/
    exports/
```

- `data/works/*.json`: 作品メタデータ
- `data/episodes/*.json`: 話数ごとの構造化状態
- `data/exports/*`: txt / csv 書き出し

## 依存関係

```powershell
cd C:\work\codex\project
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 起動方法

```powershell
cd C:\work\codex\project
.venv\Scripts\Activate.ps1
python app.py
```

## Step 6 の使い方

- 話数編集ページで `保存` を押すと、現在の作品と話数の状態を JSON に保存
- ブラウザを更新しても、保存済みデータがあれば起動時に再読込
- 作品一覧の `保存済みを再読込` でディスク上の最新保存状態を読み直し可能
- `テキストを書き出す` は `[HH:MM:SS] 表示名：edited_text` 形式で txt 出力
- `CSVを書き出す` は `start_time, display_name, edited_text` 形式で csv 出力

## エラーメッセージ

- 保存に失敗しました
- 読み込みに失敗しました
- 書き出しに失敗しました

## 注意

- canonical state は JSON にそのまま保存しており、描画文字列だけを保存する実装にはしていません
- 再読込時は JSON から state を再構築したうえで speaker/group view を再計算します
