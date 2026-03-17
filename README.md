# Windows Local Subtitle Assistant - Phase 1

Windows ローカルで動かす字幕補助ツールの Phase 1 実装です。  
この段階では、Gradio の UI 骨組みと構造化ステート管理のみを実装しています。

## 実装範囲

- 2ページ UI
  - 字幕作成ページ
  - 話者一覧ページ
- モック字幕データの表示
- 字幕行の編集
- 話者名の変更
- 話者の統合
- 話者の入れ替え
- UI の即時更新
- 構造化ステートからの再描画

未実装:

- ffmpeg による音声抽出
- Whisper 互換の文字起こし
- 話者分離
- 音声強調処理

## ディレクトリ構成

```text
project/
  app.py
  README.md
  requirements.txt
  ui/
    __init__.py
    renderers.py
  core/
    __init__.py
    state_ops.py
  models/
    __init__.py
    state.py
  services/
    __init__.py
    mock_pipeline.py
  data/
```

## セットアップ

1. Python 3.10 以上を用意します
2. 仮想環境を作成します
3. 依存関係をインストールします

```powershell
cd project
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 起動

```powershell
cd project
python app.py
```

起動後はブラウザで Gradio UI を開き、適当な `mp4` / `wav` / `mp3` を選ぶと、実処理の代わりにモック字幕が表示されます。

## 状態管理の考え方

字幕編集は文字列置換ではなく、以下の構造化データを保持します。

- `SubtitleSegment`
  - セグメント単位の時刻、話者、元テキスト、編集テキスト
- `SpeakerProfile`
  - 話者 ID、表示名、件数、サンプル発話
- `AppState`
  - 入力条件、字幕一覧、話者一覧、統合マップ

UI は毎回この状態から再描画されるため、Phase 2 以降で音声処理を追加しても保守しやすい構成です。

## 次フェーズ拡張メモ

- `services/` に ffmpeg ベースの変換処理を追加
- `services/` に文字起こしと diarization のローカル実装を追加
- `core/state_ops.py` に統合ロジックを追加し、実結果と UI を接続
- `data/` に一時 wav や処理キャッシュを保持
