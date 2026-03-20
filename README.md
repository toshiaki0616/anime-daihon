# Subtitle Library UI - Step 5

Python + Gradio で作る字幕管理アプリの Step 5 実装です。  
この段階では文字起こしと話者分離の結果に対して、rename / merge / swap / 字幕編集の補正ワークフローを安定化しています。

## 実装範囲

- 作品一覧ページ
- 作品詳細ページ
- 話数編集ページ
- 話者一覧ページ
- 前処理済み wav からの文字起こし
- ローカル speaker diarization
- 字幕セグメントへの優勢話者割り当て
- 話者A / 話者B / 話者C の正規化表示
- rename / merge / swap の安定化
- edited_text の保持
- 操作後の即時 UI 再描画

未実装:

- 自動キャラ名付与
- JSON 保存

## 依存関係

Step 5 ではローカル Whisper 互換モデルとして `openai-whisper`、ローカル話者分離として `pyannote.audio` を使います。

```powershell
cd C:\work\codex\project
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`pyannote.audio` はローカルで利用できるモデルが必要です。必要に応じて `DIARIZATION_MODEL` 環境変数でローカルパスまたは利用可能なモデル名を指定してください。

```powershell
$env:DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
```

## 起動方法

```powershell
cd C:\work\codex\project
.venv\Scripts\Activate.ps1
python app.py
```

## Step 5 のポイント

- rename は `speaker_id` を変えず、`display_name` だけ更新
- merge は元のセグメント本文を壊さず、統合先へ再割り当てして即再描画
- swap は本文や display 名を壊さず、話者割り当てだけを入れ替え
- `edited_text` は rename / merge / swap / ページ移動の後も保持
- 字幕テーブルの `話者` 列を手で直して保存すると、その行の話者割り当ても state に反映
- UI は常に canonical state から再描画されるため、表示だけを書き換える実装にはしていません
