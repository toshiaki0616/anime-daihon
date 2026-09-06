# AGENTS.md

このRepositoryは、アニメ音声から字幕候補を生成し、VAD・ASR・話者分離・時刻補正・voiceprint候補生成を行うPythonアプリを管理します。

## 開発フロー

Issue → Branch → Codex実装 → Pull Request → GitHub Actions → Review → Merge → Close

mainへ直接コミットしないでください。

## 実装方針

- 音声処理パイプラインの工程境界を維持する。
- `app.py` に新しい音声処理ロジックを戻さず、`services/` / `core/` へ責務を分離する。
- VAD、ASR、diarization、speaker assignment、timestamp refinementの変更は下流工程への影響を確認する。
- voiceprintは補助機能として扱い、主たる話者割当フローを不用意に置き換えない。
- debug JSONの構造を変更する場合は互換性と利用箇所を確認する。
- 重いモデル本体や生成音声・字幕・voiceprint実データをGitへ追加しない。

## 品質ゲート

最低限PR前に以下を満たすこと。

1. `python -m compileall -q app.py core models services ui`
2. 軽量テストが存在する場合は実行する
3. 実モデルを必要とする検証をCIで実行できない場合は、PRへ未実施理由と手動確認内容を記録する
4. パイプライン変更時は関連するdebug出力・字幕segment・話者ラベルへの影響を記録する

## Security / Data

以下をコミットしないこと。

- 実音声・動画・字幕データ
- voiceprint実データ
- 大容量モデル
- APIキー・トークン・`.env`
- 個人PC固有の絶対パス
- debug実出力

## Issueの分け方

- feature: 新機能
- bug: 不具合
- improvement: 品質・性能・構造改善
- documentation: README・運用・設計資料

1 Issue = 1目的を基本とします。

Related: #1 #2 #3
