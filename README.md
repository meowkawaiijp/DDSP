# DDSP for Guitar

このプロジェクトは、エレキギターの音声を合成するために特化した微分可能デジタル信号処理（Differentiable Digital Signal Processing, DDSP）の実装です。ハーモニック合成、ノイズ合成、ウェーブシェイピング、トーンスタックを組み合わせて、エレキギターのサウンドを正確にモデル化します。

## 機能

- **ハーモニック合成（Harmonic Synthesis）:** ギターサウンドの基本周波数と倍音を生成します。
- **ノイズ合成（Noise Synthesis）:** サウンドのノイズ成分をモデル化します。
- **ウェーブシェイパー（Waveshaper）:** 非線形歪みを導入するためのパラメトリックtanhベースのウェーブシェイパー。
- **トーンスタック（Tonestack）:** 音色特性を形成する3バンドEQ（Low、Mid、High）。
- **トランジェント分離（Transient Separator）:** より詳細な処理のために、オーディオをトランジェント成分と定常成分に分離します。

## インストール

1. リポジトリをクローンします：
   ```bash
   git clone <repository-url>
   cd ddsp_guitar
   ```

2. 必要な依存関係をインストールします：
   ```bash
   pip install -r requirements.txt
   ```

## 基本的な使い方

モデルを読み込んでオーディオを合成する基本的な例です。

```python
import torch
from ddsp_guitar.model import GuitarDDSP

# --- 設定 ---
SAMPLE_RATE = 48000
AUDIO_LENGTH_SECONDS = 4
FRAME_RATE = 250
HOP_SIZE = SAMPLE_RATE // FRAME_RATE

# --- ダミー入力を作成 ---
# 利用可能な場合はCUDAを使用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 1
audio_length_samples = AUDIO_LENGTH_SECONDS * SAMPLE_RATE
n_frames = audio_length_samples // HOP_SIZE

# ダミーオーディオ入力（例：マイクやファイルから）
dummy_audio = torch.randn(batch_size, audio_length_samples).to(device)

# ダミーのf0とラウドネス（通常は入力オーディオから抽出されます）
dummy_f0_hz = torch.full((batch_size, n_frames), 110.0).to(device) # A2音
dummy_loudness = torch.full((batch_size, n_frames), -30.0).to(device)

# --- モデルの読み込み ---
model = GuitarDDSP(sample_rate=SAMPLE_RATE).to(device)

# --- オーディオ合成 ---
# 注意：モデルのフォワードパスは特定の形状を期待します。
# これは簡略化された例です。
# 入力テンソルを期待されるモデルの次元に適合させる必要があります。
# モデルのフォワードパスへのオーディオ入力は、通常、生のオーディオではなく特定の特徴表現です。
# この例では、正しい形状のテンソルを渡しますが、適切な条件付けなしでは意味のある音は生成されません。
conditioning_input = torch.randn(batch_size, n_frames).to(device)


# 実際の使用例では、`dummy_audio`から特徴量を抽出し、
# それをモデルのフォワードパスへの入力として使用します。
# `model.py`の`forward`メソッドは`x`、`f0_hz`、`loudness`を期待します。
# ここでは`x`が条件付け入力であると仮定します。
output_audio = model(conditioning_input, dummy_f0_hz, dummy_loudness)

print("合成完了！")
print("出力オーディオの形状:", output_audio.shape)
```

より詳細な例とチュートリアルについては、ドキュメントと提供されているGoogle Colabノートブックを参照してください。

