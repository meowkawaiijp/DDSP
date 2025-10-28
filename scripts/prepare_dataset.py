#!/usr/bin/env python3
"""
実際のギター音源からデータセットを準備するスクリプト

ユーザーが録音したギター音源を整理し、トレーニング用のデータセットを作成します。
"""

import argparse
import numpy as np
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import warnings


def check_audio_quality(audio: np.ndarray, sample_rate: int, filename: str) -> bool:
    """
    オーディオファイルの品質をチェック
    
    Args:
        audio: オーディオ信号
        sample_rate: サンプリングレート
        filename: ファイル名（エラーメッセージ用）
        
    Returns:
        品質が許容範囲内ならTrue
    """
    issues = []
    
    # クリッピングチェック
    if np.max(np.abs(audio)) >= 0.99:
        issues.append("クリッピングの可能性")
    
    # サイレンスチェック
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 0.001:
        issues.append("音量が非常に小さい")
    
    # DCオフセットチェック
    dc_offset = np.mean(audio)
    if abs(dc_offset) > 0.01:
        issues.append(f"DCオフセット検出 ({dc_offset:.4f})")
    
    if issues:
        warnings.warn(f"{filename}: {', '.join(issues)}")
        return False
    
    return True


def process_audio_file(
    input_path: Path,
    target_sample_rate: int = 48000,
    normalize: bool = True,
    remove_dc: bool = True
) -> tuple[np.ndarray, int]:
    """
    オーディオファイルを処理
    
    Args:
        input_path: 入力ファイルパス
        target_sample_rate: 目標サンプリングレート
        normalize: 正規化するか
        remove_dc: DCオフセットを除去するか
        
    Returns:
        (processed_audio, sample_rate)
    """
    # オーディオを読み込み
    audio, sr = sf.read(input_path)
    
    # ステレオからモノラルに変換
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    
    # float32に変換
    audio = audio.astype(np.float32)
    
    # DCオフセット除去
    if remove_dc:
        audio = audio - np.mean(audio)
    
    # リサンプリング（必要な場合）
    if sr != target_sample_rate:
        # 簡易的なリサンプリング（本番環境ではtorchaudio.resampleなどを推奨）
        import torch
        audio_tensor = torch.from_numpy(audio).view(1, 1, -1)
        n_samples = int(len(audio) * target_sample_rate / sr)
        audio_tensor = torch.nn.functional.interpolate(
            audio_tensor, size=n_samples, mode='linear', align_corners=False
        )
        audio = audio_tensor.view(-1).numpy()
        sr = target_sample_rate
    
    # 正規化
    if normalize:
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.8
    
    return audio, sr


def split_audio_into_segments(
    audio: np.ndarray,
    sample_rate: int,
    segment_duration: float = 4.0,
    overlap: float = 0.5
) -> list[np.ndarray]:
    """
    長いオーディオを複数のセグメントに分割
    
    Args:
        audio: オーディオ信号
        sample_rate: サンプリングレート
        segment_duration: セグメント長（秒）
        overlap: オーバーラップ比率（0-1）
        
    Returns:
        セグメントのリスト
    """
    segment_samples = int(segment_duration * sample_rate)
    hop_samples = int(segment_samples * (1 - overlap))
    
    segments = []
    start = 0
    
    while start + segment_samples <= len(audio):
        segment = audio[start:start + segment_samples]
        segments.append(segment)
        start += hop_samples
    
    # 最後のセグメント（長さが足りない場合はパディング）
    if start < len(audio):
        remaining = audio[start:]
        if len(remaining) > segment_samples * 0.5:  # 半分以上あれば採用
            padded = np.zeros(segment_samples, dtype=np.float32)
            padded[:len(remaining)] = remaining
            segments.append(padded)
    
    return segments


def prepare_dataset(
    input_dir: str,
    output_dir: str,
    train_ratio: float = 0.8,
    segment_duration: float = 4.0,
    overlap: float = 0.0,
    sample_rate: int = 48000,
    normalize: bool = True,
    remove_dc: bool = True,
    check_quality: bool = True
):
    """
    実際のギター音源からデータセットを準備
    
    Args:
        input_dir: 入力ディレクトリ（WAVファイルを含む）
        output_dir: 出力ディレクトリ
        train_ratio: トレーニングデータの比率（0-1）
        segment_duration: セグメント長（秒）
        overlap: オーバーラップ比率（0-1）
        sample_rate: 目標サンプリングレート
        normalize: 正規化するか
        remove_dc: DCオフセットを除去するか
        check_quality: 品質チェックを行うか
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # 入力ファイルを取得
    wav_files = list(input_path.glob('*.wav')) + list(input_path.glob('*.WAV'))
    
    if not wav_files:
        print(f"エラー: {input_dir} にWAVファイルが見つかりません")
        return
    
    print(f"{len(wav_files)} 個のWAVファイルを見つけました")
    
    # ディレクトリ構造を作成
    for split in ['train', 'val']:
        (output_path / split / 'audio').mkdir(parents=True, exist_ok=True)
    
    # すべてのセグメントを収集
    all_segments = []
    
    print("\nオーディオファイルを処理中...")
    for wav_file in tqdm(wav_files):
        try:
            # オーディオを処理
            audio, sr = process_audio_file(
                wav_file,
                target_sample_rate=sample_rate,
                normalize=normalize,
                remove_dc=remove_dc
            )
            
            # 品質チェック
            if check_quality:
                if not check_audio_quality(audio, sr, wav_file.name):
                    print(f"  警告: {wav_file.name} に品質問題があります（処理は続行）")
            
            # セグメントに分割
            if len(audio) / sr > segment_duration:
                segments = split_audio_into_segments(
                    audio, sr, segment_duration, overlap
                )
                all_segments.extend(segments)
            else:
                # 短いファイルはパディング
                padded = np.zeros(int(segment_duration * sr), dtype=np.float32)
                padded[:len(audio)] = audio
                all_segments.append(padded)
                
        except Exception as e:
            print(f"  エラー: {wav_file.name} の処理に失敗 - {e}")
            continue
    
    if not all_segments:
        print("エラー: 有効なセグメントが生成されませんでした")
        return
    
    print(f"\n合計 {len(all_segments)} セグメントを生成しました")
    
    # トレーニング/検証に分割
    np.random.shuffle(all_segments)
    n_train = int(len(all_segments) * train_ratio)
    train_segments = all_segments[:n_train]
    val_segments = all_segments[n_train:]
    
    print(f"トレーニング: {len(train_segments)} セグメント")
    print(f"検証: {len(val_segments)} セグメント")
    
    # 保存
    print("\nトレーニングデータを保存中...")
    for i, segment in enumerate(tqdm(train_segments)):
        output_file = output_path / 'train' / 'audio' / f'sample_{i:04d}.wav'
        sf.write(output_file, segment, sample_rate)
    
    print("検証データを保存中...")
    for i, segment in enumerate(tqdm(val_segments)):
        output_file = output_path / 'val' / 'audio' / f'sample_{i:04d}.wav'
        sf.write(output_file, segment, sample_rate)
    
    print(f"\n完了！データセットは {output_dir} に保存されました。")
    print("\nデータセット統計:")
    print(f"  合計セグメント: {len(all_segments)}")
    print(f"  トレーニング: {len(train_segments)} ({len(train_segments)/len(all_segments)*100:.1f}%)")
    print(f"  検証: {len(val_segments)} ({len(val_segments)/len(all_segments)*100:.1f}%)")
    print(f"  セグメント長: {segment_duration}秒")
    print(f"  サンプリングレート: {sample_rate}Hz")


def main():
    parser = argparse.ArgumentParser(
        description='実際のギター音源からトレーニング用データセットを準備します'
    )
    parser.add_argument(
        'input_dir',
        type=str,
        help='入力ディレクトリ（WAVファイルを含む）'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='dataset',
        help='出力ディレクトリのパス（デフォルト: dataset）'
    )
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.8,
        help='トレーニングデータの比率（デフォルト: 0.8）'
    )
    parser.add_argument(
        '--segment_duration',
        type=float,
        default=4.0,
        help='セグメント長（秒、デフォルト: 4.0）'
    )
    parser.add_argument(
        '--overlap',
        type=float,
        default=0.0,
        help='セグメントのオーバーラップ比率（0-1、デフォルト: 0.0）'
    )
    parser.add_argument(
        '--sample_rate',
        type=int,
        default=48000,
        help='目標サンプリングレート（Hz、デフォルト: 48000）'
    )
    parser.add_argument(
        '--no-normalize',
        action='store_true',
        help='正規化を無効化'
    )
    parser.add_argument(
        '--no-remove-dc',
        action='store_true',
        help='DCオフセット除去を無効化'
    )
    parser.add_argument(
        '--no-quality-check',
        action='store_true',
        help='品質チェックを無効化'
    )
    
    args = parser.parse_args()
    
    prepare_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        segment_duration=args.segment_duration,
        overlap=args.overlap,
        sample_rate=args.sample_rate,
        normalize=not args.no_normalize,
        remove_dc=not args.no_remove_dc,
        check_quality=not args.no_quality_check
    )


if __name__ == '__main__':
    main()

