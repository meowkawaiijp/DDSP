#!/usr/bin/env python3
"""
ダミーデータセット生成スクリプト

テスト・開発用のシンプルなギター風音源を生成します。
実際のギター音源の代わりに、サイン波ベースの音を生成します。
"""

import argparse
import numpy as np
import soundfile as sf
from pathlib import Path
from tqdm import tqdm


# ==========================
# ユーティリティ関数群
# ==========================

def _one_pole_lowpass(x_prev: float, x_cur: float, y_prev: float, coeff: float) -> float:
    """
    1次ローパスフィルタの1サンプル更新。
    coeffが大きいほどスムーズ（高域が削れる）。
    """
    # y[n] = (1 - a) * x[n] + a * y[n-1]
    return (1.0 - coeff) * x_cur + coeff * y_prev


def _convolve_short_ir(x: np.ndarray, ir: np.ndarray) -> np.ndarray:
    """
    短いIRとの軽量な畳み込み（時間領域）。
    IRは極力短く（<= 256）保つこと。
    """
    n = len(x)
    m = len(ir)
    y = np.zeros(n + m - 1, dtype=np.float32)
    # 単純な時間畳み込み（IR短い想定）
    for i in range(m):
        y[i:i + n] += x.astype(np.float32) * ir[i]
    return y[:n]


def _make_body_ir(sample_rate: int, length_samples: int = 192) -> np.ndarray:
    """
    簡易ボディ共鳴IRを生成（複数の減衰サイン合成）。
    長さは短めにし、音色付けに特化。
    """
    t = np.arange(length_samples) / sample_rate
    # 代表的な共鳴（簡易）
    modes_hz = [95.0, 190.0, 380.0, 760.0, 1200.0]
    ir = np.zeros_like(t, dtype=np.float32)
    for i, f in enumerate(modes_hz):
        # 低域モードは長く残り、高域は速く減衰
        decay = np.exp(-t * (1.2 + 0.6 * i))
        ir += (0.6 / (i + 1)) * decay * np.sin(2 * np.pi * f * t)
    # 先頭に軽いインパルス成分を加えてアタックを強調
    ir[0] += 1.0
    # 正規化してゲインを統一
    ir /= np.max(np.abs(ir)) + 1e-8
    # 軽くゲインを下げてクリップ回避
    return (ir * 0.4).astype(np.float32)


def _apply_dc_blocker(x: np.ndarray, alpha: float = 0.995) -> np.ndarray:
    """
    DC除去用の単純なハイパス（1次）
    """
    y = np.zeros_like(x)
    x_prev = 0.0
    y_prev = 0.0
    for i in range(len(x)):
        y[i] = x[i] - x_prev + alpha * y_prev
        x_prev = x[i]
        y_prev = y[i]
    return y


# ==========================
# 演奏表現エフェクト関数群
# ==========================

def apply_vibrato(
    audio: np.ndarray,
    f0: float,
    sample_rate: int,
    depth_cents: float = 50.0,
    rate_hz: float = 5.0,
    fade_in_sec: float = 0.1
) -> np.ndarray:
    """
    ビブラートを適用（音高の周期的な揺らぎ）
    
    Args:
        audio: 入力オーディオ
        f0: 基本周波数
        sample_rate: サンプリングレート
        depth_cents: ビブラートの深さ（セント単位、100cent = 半音）
        rate_hz: ビブラートの速度（Hz）
        fade_in_sec: ビブラートのフェードイン時間
    """
    n = len(audio)
    t = np.arange(n) / sample_rate
    
    # ビブラートのフェードイン
    fade_in_samples = int(fade_in_sec * sample_rate)
    fade_curve = np.ones(n, dtype=np.float32)
    if fade_in_samples > 0 and fade_in_samples < n:
        fade_curve[:fade_in_samples] = np.linspace(0, 1, fade_in_samples) ** 2
    
    # ビブラートLFO（低周波振動）
    lfo = np.sin(2 * np.pi * rate_hz * t) * fade_curve
    
    # セントから周波数比へ変換
    freq_ratio = 2 ** (depth_cents * lfo / 1200.0)
    
    # 時間伸縮による音高変調（簡易版: 位相補間）
    # より正確にはリサンプリングが必要だが、ここでは軽量な近似
    delay_samples = (1.0 - freq_ratio) * (sample_rate / f0)
    
    output = np.zeros_like(audio)
    for i in range(n):
        # 読み取り位置を計算
        read_pos = i + delay_samples[i]
        
        # 境界チェック
        read_pos = np.clip(read_pos, 0, n - 1)
        
        # 線形補間
        idx_low = int(np.floor(read_pos))
        idx_high = min(idx_low + 1, n - 1)
        frac = read_pos - idx_low
        
        # 境界内の安全な読み取り
        if 0 <= idx_low < n:
            output[i] = audio[idx_low] * (1.0 - frac)
            if idx_high < n and idx_high != idx_low:
                output[i] += audio[idx_high] * frac
    
    # 異常値のチェック
    output = np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)
    
    return output.astype(np.float32)


def apply_pitch_bend(
    audio: np.ndarray,
    f0_start: float,
    f0_end: float,
    sample_rate: int,
    bend_duration_ratio: float = 0.3
) -> np.ndarray:
    """
    ピッチベンド（チョーキング）を適用
    
    Args:
        audio: 入力オーディオ
        f0_start: 開始周波数
        f0_end: 終了周波数
        sample_rate: サンプリングレート
        bend_duration_ratio: ベンドにかける時間の割合（0-1）
    """
    n = len(audio)
    bend_samples = int(n * bend_duration_ratio)
    
    # ベンドカーブ（前半でベンド、後半は一定）
    freq_curve = np.ones(n, dtype=np.float32) * f0_end
    if bend_samples > 0:
        # S字カーブでスムーズにベンド
        t = np.linspace(0, 1, bend_samples)
        smooth_curve = t ** 2 * (3 - 2 * t)  # スムーズステップ
        freq_curve[:bend_samples] = f0_start + (f0_end - f0_start) * smooth_curve
    
    # 周波数比を計算
    freq_ratio = freq_curve / f0_start
    
    # 累積位相を使ってリサンプリング
    phase_increment = freq_ratio
    cumulative_phase = np.cumsum(phase_increment)
    read_positions = cumulative_phase - cumulative_phase[0]
    
    output = np.zeros_like(audio)
    for i in range(n):
        read_pos = read_positions[i]
        
        # 境界チェック
        read_pos = np.clip(read_pos, 0, n - 1)
        
        idx_low = int(np.floor(read_pos))
        idx_high = min(idx_low + 1, n - 1)
        frac = read_pos - idx_low
        
        # 境界内の安全な読み取り
        if 0 <= idx_low < n:
            output[i] = audio[idx_low] * (1.0 - frac)
            if idx_high < n and idx_high != idx_low:
                output[i] += audio[idx_high] * frac
    
    # 異常値のチェック
    output = np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)
    
    return output.astype(np.float32)


def generate_slide_note(
    f0_start: float,
    f0_end: float,
    duration: float,
    sample_rate: int,
    velocity: float = 0.7,
    slide_duration_ratio: float = 0.5
) -> np.ndarray:
    """
    スライド音を生成（音程間の連続的な移動）
    
    Args:
        f0_start: 開始周波数
        f0_end: 終了周波数
        duration: 継続時間
        sample_rate: サンプリングレート
        velocity: ベロシティ
        slide_duration_ratio: スライドにかける時間の割合
    """
    n_samples = int(duration * sample_rate)
    slide_samples = int(n_samples * slide_duration_ratio)
    
    # スライドカーブ（指数的に滑らか）
    if slide_samples > 0:
        t = np.linspace(0, 1, slide_samples)
        # 対数スケールでスライド（より自然な音程変化）
        log_start = np.log(f0_start)
        log_end = np.log(f0_end)
        smooth_curve = t ** 1.5  # 加速感のあるカーブ
        freq_curve = np.exp(log_start + (log_end - log_start) * smooth_curve)
        
        # 残りは終了周波数で一定
        if n_samples > slide_samples:
            freq_curve = np.concatenate([
                freq_curve,
                np.full(n_samples - slide_samples, f0_end, dtype=np.float32)
            ])
    else:
        freq_curve = np.full(n_samples, f0_end, dtype=np.float32)
    
    # 各瞬間の周波数で短いKS音を合成して接続
    # 簡易版: セグメント分割して生成
    segment_duration = 0.02  # 20ms セグメント
    segment_samples = int(segment_duration * sample_rate)
    
    output = np.zeros(n_samples, dtype=np.float32)
    overlap = segment_samples // 4  # オーバーラップを減らして加算時の歪みを軽減
    
    pos = 0
    while pos < n_samples:
        end_pos = min(pos + segment_samples, n_samples)
        seg_len = end_pos - pos
        
        if seg_len <= 0:
            break
        
        # このセグメントの平均周波数
        f0_seg = float(np.mean(freq_curve[pos:end_pos]))
        
        # 短い音を生成
        seg_audio = karplus_strong_pluck(
            f0=f0_seg,
            duration=seg_len / sample_rate,
            sample_rate=sample_rate,
            brightness=0.5 + 0.3 * velocity,
            pick_position=0.3,
            damping_end=0.88
        )
        
        # 生成された音の長さを確認して調整
        actual_len = min(len(seg_audio), seg_len, end_pos - pos)
        
        # クロスフェード（より滑らかに）
        if pos > 0 and overlap > 0:
            fade_len = min(overlap, actual_len, n_samples - pos)
            if fade_len > 1:
                # S字カーブでより滑らかなクロスフェード
                t = np.linspace(0, 1, fade_len).astype(np.float32)
                fade_in = t ** 2 * (3 - 2 * t)  # スムーズステップ
                fade_out = 1.0 - fade_in
                
                # 新しいセグメントはフェードイン
                seg_audio[:fade_len] *= fade_in
                
                # 既存の音はフェードアウト
                if pos + fade_len <= n_samples:
                    output[pos:pos + fade_len] *= fade_out
        
        # 安全に追加（境界チェック強化）
        add_len = min(actual_len, n_samples - pos)
        if add_len > 0:
            output[pos:pos + add_len] += seg_audio[:add_len]
        
        pos += segment_samples - overlap
    
    # セグメント加算後の正規化（クリッピング防止）
    # 異常値のチェック
    output = np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)
    
    peak = float(np.max(np.abs(output)) + 1e-8)
    if peak > 0.001:  # ほぼ無音でない場合のみ正規化
        # より控えめな正規化
        target_level = 0.6 * velocity
        output = (output / peak) * target_level
    
    return output.astype(np.float32)


def apply_palm_mute(
    audio: np.ndarray,
    mute_amount: float = 0.7,
    damping_freq: float = 800.0,
    sample_rate: int = 48000
) -> np.ndarray:
    """
    パームミュート効果を適用
    
    Args:
        audio: 入力オーディオ
        mute_amount: ミュートの強さ（0-1）
        damping_freq: ダンピング周波数（Hz）
        sample_rate: サンプリングレート
    """
    # ローパスフィルタでこもった音色に
    from scipy import signal
    
    # カットオフ周波数をミュート量に応じて調整
    cutoff = damping_freq * (1.0 - 0.7 * mute_amount)
    nyquist = sample_rate / 2
    normalized_cutoff = cutoff / nyquist
    
    # 正規化カットオフ周波数を安全な範囲に制限（0.01 ~ 0.99）
    normalized_cutoff = np.clip(normalized_cutoff, 0.01, 0.99)
    
    # バターワースフィルタ
    b, a = signal.butter(2, normalized_cutoff, btype='low')
    filtered = signal.filtfilt(b, a, audio)
    
    # サステインを短くする（エンベロープ調整）
    n = len(audio)
    decay_curve = np.exp(-np.arange(n) / (sample_rate * 0.15 * (1.0 - 0.7 * mute_amount)))
    
    # 元の音とブレンド
    output = filtered * decay_curve
    output = audio * (1.0 - mute_amount) + output * mute_amount
    
    # 異常値のチェック
    output = np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)
    
    # ブレンド後の安全正規化（クリッピング防止）
    peak = float(np.max(np.abs(output)) + 1e-8)
    if peak > 0.9:  # より高い閾値で正規化
        output = output / peak * 0.85
    
    return output.astype(np.float32)


# ==========================
# 音楽理論・フレーズ構造関数群
# ==========================

def get_chord_tones(chord_type: str, root_freq: float) -> list:
    """
    コードトーンの周波数リストを取得
    
    Args:
        chord_type: コードタイプ ('major', 'minor', 'dominant7', 'minor7')
        root_freq: ルート音の周波数
    
    Returns:
        コードトーンの周波数リスト
    """
    # 半音比率
    semitone_ratio = 2 ** (1/12)
    
    intervals = {
        'major': [0, 4, 7],           # Root, Major 3rd, Perfect 5th
        'minor': [0, 3, 7],           # Root, Minor 3rd, Perfect 5th
        'dominant7': [0, 4, 7, 10],   # Root, Major 3rd, Perfect 5th, Minor 7th
        'minor7': [0, 3, 7, 10],      # Root, Minor 3rd, Perfect 5th, Minor 7th
    }
    
    if chord_type not in intervals:
        chord_type = 'minor'
    
    tones = []
    for interval in intervals[chord_type]:
        freq = root_freq * (semitone_ratio ** interval)
        tones.append(freq)
    
    return tones


def create_scale_from_root(root_freq: float, scale_type: str = 'minor_pentatonic') -> np.ndarray:
    """
    ルート音からスケールを生成
    
    Args:
        root_freq: ルート音の周波数
        scale_type: スケールタイプ
    
    Returns:
        スケール音の周波数配列（複数オクターブ）
    """
    semitone_ratio = 2 ** (1/12)
    
    # スケールインターバル（半音単位）
    intervals = {
        'minor_pentatonic': [0, 3, 5, 7, 10],  # E, G, A, B, D
        'major_pentatonic': [0, 2, 4, 7, 9],    # C, D, E, G, A
        'blues': [0, 3, 5, 6, 7, 10],           # Minor penta + blue note
        'natural_minor': [0, 2, 3, 5, 7, 8, 10], # Natural minor scale
    }
    
    if scale_type not in intervals:
        scale_type = 'minor_pentatonic'
    
    scale = []
    # 2オクターブ分生成
    for octave in [0.5, 1.0, 2.0, 4.0]:
        for interval in intervals[scale_type]:
            freq = root_freq * octave * (semitone_ratio ** interval)
            if 60.0 <= freq <= 1500.0:  # 実用的な範囲に制限
                scale.append(freq)
    
    return np.array(sorted(scale))


def generate_phrase_pattern(
    pattern_type: str,
    scale: np.ndarray,
    num_notes: int,
    start_note_idx: int = None
) -> list:
    """
    フレーズパターンを生成
    
    Args:
        pattern_type: パターンタイプ ('ascending', 'descending', 'arpeggio', 'random_walk')
        scale: スケール音の配列
        num_notes: 音符数
        start_note_idx: 開始音のインデックス（Noneの場合はランダム）
    
    Returns:
        周波数のリスト
    """
    if start_note_idx is None:
        start_note_idx = np.random.randint(len(scale) // 3, 2 * len(scale) // 3)
    
    notes = []
    
    if pattern_type == 'ascending':
        # 上昇スケール
        for i in range(num_notes):
            idx = (start_note_idx + i) % len(scale)
            notes.append(scale[idx])
    
    elif pattern_type == 'descending':
        # 下降スケール
        for i in range(num_notes):
            idx = (start_note_idx - i) % len(scale)
            notes.append(scale[idx])
    
    elif pattern_type == 'arpeggio':
        # アルペジオ（3度飛ばし）
        for i in range(num_notes):
            idx = (start_note_idx + i * 2) % len(scale)
            notes.append(scale[idx])
    
    elif pattern_type == 'wave':
        # 波型パターン（上下繰り返し）
        direction = 1
        idx = start_note_idx
        for i in range(num_notes):
            notes.append(scale[idx % len(scale)])
            idx += direction * (1 + (i % 2))
            if idx >= len(scale) - 1 or idx <= 0:
                direction *= -1
    
    else:  # 'random_walk'
        # ランダムウォーク（近隣の音に移動）
        idx = start_note_idx
        for i in range(num_notes):
            notes.append(scale[idx % len(scale)])
            step = np.random.choice([-2, -1, 0, 1, 2], p=[0.1, 0.3, 0.2, 0.3, 0.1])
            idx = np.clip(idx + step, 0, len(scale) - 1)
    
    return notes


def generate_motif(
    scale: np.ndarray,
    length: int = 4,
    rhythm_pattern: list = None
) -> dict:
    """
    モチーフ（短いメロディックパターン）を生成
    
    Args:
        scale: スケール音の配列
        length: モチーフの長さ（音符数）
        rhythm_pattern: リズムパターン（各音符の長さ、Noneの場合は自動生成）
    
    Returns:
        モチーフ情報（pitches, rhythms, velocities）
    """
    if rhythm_pattern is None:
        # リズムパターンのテンプレート
        rhythm_templates = [
            [1, 1, 2],
            [2, 1, 1],
            [1, 1, 1, 1],
            [2, 2],
            [1, 2, 1],
        ]
        rhythm_pattern = rhythm_templates[np.random.randint(len(rhythm_templates))]
    
    # リズムパターンを必要な長さに調整
    while sum(rhythm_pattern) < length:
        rhythm_pattern.append(1)
    
    # パターンタイプをランダムに選択
    pattern_types = ['ascending', 'descending', 'arpeggio', 'wave', 'random_walk']
    pattern_type = np.random.choice(pattern_types)
    
    # ピッチを生成
    pitches = generate_phrase_pattern(
        pattern_type=pattern_type,
        scale=scale,
        num_notes=len(rhythm_pattern)
    )
    
    # ベロシティに変化を付ける（強拍を強調）
    velocities = []
    for i in range(len(rhythm_pattern)):
        if i == 0:
            vel = np.random.uniform(0.7, 0.9)  # 開始音は強め
        elif i % 2 == 0:
            vel = np.random.uniform(0.6, 0.8)  # 偶数音符は中程度
        else:
            vel = np.random.uniform(0.5, 0.7)  # 奇数音符は弱め
        velocities.append(vel)
    
    return {
        'pitches': pitches,
        'rhythms': rhythm_pattern,
        'velocities': velocities,
        'pattern_type': pattern_type
    }


def develop_motif(motif: dict, development_type: str) -> dict:
    """
    モチーフを発展させる
    
    Args:
        motif: 元のモチーフ
        development_type: 発展タイプ ('repeat', 'transpose', 'invert', 'retrograde', 'vary_rhythm')
    
    Returns:
        発展させたモチーフ
    """
    if development_type == 'repeat':
        # そのまま繰り返し
        return motif.copy()
    
    elif development_type == 'transpose':
        # 移調（音程を上下に移動）
        semitone_ratio = 2 ** (1/12)
        shift = np.random.choice([-5, -3, -2, 2, 3, 5, 7])  # 半音単位
        new_pitches = [p * (semitone_ratio ** shift) for p in motif['pitches']]
        return {
            'pitches': new_pitches,
            'rhythms': motif['rhythms'].copy(),
            'velocities': motif['velocities'].copy(),
            'pattern_type': motif['pattern_type']
        }
    
    elif development_type == 'invert':
        # 反転（音程の上下を逆に）
        pitches = motif['pitches']
        center = pitches[0]
        new_pitches = [center * center / p for p in pitches]
        return {
            'pitches': new_pitches,
            'rhythms': motif['rhythms'].copy(),
            'velocities': motif['velocities'].copy(),
            'pattern_type': motif['pattern_type']
        }
    
    elif development_type == 'retrograde':
        # 逆行（後ろから演奏）
        return {
            'pitches': motif['pitches'][::-1],
            'rhythms': motif['rhythms'][::-1],
            'velocities': motif['velocities'][::-1],
            'pattern_type': motif['pattern_type']
        }
    
    elif development_type == 'vary_rhythm':
        # リズム変化
        new_rhythms = []
        for r in motif['rhythms']:
            if r >= 2 and np.random.rand() < 0.5:
                # 長い音を分割
                new_rhythms.extend([1, 1])
            else:
                new_rhythms.append(r)
        return {
            'pitches': motif['pitches'],
            'rhythms': new_rhythms,
            'velocities': motif['velocities'],
            'pattern_type': motif['pattern_type']
        }
    
    return motif


def create_phrase_structure(duration: float, bpm: float) -> dict:
    """
    フレーズ構造を作成（開始-展開-終止）
    
    Args:
        duration: フレーズの長さ（秒）
        bpm: テンポ
    
    Returns:
        フレーズ構造情報
    """
    sec_per_beat = 60.0 / bpm
    total_beats = duration / sec_per_beat
    
    # セクション分割（開始30%, 展開45%, 終止25%）
    intro_beats = total_beats * 0.30
    development_beats = total_beats * 0.45
    ending_beats = total_beats * 0.25
    
    return {
        'total_duration': duration,
        'bpm': bpm,
        'sec_per_beat': sec_per_beat,
        'sections': [
            {
                'name': 'intro',
                'start': 0.0,
                'duration': intro_beats * sec_per_beat,
                'beats': intro_beats,
                'density': 'medium',  # 音符密度
                'style': 'simple'     # 演奏スタイル
            },
            {
                'name': 'development',
                'start': intro_beats * sec_per_beat,
                'duration': development_beats * sec_per_beat,
                'beats': development_beats,
                'density': 'high',
                'style': 'complex'
            },
            {
                'name': 'ending',
                'start': (intro_beats + development_beats) * sec_per_beat,
                'duration': ending_beats * sec_per_beat,
                'beats': ending_beats,
                'density': 'low',
                'style': 'resolving'
            }
        ]
    }


def _pick_position_shaper(noise: np.ndarray, pick_pos_ratio: float) -> np.ndarray:
    """
    ピック位置を模したノッチ列（簡易）で初期ノイズを整形。
    pick_pos_ratio: (0, 1) 弦長に対するピック位置。0.2~0.6程度を想定。
    """
    n = len(noise)
    delay = max(1, int(round(pick_pos_ratio * n)))
    shaped = noise.copy()
    # x[n] - x[n-delay] に近い形でコーム的ノッチを作る
    if delay < n:
        shaped[delay:] -= 0.85 * noise[:-delay]
    # 軽く整形
    shaped = shaped * np.hanning(n)
    return shaped.astype(np.float32)


def karplus_strong_pluck(
    f0: float,
    duration: float,
    sample_rate: int = 48000,
    brightness: float = 0.6,
    pick_position: float = 0.3,
    damping_end: float = 0.85,
) -> np.ndarray:
    """
    カープラス・ストロングによる撥弦音合成（簡易・高安定版）。

    - 初期ノイズにピック位置整形
    - ループ内に時間変化ローパスを挿入して高域が減衰
    - 緩やかな分散（実質わずかな遅延補間の揺らぎ）
    """
    n_samples = int(duration * sample_rate)
    if n_samples <= 1:
        return np.zeros(1, dtype=np.float32)

    # 遅延長（分数遅延は線形補間）
    delay_exact = sample_rate / max(20.0, f0)
    delay_int = int(np.floor(delay_exact))
    frac = delay_exact - delay_int
    if delay_int < 2:
        delay_int = 2
        frac = 0.0

    # 初期バッファ（ピック位置で整形したノイズ）
    init = np.random.uniform(-1.0, 1.0, delay_int).astype(np.float32)
    init = _pick_position_shaper(init, np.clip(pick_position, 0.05, 0.9))
    buf = init.copy()

    y = np.zeros(n_samples, dtype=np.float32)
    y_prev = 0.0

    # 明るさ→ローパス係数へ写像
    # brightness=1.0で鋭く、=0で鈍い。係数は高いほどスムーズ（暗い）。
    lp_start = float(np.clip(1.0 - brightness, 0.05, 0.95))
    lp_end = float(np.clip(damping_end, 0.6, 0.98))  # 時間経過でさらに暗く

    # ごく浅い揺らぎ（分散の代替）
    jitter_depth = 0.02  # 遅延補間の揺らぎ（小）

    read_idx = 0  # 書き込みは先頭、読み出しは遅延相当後ろから
    for n in range(n_samples):
        # 時間に応じてローパス係数を線形変化
        t = n / n_samples
        lp_coeff = lp_start * (1.0 - t) + lp_end * t

        # 分数遅延の線形補間 + 微小ジッタ
        frac_jitter = frac + jitter_depth * (np.random.rand() * 2 - 1) * 0.01
        idx_a = (read_idx - delay_int) % delay_int
        idx_b = (idx_a - 1) % delay_int
        delayed = (1.0 - frac_jitter) * buf[idx_a] + frac_jitter * buf[idx_b]

        # 平均化ダンピング（KSの基本）
        avg = 0.5 * (buf[read_idx] + delayed)

        # 1次LPで高域を落とす
        out = _one_pole_lowpass(buf[read_idx], avg, y_prev, coeff=lp_coeff)
        y[n] = out
        y_prev = out

        # フィードバック（書き戻し）
        buf[read_idx] = out
        read_idx = (read_idx + 1) % delay_int

    return y


def generate_guitar_note(
    f0: float,
    duration: float,
    sample_rate: int = 48000,
    velocity: float = 0.8,
    pick_position: float = 0.3,
    with_pick_noise: bool = True,
    vibrato_depth: float = 0.0,
    vibrato_rate: float = 5.0,
    bend_target: float = None,
    bend_duration_ratio: float = 0.3,
    palm_mute: float = 0.0,
) -> np.ndarray:
    """
    撥弦音（単音）を生成。
    - KSプラック主体
    - ピック位置/ベロシティに応じて明るさやアタックを調整
    - 短いボディIRで色付け
    - ビブラート、ベンド、ミュート等の演奏表現に対応
    
    Args:
        f0: 基本周波数
        duration: 継続時間
        sample_rate: サンプリングレート
        velocity: ベロシティ（0-1）
        pick_position: ピック位置（0-1）
        with_pick_noise: ピッキングノイズを含むか
        vibrato_depth: ビブラートの深さ（セント単位、0で無効）
        vibrato_rate: ビブラートの速度（Hz）
        bend_target: ベンド先の周波数（Noneで無効）
        bend_duration_ratio: ベンドにかける時間の割合
        palm_mute: パームミュートの強さ（0-1）
    """
    velocity = float(np.clip(velocity, 0.0, 1.0))
    brightness = 0.4 + 0.55 * velocity
    damping_end = 0.82 + 0.12 * (1.0 - velocity)

    ks = karplus_strong_pluck(
        f0=f0,
        duration=duration,
        sample_rate=sample_rate,
        brightness=brightness,
        pick_position=pick_position,
        damping_end=damping_end,
    )

    n = len(ks)

    # ピッキングノイズ（可変）
    if with_pick_noise:
        pick_ms = 6 + int(6 * velocity)
        pick_samples = max(4, min(n, int(pick_ms * 1e-3 * sample_rate)))
        pick_noise = np.random.randn(pick_samples).astype(np.float32)
        pick_env = np.exp(-np.linspace(0, 12, pick_samples)).astype(np.float32)
        pick_color = _one_pole_lowpass(0.0, 1.0, 0.0, 0.2)  # ダミー呼び出しでmypy静音
        # 高域寄りにするため、DCブロック後に少しゲイン下げ
        pick_noise = _apply_dc_blocker(pick_noise) * pick_env * (0.2 + 0.4 * velocity)
        # 念のため長さを再確認（DC blockerの副作用対策）
        actual_pick = min(len(pick_noise), pick_samples, n)
        ks[:actual_pick] += pick_noise[:actual_pick]

    # ボディIR
    body_ir = _make_body_ir(sample_rate)
    ks = _convolve_short_ir(ks, body_ir)
    # 畳み込み後は長さが変わるので元の長さに切り詰め
    ks = ks[:n]

    # 軽いエンベロープ（アタック短/ディケイ中/リリース短）
    attack_time = max(0.002, 0.006 - 0.004 * velocity)
    decay_time = 0.18 + 0.12 * (1.0 - velocity)
    sustain_level = 0.5 + 0.3 * velocity
    release_time = 0.25

    n_samples = len(ks)
    attack = min(int(attack_time * sample_rate), max(1, n_samples // 6))
    decay = min(int(decay_time * sample_rate), max(1, n_samples // 4))
    release = min(int(release_time * sample_rate), max(1, n_samples // 3))
    sustain = max(0, n_samples - attack - decay - release)

    env = np.ones(n_samples, dtype=np.float32)
    if attack > 0:
        env[:attack] = np.linspace(0, 1, attack, endpoint=False) ** 0.5
    if decay > 0:
        start = attack
        end = min(n_samples, start + decay)
        seg = end - start
        if seg > 0:
            curve = np.linspace(0, 1, seg)
            env[start:end] = 1.0 - (1.0 - sustain_level) * (curve ** 1.7)
    if sustain > 0:
        s0 = attack + decay
        s1 = min(n_samples - release, s0 + sustain)
        if s1 > s0:
            env[s0:s1] = sustain_level
    if release > 0 and n_samples > release:
        tail = np.linspace(1.0, 0.0, release)
        env[-release:] *= tail ** 2

    ks *= env

    # DC除去を先に実施
    ks = _apply_dc_blocker(ks)
    
    # エフェクト前の正規化
    peak = float(np.max(np.abs(ks)) + 1e-8)
    target_gain = 0.7 * velocity + 0.3 * (1.0 - velocity)  # より控えめなゲイン
    ks = (ks / peak) * target_gain
    
    # 演奏表現エフェクトを適用
    
    # ピッチベンドを適用（指定された場合）
    if bend_target is not None and bend_target != f0:
        ks = apply_pitch_bend(
            ks, 
            f0_start=f0, 
            f0_end=bend_target,
            sample_rate=sample_rate,
            bend_duration_ratio=bend_duration_ratio
        )
    
    # ビブラートを適用（指定された場合）
    if vibrato_depth > 0:
        ks = apply_vibrato(
            ks,
            f0=bend_target if bend_target is not None else f0,
            sample_rate=sample_rate,
            depth_cents=vibrato_depth,
            rate_hz=vibrato_rate,
            fade_in_sec=0.1
        )
    
    # パームミュートを適用（指定された場合）
    if palm_mute > 0:
        ks = apply_palm_mute(
            ks,
            mute_amount=palm_mute,
            damping_freq=800.0,
            sample_rate=sample_rate
        )
    
    # エフェクト後の安全正規化（クリッピング防止）
    peak_after = float(np.max(np.abs(ks)) + 1e-8)
    if peak_after > 0.85:  # エフェクトでゲインが増えた場合
        ks = (ks / peak_after) * 0.85
    
    # 要求されたdurationに正確に合わせる
    requested_samples = int(duration * sample_rate)
    if len(ks) > requested_samples:
        ks = ks[:requested_samples]
    elif len(ks) < requested_samples:
        ks = np.pad(ks, (0, requested_samples - len(ks)), mode='constant')
    
    return ks.astype(np.float32)


def legacy_generate_guitar_phrase(duration: float, sample_rate: int = 48000) -> np.ndarray:
    """
    旧ロジック: モチーフ/セクションベースの汎用フレーズ生成。
    """
    n_samples = int(duration * sample_rate)
    audio = np.zeros(n_samples, dtype=np.float32)
    
    bpm = float(np.random.uniform(85, 130))
    sec_per_beat = 60.0 / bpm
    step_sec = sec_per_beat / 4.0
    
    root_freq = np.random.choice([82.41, 110.0, 146.83])
    scale_types = ['minor_pentatonic', 'blues', 'natural_minor']
    scale_type = np.random.choice(scale_types)
    scale = create_scale_from_root(root_freq, scale_type)
    
    structure = create_phrase_structure(duration, bpm)
    play_styles = ['legato', 'staccato', 'mixed', 'riff']
    play_style = np.random.choice(play_styles)
    
    swing = np.random.uniform(0.0, 0.15)
    
    main_motif = generate_motif(scale, length=4)
    development_types = ['repeat', 'transpose', 'invert', 'retrograde', 'vary_rhythm']
    
    current_time = 0.0
    last_end = 0
    last_pitch = None
    
    for section in structure['sections']:
        section_end = section['start'] + section['duration']
        density = section['density']
        
        if section['name'] == 'intro':
            current_motif = main_motif
            technique_probs = {'normal': 0.85, 'vibrato': 0.08, 'bend': 0.04, 'slide': 0.02, 'mute': 0.01}
        elif section['name'] == 'development':
            dev_type = np.random.choice(development_types)
            current_motif = develop_motif(main_motif, dev_type)
            technique_probs = {'normal': 0.70, 'vibrato': 0.15, 'bend': 0.08, 'slide': 0.04, 'mute': 0.03}
        else:
            current_motif = main_motif
            technique_probs = {'normal': 0.75, 'vibrato': 0.12, 'bend': 0.07, 'slide': 0.03, 'mute': 0.03}
        
        motif_idx = 0
        while current_time < section_end and current_time < duration:
            rest_prob = {'low': 0.3, 'medium': 0.15, 'high': 0.08}.get(density, 0.15)
            if np.random.rand() < rest_prob:
                current_time += step_sec * np.random.choice([1, 2])
                continue
            
            if motif_idx >= len(current_motif['pitches']):
                if np.random.rand() < 0.4:
                    dev_type = np.random.choice(development_types)
                    current_motif = develop_motif(main_motif, dev_type)
                motif_idx = 0
            
            f0 = current_motif['pitches'][motif_idx]
            rhythm_len = current_motif['rhythms'][motif_idx]
            velocity = current_motif['velocities'][motif_idx]
            
            if play_style == 'staccato':
                note_duration = rhythm_len * step_sec * 0.6
                with_pick = True
            elif play_style == 'legato':
                note_duration = rhythm_len * step_sec * 1.2
                with_pick = (motif_idx == 0)
            elif play_style == 'riff':
                note_duration = rhythm_len * step_sec * 0.8
                with_pick = True
                velocity *= 0.9
            else:
                note_duration = rhythm_len * step_sec
                with_pick = (np.random.rand() < 0.7)
            
            step_count = int(current_time / step_sec)
            timing_offset = 0.0
            if step_count % 2 == 1:
                timing_offset += step_sec * swing
            timing_offset += np.random.uniform(-0.008, 0.008)
            
            start_time = current_time + timing_offset
            start_sample = int(start_time * sample_rate)
            if start_sample >= n_samples:
                break
            
            technique = np.random.choice(list(technique_probs.keys()), p=list(technique_probs.values()))

            vibrato_depth = 0.0
            vibrato_rate = 5.0
            bend_target = None
            palm_mute = 0.0
            use_slide = False
            
            if technique == 'vibrato' and note_duration > 0.3:
                vibrato_depth = np.random.uniform(15.0, 35.0)  # より控えめな深さ
                vibrato_rate = np.random.uniform(4.5, 6.5)
            elif technique == 'bend':
                semitone_ratio = 2 ** (1/12)
                bend_amount = np.random.choice([1, 2])
                bend_target = f0 * (semitone_ratio ** bend_amount)
            elif technique == 'slide' and last_pitch is not None:
                use_slide = True
            elif technique == 'mute':
                palm_mute = np.random.uniform(0.5, 0.8)
            
            if use_slide and last_pitch is not None:
                note = generate_slide_note(last_pitch, f0, note_duration, sample_rate, velocity * 0.85, 0.5)
            else:
                pick_pos = float(np.clip(np.random.normal(0.3, 0.1), 0.15, 0.6))
                note = generate_guitar_note(
                    f0=f0,
                    duration=note_duration,
                    sample_rate=sample_rate,
                    velocity=velocity,
                    pick_position=pick_pos,
                    with_pick_noise=with_pick,
                    vibrato_depth=vibrato_depth,
                    vibrato_rate=vibrato_rate,
                    bend_target=bend_target,
                    bend_duration_ratio=0.3,
                    palm_mute=palm_mute
                )
            
            note = note * 0.65
            if start_sample < 0 or start_sample >= n_samples:
                current_time += rhythm_len * step_sec
                motif_idx += 1
                continue
            
            end_sample = min(start_sample + len(note), n_samples)
            note_len = end_sample - start_sample
            if note_len > 0 and end_sample > start_sample:
                overlap = max(0, last_end - start_sample)
                if overlap > 0 and play_style == 'legato':
                    xf = min(overlap, int(0.008 * sample_rate), note_len)
                    if xf > 1 and start_sample + xf <= n_samples:
                        t = np.linspace(0, 1, xf).astype(np.float32)
                        fade_out = 1.0 - (t ** 2 * (3 - 2 * t))
                        fade_in = t ** 2 * (3 - 2 * t)
                        audio[start_sample:start_sample + xf] *= fade_out
                        note[:xf] *= fade_in
                audio[start_sample:end_sample] += note[:note_len]
                last_end = end_sample
            
            last_pitch = bend_target if bend_target is not None else f0
            current_time += rhythm_len * step_sec
            motif_idx += 1
    
    audio = _apply_dc_blocker(audio)
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(np.abs(audio)) + 1e-8)
    if peak > 0.001:
        target_level = 0.7
        audio = (audio / peak) * target_level
    audio = np.tanh(audio * 1.2) * 0.85
    return audio.astype(np.float32)


# ===============
# ブルース/ロック拡張
# ===============
def _key_to_root_freq(key: str) -> float:
    key = (key or 'E').upper()
    mapping = {
        'E': 82.41,  # E2
        'A': 110.00, # A2
        'G': 98.00,  # G2
        'D': 73.42,  # D2
        'C': 65.41,  # C2
        'B': 61.74,  # B1
        'F': 87.31,  # F2
    }
    return mapping.get(key, 82.41)


def schedule_blues_regions(duration: float, bpm: float, key_root_freq: float) -> list:
    """
    12小節ブルース(I7 IV7 V7)から時間長に収まる範囲で区間を切り出す。
    戻り値: [{'chord':'I7','start':t0,'end':t1,'root':freq}, ...]
    """
    sec_per_beat = 60.0 / bpm
    bar_sec = 4.0 * sec_per_beat
    pattern = ['I7','I7','I7','I7', 'IV7','IV7','I7','I7', 'V7','IV7','I7','V7']
    # 常に最初から開始（より自然な進行）
    total_bars = max(1, int(np.ceil(duration / bar_sec)))
    start_bar = 0  # 常に最初から開始
    regions = []
    t = 0.0
    for i in range(total_bars):
        chord = pattern[(start_bar + i) % len(pattern)]
        start = t
        end = min(duration, t + bar_sec)
        if chord == 'I7':
            root = key_root_freq
        elif chord == 'IV7':
            root = key_root_freq * (2 ** (5/12))
        else:  # 'V7'
            root = key_root_freq * (2 ** (7/12))
        regions.append({'chord': chord, 'start': start, 'end': end, 'root': float(root)})
        t += bar_sec
        if t >= duration:
            break
    return regions


def sample_chord_aware_pitch(chord_region: dict, scale: np.ndarray) -> float:
    """
    コードトーン重みでスケールから周波数をサンプル。
    ルート/3rd/5th/b7 を高重み、ブルーノート(b5)にボーナス。
    """
    chord_root = chord_region['root']
    chord_tones = np.array(get_chord_tones('dominant7', chord_root), dtype=np.float32)
    # b5 を推定（ルートから6半音）
    blue_note = chord_root * (2 ** (6/12))

    # 距離をセントで計算して重み付け
    eps = 1e-9
    weights = []
    for f in scale:
        # 近いコードトーンまでの最小距離
        cents = np.min(np.abs(1200.0 * np.log2((f + eps) / (chord_tones + eps))))
        w = np.exp(- (cents / 35.0) ** 2) + 0.03  # ベースライン
        # ブルーノート近傍ブースト
        cents_blue = np.abs(1200.0 * np.log2((f + eps) / (blue_note + eps)))
        if cents_blue < 25.0:
            w += 0.2
        weights.append(w)
    weights = np.array(weights, dtype=np.float32)
    weights_sum = float(np.sum(weights))
    if weights_sum <= 0:
        return float(scale[len(scale)//2])
    probs = weights / weights_sum
    idx = int(np.random.choice(np.arange(len(scale)), p=probs))
    return float(scale[idx])


def make_shuffle_grid(sec_per_beat: float, duration: float, shuffle_ratio: float = 2.0/3.0) -> tuple:
    """
    三連系シャッフルの簡易グリッドを返す（onsets[sec], durations[sec]）。
    1拍を (shuffle_ratio, 1-shuffle_ratio) に二分する8分基準。
    """
    onsets = []
    durs = []
    t = 0.0
    while t < duration:
        # オン（長い方）
        onsets.append(t)
        durs.append(sec_per_beat * shuffle_ratio)
        t_next = t + sec_per_beat * shuffle_ratio
        if t_next >= duration:
            break
        # オフ（短い方）
        onsets.append(t_next)
        durs.append(sec_per_beat * (1.0 - shuffle_ratio))
        t = t + sec_per_beat
    return np.array(onsets, dtype=np.float32), np.array(durs, dtype=np.float32)


def render_hopo(base_f0: float, target_f0: float, duration: float, sample_rate: int, velocity: float) -> np.ndarray:
    """
    ハンマリング/プリング（無ピック2音）
    """
    d1 = max(0.05, duration * 0.45)
    d2 = max(0.04, duration - d1)
    a = generate_guitar_note(base_f0, d1, sample_rate, velocity*0.9, with_pick_noise=False)
    b = generate_guitar_note(target_f0, d2, sample_rate, velocity*0.95, with_pick_noise=False)
    out = np.concatenate([a, b])
    if len(out) > int(duration * sample_rate):
        out = out[:int(duration * sample_rate)]
    elif len(out) < int(duration * sample_rate):
        out = np.pad(out, (0, int(duration * sample_rate) - len(out)))
    return out.astype(np.float32)


def render_double_stop(f0_a: float, f0_b: float, duration: float, sample_rate: int, velocity: float) -> np.ndarray:
    a = generate_guitar_note(f0_a, duration, sample_rate, velocity, with_pick_noise=True)
    b = generate_guitar_note(f0_b, duration, sample_rate, velocity*0.9, with_pick_noise=False)
    mix = 0.7 * (a + b)
    peak = float(np.max(np.abs(mix)) + 1e-8)
    if peak > 0.95:
        mix = mix / peak * 0.9
    return mix.astype(np.float32)


def render_prebend_release(base_f0: float, duration: float, sample_rate: int, velocity: float) -> np.ndarray:
    """
    プレベンドしてリリース（上からベース音へ滑り降りる）。
    簡易実装: 半音/全音上から base へスライド。
    """
    semitone_ratio = 2 ** (1/12)
    up = int(np.random.choice([1, 2]))
    start_f0 = base_f0 * (semitone_ratio ** up)
    return generate_slide_note(start_f0, base_f0, duration, sample_rate, velocity*0.95, 0.5)


def generate_guitar_phrase(
    duration: float,
    sample_rate: int = 48000,
    style: str = 'blues_rock',
    bpm: float = None,
    key: str = 'E',
    seed: int = None,
    enable_legacy: bool = True,
) -> np.ndarray:
    """
    ブルース/ロック指向のフレーズ生成（デフォルト）。
    enable_legacy=True で旧ロジックを使用。
    """
    if enable_legacy or (style and style != 'blues_rock'):
        return legacy_generate_guitar_phrase(duration, sample_rate)

    if seed is not None:
        try:
            np.random.seed(int(seed))
        except Exception:
            pass

    n_samples = int(duration * sample_rate)
    audio = np.zeros(n_samples, dtype=np.float32)

    bpm_val = float(bpm) if bpm is not None else float(np.random.uniform(88.0, 120.0))
    sec_per_beat = 60.0 / bpm_val
    step_sec = sec_per_beat / 4.0  # 16分基準（後でシャッフル補正）

    root_freq = _key_to_root_freq(key)
    scale = create_scale_from_root(root_freq, 'blues')

    # コード進行の区間（I/IV/V）
    regions = schedule_blues_regions(duration, bpm_val, root_freq)

    # 密度と構造は従来の3部構成を簡易活用
    structure = create_phrase_structure(duration, bpm_val)

    # シャッフルオフセット: オフビートの8分を 1/6 拍だけ後ろへ
    def shuffle_timing_offset(cur_t: float) -> float:
        eighth = sec_per_beat / 2.0
        idx = int(cur_t / eighth)
        is_off = (idx % 2 == 1)
        base = (sec_per_beat / 6.0) if is_off else 0.0
        human = float(np.random.uniform(-0.006, 0.006))
        return base + human

    # シャッフル長さ: オン=2/3拍, オフ=1/3拍（近似的に8分相当長のみに適用）
    def shuffle_duration(nominal: float, start_t: float) -> float:
        eighth = sec_per_beat / 2.0
        idx = int(start_t / eighth)
        is_off = (idx % 2 == 1)
        if 0.45 * sec_per_beat <= nominal <= 0.8 * sec_per_beat:
            return (sec_per_beat * (1.0/3.0 if is_off else 2.0/3.0))
        return nominal

    # セクション内で配置
    current_time = 0.0
    last_end = 0
    last_pitch = None

    # メインモチーフ（ベロシティ/リズム用）
    motif = generate_motif(scale, length=4)
    development_types = ['repeat', 'transpose', 'invert', 'retrograde', 'vary_rhythm']
    motif_idx = 0

    # テクニック確率（ブルース/ロック）- より自然な分布に調整
    base_technique_probs = {'normal': 0.80, 'vibrato': 0.08, 'bend': 0.05, 'slide': 0.02, 'mute': 0.02, 'hopo': 0.02, 'double': 0.01, 'prebend': 0.0}
    
    # フレーズの開始を穏やかに（イントロ期間）
    intro_duration = min(0.5, duration * 0.15)
    # フレーズの終わりにフェードアウト期間
    outro_duration = min(0.8, duration * 0.2)
    outro_start = duration - outro_duration

    def chord_region_at(t: float) -> dict:
        for r in regions:
            if r['start'] <= t < r['end']:
                return r
        return regions[-1]

    for section in structure['sections']:
        section_end = section['start'] + section['duration']
        density = section['density']
        
        # セクションごとのテクニック確率を調整
        if section['name'] == 'intro':
            # イントロはシンプルに
            technique_probs = {'normal': 0.90, 'vibrato': 0.05, 'bend': 0.02, 'slide': 0.01, 'mute': 0.01, 'hopo': 0.01, 'double': 0.0, 'prebend': 0.0}
        elif section['name'] == 'development':
            # 展開部はバリエーション豊富に
            technique_probs = {'normal': 0.70, 'vibrato': 0.12, 'bend': 0.08, 'slide': 0.03, 'mute': 0.03, 'hopo': 0.03, 'double': 0.01, 'prebend': 0.0}
        else:  # ending
            # エンディングは落ち着いて
            technique_probs = {'normal': 0.85, 'vibrato': 0.10, 'bend': 0.03, 'slide': 0.01, 'mute': 0.01, 'hopo': 0.0, 'double': 0.0, 'prebend': 0.0}
        
        while current_time < section_end and current_time < duration:
            # 休符確率（密度で調整）
            rest_prob = {'low': 0.28, 'medium': 0.14, 'high': 0.08}.get(density, 0.14)
            
            # イントロとアウトロでは休符を増やす
            if current_time < intro_duration:
                rest_prob *= 1.5
            elif current_time > outro_start:
                rest_prob *= 2.0
                
            if np.random.rand() < rest_prob:
                current_time += step_sec * np.random.choice([1, 2])
                continue

            # モチーフ進行とリズム/ダイナミクス取得
            if motif_idx >= len(motif['rhythms']) or motif_idx >= len(motif['velocities']):
                # セクションに応じて展開確率を調整
                if section['name'] == 'intro':
                    # イントロは同じモチーフを繰り返す
                    if np.random.rand() < 0.2:
                        dev_type = 'repeat'
                        motif = develop_motif(motif, dev_type)
                elif section['name'] == 'development':
                    # 展開部はバリエーションを増やす
                    if np.random.rand() < 0.6:
                        # より音楽的な展開を優先
                        dev_type = np.random.choice(['repeat', 'transpose', 'vary_rhythm'], p=[0.3, 0.5, 0.2])
                        motif = develop_motif(motif, dev_type)
                else:  # ending
                    # エンディングは安定したパターンに
                    if np.random.rand() < 0.3:
                        dev_type = np.random.choice(['repeat', 'transpose'], p=[0.7, 0.3])
                        motif = develop_motif(motif, dev_type)
                motif_idx = 0
            rhythm_len = motif['rhythms'][motif_idx]
            velocity = motif['velocities'][motif_idx]
            
            # イントロとアウトロでベロシティを調整
            if current_time < intro_duration:
                # イントロは徐々に音量を上げる
                intro_ratio = current_time / intro_duration
                velocity *= (0.5 + 0.5 * intro_ratio)
            elif current_time > outro_start:
                # アウトロは徐々に音量を下げる
                outro_ratio = (duration - current_time) / outro_duration
                velocity *= (0.3 + 0.7 * outro_ratio)

            # コードに応じた音高選択
            region = chord_region_at(current_time)
            f0 = sample_chord_aware_pitch(region, scale)

            # シャッフル時間補正
            timing_offset = shuffle_timing_offset(current_time)
            start_time = current_time + timing_offset
            if start_time < 0:
                start_time = current_time
            start_sample = int(start_time * sample_rate)
            if start_sample >= n_samples:
                break

            # 名目上の音長とスタイル適用
            nominal = rhythm_len * step_sec
            note_duration = shuffle_duration(nominal, start_time)
            
            # アウトロでリタルダンド（テンポを落とす）
            if current_time > outro_start:
                ritardando_ratio = 1.0 + (1.0 - (duration - current_time) / outro_duration) * 0.3
                note_duration *= ritardando_ratio

            # スタイル軽微調整（セクションに応じて）
            if section['name'] == 'intro':
                # イントロはシンプルで明確に
                play_styles = ['mixed', 'staccato']
                play_style = np.random.choice(play_styles)
            elif section['name'] == 'development':
                # 展開部はバリエーション豊富に
                play_styles = ['legato', 'staccato', 'mixed', 'riff']
                play_style = np.random.choice(play_styles)
            else:  # ending
                # エンディングは落ち着いて
                play_styles = ['legato', 'mixed']
                play_style = np.random.choice(play_styles)
            
            if play_style == 'staccato':
                note_duration *= 0.7
                with_pick = True
            elif play_style == 'legato':
                note_duration *= 1.15
                with_pick = (motif_idx == 0)
            elif play_style == 'riff':
                note_duration *= 0.9
                with_pick = True
                velocity *= 0.92
            else:
                with_pick = (np.random.rand() < 0.7)

            # テクニック選択
            tech = np.random.choice(list(technique_probs.keys()), p=list(technique_probs.values()))

            note = None
            if tech == 'hopo' and note_duration > 0.12:
                # 半音/全音のHOPO
                semitone_ratio = 2 ** (1/12)
                up = np.random.choice([1, 2])
                target = f0 * (semitone_ratio ** up)
                note = render_hopo(f0, target, note_duration, sample_rate, velocity)
            elif tech == 'double' and note_duration > 0.12:
                # ダブルストップ: 5度上 or 6度（ブルース響き）
                fifth = f0 * (2 ** (7/12))
                sixth = f0 * (2 ** (9/12))
                other = sixth if np.random.rand() < 0.5 else fifth
                note = render_double_stop(f0, other, note_duration, sample_rate, velocity)
            elif tech == 'prebend' and note_duration > 0.12:
                note = render_prebend_release(f0, note_duration, sample_rate, velocity)
            else:
                vibrato_depth = 0.0
                vibrato_rate = 5.0
                bend_target = None
                palm_mute = 0.0
                use_slide = False

                if tech == 'vibrato' and note_duration > 0.25:
                    vibrato_depth = np.random.uniform(15.0, 35.0)  # より控えめな深さ
                    vibrato_rate = np.random.uniform(4.5, 6.5)
                elif tech == 'bend':
                    semitone_ratio = 2 ** (1/12)
                    amount = np.random.choice([1, 2])
                    bend_target = f0 * (semitone_ratio ** amount)
                elif tech == 'slide' and last_pitch is not None:
                    use_slide = True
                elif tech == 'mute':
                    palm_mute = np.random.uniform(0.5, 0.8)

                if use_slide and last_pitch is not None:
                    note = generate_slide_note(last_pitch, f0, note_duration, sample_rate, velocity * 0.85, 0.5)
                else:
                    pick_pos = float(np.clip(np.random.normal(0.3, 0.1), 0.15, 0.6))
                    note = generate_guitar_note(
                        f0=f0,
                        duration=note_duration,
                        sample_rate=sample_rate,
                        velocity=velocity,
                        pick_position=pick_pos,
                        with_pick_noise=with_pick,
                        vibrato_depth=vibrato_depth,
                        vibrato_rate=vibrato_rate,
                        bend_target=bend_target,
                        bend_duration_ratio=0.3,
                        palm_mute=palm_mute,
                    )

            note = (note * 0.65).astype(np.float32)

            if start_sample < 0 or start_sample >= n_samples:
                current_time += rhythm_len * step_sec
                motif_idx += 1
                continue

            end_sample = min(start_sample + len(note), n_samples)
            note_len = end_sample - start_sample
            if note_len > 0 and end_sample > start_sample:
                overlap = max(0, last_end - start_sample)
                if overlap > 0 and play_style == 'legato':
                    xf = min(overlap, int(0.008 * sample_rate), note_len)
                    if xf > 1 and start_sample + xf <= n_samples:
                        t = np.linspace(0, 1, xf).astype(np.float32)
                        fade_out = 1.0 - (t ** 2 * (3 - 2 * t))
                        fade_in = t ** 2 * (3 - 2 * t)
                        audio[start_sample:start_sample + xf] *= fade_out
                        note[:xf] *= fade_in
                audio[start_sample:end_sample] += note[:note_len]
                last_end = end_sample

            last_pitch = f0
            current_time += rhythm_len * step_sec
            motif_idx += 1

    # 終止：可能ならトニックへ解決（軽いターンアラウンド）
    try:
        resolve_dur = min(0.4, duration * 0.25)
        if resolve_dur > 0.08:
            resolve_start = max(0, n_samples - int(resolve_dur * sample_rate))
            end_sample = n_samples
            # より穏やかな終止音
            base_note = generate_guitar_note(root_freq, resolve_dur, sample_rate, velocity=0.6, with_pick_noise=False, vibrato_depth=12.0, vibrato_rate=5.0)
            xf = min(int(0.015 * sample_rate), end_sample - resolve_start, len(base_note))
            if xf > 1:
                t = np.linspace(0, 1, xf).astype(np.float32)
                fade_out = 1.0 - (t ** 2 * (3 - 2 * t))
                audio[resolve_start:resolve_start + xf] *= fade_out
                base_note[:xf] *= t ** 2 * (3 - 2 * t)
            sl = min(len(base_note), end_sample - resolve_start)
            if sl > 0:
                audio[resolve_start:resolve_start + sl] += (base_note[:sl] * 0.7)
    except Exception:
        pass
    
    # アウトロ部分に全体的なフェードアウトを適用
    outro_fade_start = int(outro_start * sample_rate)
    if outro_fade_start < n_samples:
        fade_length = n_samples - outro_fade_start
        fade_curve = np.linspace(1.0, 0.15, fade_length).astype(np.float32)
        # 滑らかな減衰カーブ
        fade_curve = fade_curve ** 1.5
        audio[outro_fade_start:] *= fade_curve
    
    # 既存仕上げ処理
    audio = _apply_dc_blocker(audio)
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(np.abs(audio)) + 1e-8)
    if peak > 0.001:
        audio = (audio / peak) * 0.7
    audio = np.tanh(audio * 1.2) * 0.85
    return audio.astype(np.float32)


def create_dummy_dataset(
    output_dir: str,
    n_train: int = 100,
    n_val: int = 20,
    duration: float = 4.0,
    sample_rate: int = 48000
):
    """
    ダミーデータセットを作成
    
    Args:
        output_dir: 出力ディレクトリ
        n_train: トレーニングサンプル数
        n_val: 検証サンプル数
        duration: 各サンプルの継続時間（秒）
        sample_rate: サンプリングレート
    """
    output_path = Path(output_dir)
    
    # ディレクトリ構造を作成
    for split in ['train', 'val']:
        for subdir in ['audio']:
            (output_path / split / subdir).mkdir(parents=True, exist_ok=True)
    
    print(f"ダミーデータセットを作成中: {output_dir}")
    print(f"トレーニングサンプル: {n_train}, 検証サンプル: {n_val}")
    print(f"サンプル長: {duration}秒, サンプリングレート: {sample_rate}Hz")
    
    # トレーニングデータを生成
    print("\nトレーニングデータを生成中...")
    for i in tqdm(range(n_train)):
        audio = generate_guitar_phrase(duration, sample_rate)
        audio_path = output_path / 'train' / 'audio' / f'sample_{i:04d}.wav'
        sf.write(audio_path, audio, sample_rate)
    
    # 検証データを生成
    print("検証データを生成中...")
    for i in tqdm(range(n_val)):
        audio = generate_guitar_phrase(duration, sample_rate)
        audio_path = output_path / 'val' / 'audio' / f'sample_{i:04d}.wav'
        sf.write(audio_path, audio, sample_rate)
    
    print(f"\n完了！データセットは {output_dir} に保存されました。")
    print("\nデータセット構造:")
    print(f"{output_dir}/")
    print(f"├── train/")
    print(f"│   └── audio/")
    print(f"│       ├── sample_0000.wav")
    print(f"│       └── ...")
    print(f"└── val/")
    print(f"    └── audio/")
    print(f"        ├── sample_0000.wav")
    print(f"        └── ...")


def main():
    parser = argparse.ArgumentParser(
        description='DDSPギター用のダミーデータセットを生成します'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='dataset',
        help='出力ディレクトリのパス（デフォルト: dataset）'
    )
    parser.add_argument(
        '--n_train',
        type=int,
        default=100,
        help='トレーニングサンプル数（デフォルト: 100）'
    )
    parser.add_argument(
        '--n_val',
        type=int,
        default=20,
        help='検証サンプル数（デフォルト: 20）'
    )
    parser.add_argument(
        '--duration',
        type=float,
        default=4.0,
        help='各サンプルの継続時間（秒、デフォルト: 4.0）'
    )
    parser.add_argument(
        '--sample_rate',
        type=int,
        default=48000,
        help='サンプリングレート（Hz、デフォルト: 48000）'
    )
    
    args = parser.parse_args()
    
    create_dummy_dataset(
        output_dir=args.output_dir,
        n_train=args.n_train,
        n_val=args.n_val,
        duration=args.duration,
        sample_rate=args.sample_rate
    )


if __name__ == '__main__':
    main()

