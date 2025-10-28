#!/usr/bin/env python3
"""
DDSP Guitar - トレーニングスクリプト

datasetディレクトリのデータを使ってモデルを学習します。
"""

import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime

from ddsp_guitar.model import GuitarDDSP
from ddsp_guitar.losses.msstft import MultiScaleSTFTLoss
from ddsp_guitar.losses.transient import TransientLoss
from ddsp_guitar.data.dataset import GuitarDIDataset


def parse_args():
    parser = argparse.ArgumentParser(description='DDSP Guitar モデルのトレーニング')
    
    # データセット設定
    parser.add_argument('--train_dir', type=str, default='dataset/train/audio',
                        help='トレーニングデータのディレクトリ')
    parser.add_argument('--val_dir', type=str, default='dataset/val/audio',
                        help='検証データのディレクトリ')
    parser.add_argument('--sample_rate', type=int, default=48000,
                        help='サンプリングレート')
    parser.add_argument('--segment_seconds', type=float, default=1.0,
                        help='セグメント長（秒）')
    
    # モデル設定
    parser.add_argument('--num_harm', type=int, default=64,
                        help='ハーモニクス数')
    parser.add_argument('--num_noise', type=int, default=8,
                        help='ノイズバンド数')
    
    # トレーニング設定
    parser.add_argument('--batch_size', type=int, default=4,
                        help='バッチサイズ')
    parser.add_argument('--epochs', type=int, default=100,
                        help='エポック数')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学習率')
    parser.add_argument('--weight_decay', type=float, default=1e-6,
                        help='Weight decay')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='勾配クリッピング')
    
    # 損失関数の重み
    parser.add_argument('--loss_stft_weight', type=float, default=1.0,
                        help='STFT損失の重み')
    parser.add_argument('--loss_l1_weight', type=float, default=0.5,
                        help='L1損失の重み')
    parser.add_argument('--loss_transient_weight', type=float, default=0.5,
                        help='トランジェント損失の重み')
    
    # チェックポイント設定
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='チェックポイント保存ディレクトリ')
    parser.add_argument('--save_every', type=int, default=10,
                        help='N エポックごとにチェックポイントを保存')
    parser.add_argument('--resume', type=str, default=None,
                        help='再開するチェックポイントのパス')
    
    # ログ設定
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='TensorBoardログディレクトリ')
    parser.add_argument('--log_every', type=int, default=10,
                        help='N ステップごとにログを記録')
    
    # その他
    parser.add_argument('--num_workers', type=int, default=2,
                        help='DataLoaderのワーカー数')
    parser.add_argument('--device', type=str, default='cuda',
                        help='デバイス (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42,
                        help='ランダムシード')
    
    return parser.parse_args()


def set_seed(seed):
    """再現性のためのシード設定"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)


def save_checkpoint(model, optimizer, scheduler, epoch, global_step, loss, checkpoint_path):
    """チェックポイントを保存"""
    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"チェックポイントを保存: {checkpoint_path}")


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """チェックポイントを読み込み"""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1
    global_step = checkpoint['global_step']
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"チェックポイントを読み込み: {checkpoint_path}")
    print(f"エポック {start_epoch} から再開")
    
    return start_epoch, global_step


def train_epoch(model, dataloader, optimizer, scheduler, losses, device, args, writer, global_step, epoch):
    """1エポックのトレーニング"""
    model.train()
    epoch_losses = {'total': 0.0, 'stft': 0.0, 'l1': 0.0, 'transient': 0.0}
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch_idx, (x, f0, loud, y) in enumerate(pbar):
        # デバイスに転送
        x = x.to(device)
        f0 = f0.to(device)
        loud = loud.to(device)
        y = y.to(device)
        
        # フォワードパス
        y_pred = model(x, f0, loud)
        
        # 損失計算
        loss_stft = losses['stft'](y_pred, y)
        loss_l1 = losses['l1'](y_pred, y)
        loss_transient = losses['transient'](y_pred, y)
        
        loss = (args.loss_stft_weight * loss_stft + 
                args.loss_l1_weight * loss_l1 + 
                args.loss_transient_weight * loss_transient)
        
        # バックワード
        optimizer.zero_grad()
        loss.backward()
        
        # 勾配クリッピング
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        
        optimizer.step()
        
        # 損失を記録
        epoch_losses['total'] += loss.item()
        epoch_losses['stft'] += loss_stft.item()
        epoch_losses['l1'] += loss_l1.item()
        epoch_losses['transient'] += loss_transient.item()
        
        # TensorBoard ログ
        if global_step % args.log_every == 0:
            writer.add_scalar('Train/Loss_Total', loss.item(), global_step)
            writer.add_scalar('Train/Loss_STFT', loss_stft.item(), global_step)
            writer.add_scalar('Train/Loss_L1', loss_l1.item(), global_step)
            writer.add_scalar('Train/Loss_Transient', loss_transient.item(), global_step)
            writer.add_scalar('Train/LR', optimizer.param_groups[0]['lr'], global_step)
        
        # プログレスバー更新
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'stft': f'{loss_stft.item():.4f}',
            'l1': f'{loss_l1.item():.4f}',
            'trans': f'{loss_transient.item():.4f}'
        })
        
        global_step += 1
    
    # スケジューラのステップ
    if scheduler:
        scheduler.step()
    
    # エポックの平均損失
    n_batches = len(dataloader)
    for key in epoch_losses:
        epoch_losses[key] /= n_batches
    
    return epoch_losses, global_step


@torch.no_grad()
def validate(model, dataloader, losses, device, args):
    """検証"""
    model.eval()
    val_losses = {'total': 0.0, 'stft': 0.0, 'l1': 0.0, 'transient': 0.0}
    
    for x, f0, loud, y in tqdm(dataloader, desc='Validation'):
        x = x.to(device)
        f0 = f0.to(device)
        loud = loud.to(device)
        y = y.to(device)
        
        y_pred = model(x, f0, loud)
        
        loss_stft = losses['stft'](y_pred, y)
        loss_l1 = losses['l1'](y_pred, y)
        loss_transient = losses['transient'](y_pred, y)
        
        loss = (args.loss_stft_weight * loss_stft + 
                args.loss_l1_weight * loss_l1 + 
                args.loss_transient_weight * loss_transient)
        
        val_losses['total'] += loss.item()
        val_losses['stft'] += loss_stft.item()
        val_losses['l1'] += loss_l1.item()
        val_losses['transient'] += loss_transient.item()
    
    n_batches = len(dataloader)
    for key in val_losses:
        val_losses[key] /= n_batches
    
    return val_losses


def main():
    args = parse_args()
    
    # シード設定
    set_seed(args.seed)
    
    # デバイス設定
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDAが利用できません。CPUを使用します。")
        args.device = 'cpu'
    device = torch.device(args.device)
    print(f"使用デバイス: {device}")
    
    # ディレクトリ作成
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = Path(args.log_dir) / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 設定を保存
    config_path = checkpoint_dir / 'config.json'
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)
    print(f"設定を保存: {config_path}")
    
    # データセット
    print("\nデータセットを読み込み中...")
    train_dataset = GuitarDIDataset(
        wav_dir=args.train_dir,
        sample_rate=args.sample_rate,
        segment_seconds=args.segment_seconds
    )
    print(f"トレーニングデータ: {len(train_dataset)} サンプル")
    
    # 検証データセット（オプション）
    val_dataset = None
    val_loader = None
    if Path(args.val_dir).exists() and list(Path(args.val_dir).glob('*.wav')):
        val_dataset = GuitarDIDataset(
            wav_dir=args.val_dir,
            sample_rate=args.sample_rate,
            segment_seconds=args.segment_seconds
        )
        print(f"検証データ: {len(val_dataset)} サンプル")
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == 'cuda')
        )
    else:
        print("検証データが見つかりません。スキップします。")
    
    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda')
    )
    
    # モデル
    print("\nモデルを初期化中...")
    model = GuitarDDSP(
        sample_rate=args.sample_rate,
        num_harm=args.num_harm,
        num_noise=args.num_noise,
        device=device
    )
    
    # パラメータ数を表示
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"総パラメータ数: {n_params:,}")
    print(f"学習可能パラメータ数: {n_trainable:,}")
    
    # 損失関数
    losses = {
        'stft': MultiScaleSTFTLoss(),
        'l1': nn.L1Loss(),
        'transient': TransientLoss(sample_rate=args.sample_rate)
    }
    
    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.1
    )
    
    # TensorBoard Writer
    writer = SummaryWriter(log_dir=log_dir)
    
    # チェックポイントから再開
    start_epoch = 0
    global_step = 0
    if args.resume:
        start_epoch, global_step = load_checkpoint(
            args.resume, model, optimizer, scheduler
        )
    
    # トレーニングループ
    print(f"\nトレーニング開始: {args.epochs} エポック")
    print(f"バッチサイズ: {args.batch_size}")
    print(f"学習率: {args.lr}")
    print(f"=" * 60)
    
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, args.epochs):
        # トレーニング
        train_losses, global_step = train_epoch(
            model, train_loader, optimizer, scheduler, losses,
            device, args, writer, global_step, epoch
        )
        
        print(f"\nEpoch {epoch} トレーニング結果:")
        print(f"  Total Loss: {train_losses['total']:.4f}")
        print(f"  STFT Loss: {train_losses['stft']:.4f}")
        print(f"  L1 Loss: {train_losses['l1']:.4f}")
        print(f"  Transient Loss: {train_losses['transient']:.4f}")
        
        # 検証
        if val_loader:
            val_losses = validate(model, val_loader, losses, device, args)
            print(f"\nEpoch {epoch} 検証結果:")
            print(f"  Total Loss: {val_losses['total']:.4f}")
            print(f"  STFT Loss: {val_losses['stft']:.4f}")
            print(f"  L1 Loss: {val_losses['l1']:.4f}")
            print(f"  Transient Loss: {val_losses['transient']:.4f}")
            
            # TensorBoard に記録
            writer.add_scalar('Val/Loss_Total', val_losses['total'], epoch)
            writer.add_scalar('Val/Loss_STFT', val_losses['stft'], epoch)
            writer.add_scalar('Val/Loss_L1', val_losses['l1'], epoch)
            writer.add_scalar('Val/Loss_Transient', val_losses['transient'], epoch)
            
            # ベストモデルを保存
            if val_losses['total'] < best_val_loss:
                best_val_loss = val_losses['total']
                best_path = checkpoint_dir / 'best_model.pt'
                save_checkpoint(
                    model, optimizer, scheduler, epoch, global_step,
                    val_losses['total'], best_path
                )
                print(f"  ベストモデルを更新!")
        
        # 定期的にチェックポイントを保存
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch:04d}.pt'
            save_checkpoint(
                model, optimizer, scheduler, epoch, global_step,
                train_losses['total'], checkpoint_path
            )
        
        print("=" * 60)
    
    # 最終チェックポイントを保存
    final_path = checkpoint_dir / 'final_model.pt'
    save_checkpoint(
        model, optimizer, scheduler, args.epochs - 1, global_step,
        train_losses['total'], final_path
    )
    
    writer.close()
    print(f"\nトレーニング完了!")
    print(f"チェックポイント: {checkpoint_dir}")
    print(f"TensorBoardログ: {log_dir}")


if __name__ == "__main__":
    main()
