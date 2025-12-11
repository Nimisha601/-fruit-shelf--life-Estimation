
import sys
import os

# FORCE FIX PATH
project_root = '/content/fruit_real_proj'
if project_root not in sys.path:
    sys.path.append(project_root)

import torch, argparse, pandas as pd
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import Adam
from src.dataset import FruitShelfDataset, get_transforms
from src.model import MultiTaskNet
import numpy as np

def safe_meta_tensor(meta_list, device):
    processed = []
    if isinstance(meta_list, torch.Tensor):
        meta_list = meta_list.cpu().tolist()
    if not isinstance(meta_list, (list, tuple)):
        meta_list = [meta_list]
    for m in meta_list:
        if isinstance(m, (list, tuple)):
            vec = list(m)
            if len(vec) < 2: vec = vec + [0.0] * (2 - len(vec))
            elif len(vec) > 2: vec = vec[:2]
        else:
            try:
                val = float(m)
                vec = [val, 0.0]
            except:
                vec = [0.0, 0.0]
        processed.append(vec)
    return torch.tensor(processed, dtype=torch.float32, device=device)

def expand_meta_to_batch(meta_tensor, batch_size):
    if meta_tensor.dim() == 1:
        meta_tensor = meta_tensor.unsqueeze(0).repeat(batch_size, 1)
    elif meta_tensor.size(0) == 1 and batch_size > 1:
        meta_tensor = meta_tensor.repeat(batch_size, 1)
    elif meta_tensor.size(0) != batch_size:
        if batch_size > meta_tensor.size(0):
             reps = batch_size // meta_tensor.size(0) + 1
             meta_tensor = meta_tensor.repeat(reps, 1)[:batch_size]
    return meta_tensor

def train_loop(train_meta, val_meta, img_root, out_dir, epochs=6, batch_size=16, lr=1e-4, backbone='mobilenet_v2'):
    train_t, val_t = get_transforms()
    train_ds = FruitShelfDataset(train_meta, img_root, transform=train_t)
    val_ds = FruitShelfDataset(val_meta, img_root, transform=val_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = len(train_ds.classes)
    
    print(f"Training started on: {device}")
    
    model = MultiTaskNet(n_classes=num_classes, backbone_name=backbone, pretrained=True, meta_dim=2).to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    os.makedirs(out_dir, exist_ok=True)
    best_mae = 1e9
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            imgs = batch['image'].to(device)
            labels = batch['class'].to(device)
            days = batch['days'].to(device)
            
            meta = safe_meta_tensor(batch['meta'], device)
            meta = expand_meta_to_batch(meta, imgs.size(0))
            
            cls_logits, pred_days = model(imgs, meta)
            
            cls_loss = F.cross_entropy(cls_logits, labels.long())
            reg_loss = F.mse_loss(pred_days, days.float())
            loss = cls_loss + 0.5 * reg_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / max(1, len(train_loader))
        print(f"Epoch {epoch+1}/{epochs} train_loss: {avg_loss:.4f}")
        
        # Validation
        model.eval()
        maes = []
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch['image'].to(device)
                days = batch['days'].to(device)
                meta = safe_meta_tensor(batch['meta'], device)
                meta = expand_meta_to_batch(meta, imgs.size(0))
                
                _, pred_days = model(imgs, meta)
                maes.append(torch.mean(torch.abs(pred_days - days.float())).item())
        
        val_mae = float(np.mean(maes)) if len(maes)>0 else -1.0
        print(f"Val MAE (days): {val_mae:.3f}")
        
        # Save
        if val_mae >= 0 and val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), os.path.join(out_dir, 'best_model.pth'))
            
    print("Training done. Best MAE:", best_mae)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_meta', required=True)
    parser.add_argument('--val_meta', required=True)
    parser.add_argument('--img_root', required=True)
    parser.add_argument('--out_dir', default='./models')
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--backbone', default='mobilenet_v2')
    args = parser.parse_args()
    
    train_loop(args.train_meta, args.val_meta, args.img_root, args.out_dir, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, backbone=args.backbone)
