import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset , DataLoader
import joblib
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR 
import json


Device = "cuda" if torch.cuda.is_available() else "cpu"

cfd_ds = 'converted_dataset.npz' #interpolated + Re + Cp + Ncrit , alpha 
xfoil_ckpt = 'best_ckpt.pt'     #ckpt from 2Head Cp only model 
scaler_path = 'xf/scalers.pkl'       #scalers for trainig airfoil-ml
ft_model = 'xfoilml_finetuned.pt' #new model name

############## Define MLP #################################################
class MLP(nn.Module):
    def __init__(self, d_in: int, d_out: int, hidden: list[int], 
                 act=nn.GELU, dropout: float = 0.0):
        super().__init__()
        self.act = act  
        
        layers = []
        prev = d_in
        for h in hidden:
            layers += [nn.Linear(prev, h), act()] 
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h

        self.encoder = nn.Sequential(*layers)
        d_features = hidden[-1]
        d_out_head = d_out // 2
        
        self.head_ps = nn.Sequential(
            nn.Linear(d_features, d_features),
            act(),
            nn.Linear(d_features, d_out_head)
        )
        self.head_ss = nn.Sequential(
            nn.Linear(d_features, d_features),
            act(),
            nn.Linear(d_features, d_out_head)
        )
        
        self._init_weights()

    def _init_weights(self):
        nonlinearity = 'relu' if self.act == nn.ReLU else 'linear'
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if nonlinearity == 'relu':
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                else:
                    nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
    def forward(self, x):
        features = self.encoder(x)
        out_ps = self.head_ps(features)
        out_ss = self.head_ss(features)
        return torch.cat([out_ps, out_ss], dim=1)
    
############################################################################
# X_YPS   = slice(0, 100)
# X_YSS   = slice(100, 200)
# X_RE = 200
# X_ALPHA = 201
# X_NCRIT = 202

class CFD_Dataset(Dataset):
    def __init__(self, data_path, scalers_path):
        data = np.load(data_path)
        self.scalers = joblib.load(scalers_path)["scalers"]

        self.y_ps = data['y_ps'].astype(np.float32)
        self.y_ss = data['y_ss'].astype(np.float32)
        self.Re = data['Re'].astype(np.float32)
        self.alpha = data['alpha'].astype(np.float32)
        self.Ncrit = data['Ncrit'].astype(np.float32)
        self.Cp_ps = data['Cp_ps'].astype(np.float32)
        self.Cp_ss = data['Cp_ss'].astype(np.float32)

        self._preprocess()

    def _preprocess(self):
        self.X = np.hstack([
                self.y_ps , self.y_ss,
                self.Re[:,None],
                self.alpha[:,None],
                self.Ncrit[:,None]
                ]).astype(np.float32)
        
        self.Y = np.hstack([self.Cp_ps, self.Cp_ss]).astype(np.float32)

        re_log = np.log10(self.X[:, 200:201])
        self.X[:, 200] = self.scalers["sc_logRe"].transform(re_log).ravel()
        self.X[:, 201] = self.scalers["sc_alpha"].transform(self.X[:, 201:202]).ravel()
        self.X[:, 202] = self.X[:, 202] / np.float32(self.scalers["ncrit_div"])
        self.Y[:, :100] = self.scalers["sc_Cp_ps"].transform(self.Y[:, :100])
        self.Y[:, 100:] = self.scalers["sc_Cp_ss"].transform(self.Y[:, 100:])

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return {
            'input': torch.FloatTensor(self.X[idx]),
            'target': torch.FloatTensor(self.Y[idx])
        }


# def train
def train_epoch(dl, optimizer, model, criterion, scaler, device):
    model.train()
    losses = []
    for batch in dl:
        Xb = batch['input'].to(device, non_blocking=True)
        Yb = batch['target'].to(device, non_blocking=True)
        
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device=="cuda")):
            Yhat = model(Xb)
            loss = criterion(Yhat, Yb)
        
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(loss.item())
    
    return np.mean(losses)

# def val
def eval_epoch(dl, model, criterion, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in dl:
            Xb = batch['input'].to(device, non_blocking=True)
            Yb = batch['target'].to(device, non_blocking=True)
            
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device=="cuda")):
                Yhat = model(Xb)
                loss = criterion(Yhat, Yb)
            
            losses.append(loss.item())
    
    return np.mean(losses)


def main():
    d_in  = 203
    d_out = 200
    ft_config = {
    "lr": 1e-5,  
    "batch": 256, 
    "epochs": 30,
    "patience": 7,
    'train_split': 0.85,
    "weight_decay": 1e-4}

    print(f"Device: {Device}")

    full_dataset = CFD_Dataset(cfd_ds, scaler_path)
    print(f"Total samples: {len(full_dataset)}")
    n_train = int(len(full_dataset) * ft_config['train_split'])
    n_val = len(full_dataset) - n_train
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [n_train, n_val])
    print(f"Train: {n_train}, Val: {n_val}")

    train_loader = DataLoader(train_dataset, batch_size=ft_config['batch'],
        shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=512,
        shuffle=False, num_workers=4, pin_memory=True)
    
    # Load checkpoint
    checkpoint = torch.load(xfoil_ckpt, map_location=Device)
    model_config = checkpoint['model_config']
    model = MLP(d_in=203, d_out=200,hidden=model_config["hidden"], act=nn.ReLU, dropout=0.1).to(Device)
    # Load pretrained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    print("✓ Loaded pretrained weights from step1")
    optimizer = torch.optim.AdamW(model.parameters(), lr=ft_config['lr'],
                                   weight_decay=ft_config['weight_decay'])
    
    scheduler = CosineAnnealingLR(optimizer, T_max=ft_config['epochs'])
    mse = nn.MSELoss()
    best_val = float("inf")
    scaler = torch.amp.GradScaler(enabled=(Device == "cuda"))
    # train_hist, val_hist = [], []
    history = {
        'train_loss': [],
        'val_loss': []}

    for epoch in range(ft_config['epochs']):
        print(f"\nEpoch {epoch+1}/{ft_config['epochs']}")

        train_loss = train_epoch( train_loader, optimizer, model, mse, scaler , Device)
        val_loss = eval_epoch( val_loader,model, mse, Device)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"epoch {epoch:03d} | train {train_loss:.5f} | val {val_loss:.5f} | LR {current_lr:.2e}")

        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        if val_loss + 1e-6 < best_val:
            best_val = val_loss
            bad = 0
            # Save complete checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': ft_config,
                'architecture': {
                    'd_in': d_in,
                    'd_out': d_out,
                    'hidden': model_config["hidden"],
                },
                'original_checkpoint': xfoil_ckpt,
            }, ft_model)
            
            # Also save just the state dict (for inference)
            torch.save(model.state_dict(), ft_model.replace('.pt', '_state.pt'))
            
            print(f"  ✓ Best model saved! val: {val_loss:.6f}")
        else:
            bad += 1
            if bad >= ft_config['patience']:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    # Save history
        serializable_history = {
        key: [float(x) if isinstance(x, np.floating) else x for x in value]
        for key, value in history.items()}

        with open('finetuning_history.json', 'w') as f:
            json.dump(serializable_history, f, indent=2)
        
    # Plot training curves
    fig, ax = plt.subplots( figsize=(14, 5))
    ax.plot(history['train_loss'], label='Train', linewidth=2)
    ax.plot(history['val_loss'], label='Val', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (Scaled)')
    ax.set_title('Fine-tuning Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('finetuning_curves.png', dpi=150)
    plt.show()
    print("\n" + "="*80)
    print("FINE-TUNING COMPLETE")


if __name__ == "__main__":
    main()