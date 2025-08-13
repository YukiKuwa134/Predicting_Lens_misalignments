# -*- coding: utf-8 -*-
"""

@author: Y.Kuwahara

機械学習によるレンズ組み立てズレの推定

Predicting Lens Misalignments
MLP

"""

# モジュールインポート
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, r2_score

import multiprocessing
import time

# ---- 定数設定 ----
zernike_kou = 120
fields = (-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0)  #像高比
wavelength = 0.55 #μm


# Dataset定義
class ZernikeMisalignmentDataset(Dataset):
    def __init__(self, X_path, y_path, transform=None):
        self.X = np.load(X_path).astype(np.float32)
        self.y = np.load(y_path).astype(np.float32)
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

# MLPモデル定義 Residual接続
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return x + self.net(x)

class MLPResNet(nn.Module):
    def __init__(self, in_features, out_features=20, hidden=256, dropout=0.1):
        super().__init__()
        self.fc_in  = nn.Linear(in_features, hidden)

        self.block1 = ResidualBlock(hidden, dropout)
        self.block2 = ResidualBlock(hidden, dropout)

        self.fc_out = nn.Linear(hidden, out_features)

    def forward(self, x):
        h = F.relu(self.fc_in(x))
        h = self.block1(h)
        h = self.block2(h)
        return self.fc_out(h)

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 512
    lr = 1e-3
    max_epochs = 100
    patience = 40  # EarlyStopping

    # DataLoader
    ds = ZernikeMisalignmentDataset("DoubleGauss_wavefront_data_multi_t.npy",
                                    "DoubleGauss_misalignments.npy")

    X = np.load("DoubleGauss_wavefront_data_multi_t.npy").reshape(-1, zernike_kou * len(fields))
    y = np.load("DoubleGauss_misalignments.npy")

    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        random_state=0
    )

    # train/val split
    n = len(ds)
    n_train = int(n*0.8)
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n-n_train], generator=torch.Generator().manual_seed(0))

    train_loader = DataLoader(train_ds,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True,        # GPU転送を高速化
                              prefetch_factor=2,      # バッチ先読み数
                              persistent_workers=True # ワーカーを使い回す
                              )

    val_loader   = DataLoader(val_ds,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=4,
                              pin_memory=True,
                              persistent_workers=True
                              )

    n_surfaces = 4


    # モデル・Optimizer・Loss 設定
    model = MLPResNet(in_features=ds.X.shape[1],
                      out_features=y_train.shape[1],
                      hidden=256,
                      dropout=0.1
                    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr , weight_decay=1e-4)
    criterion = nn.MSELoss()

    # OneCycleLR 定義
    total_steps = max_epochs * len(train_loader)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=3e-3,         # サイクルのピーク学習率
        total_steps=total_steps,
        pct_start=0.3,       # 学習率上昇に使う割合
        anneal_strategy='cos',
        final_div_factor=1e4 # 最終lr
    )

    # Training Loop with EarlyStopping
    best_val_loss = float('inf')
    epochs_no_improve = 0

    # ラベルリストと同じ順番で [mm]/[deg]のインデックスを作成
    label_list = [f"{ax}{i+1}"
                for i in range(n_surfaces)   # 4
                for ax in ("dx","dy","dz","tip","tilt")]
    mm_idx  = [i for i,l in enumerate(label_list) if l.startswith(("dx","dy","dz"))]
    deg_idx = [i for i,l in enumerate(label_list) if l.startswith(("tip","tilt"))]



    for epoch in range(1, max_epochs+1):
        # --- train ---
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(x_batch)

            criterion = nn.SmoothL1Loss()

            loss_mm  = F.smooth_l1_loss(y_pred[:, mm_idx],  y_batch[:, mm_idx])
            loss_deg = F.smooth_l1_loss(y_pred[:, deg_idx], y_batch[:, deg_idx])
            loss     = loss_mm + 3.0 * loss_deg

            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item() * x_batch.size(0)
        train_loss /= n_train

        # --- validate ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                y_pred = model(x_batch)
                val_loss += criterion(y_pred, y_batch).item() * x_batch.size(0)
        val_loss /= (n - n_train)

        print(f"Epoch {epoch:3d}  Train MSE={train_loss:.5f}  Val MSE={val_loss:.5f}")

        # EarlyStopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # モデルのロード&テスト
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()

#---- 推論に切り替え ----

    # テストデータをTensor/NumPyに準備
    X_test_np = ds.X
    y_test_np = np.load("DoubleGauss_misalignments.npy")

    X_test = torch.from_numpy(X_test_np).float().to(device)
    y_test = y_test_np

    # 推論
    with torch.no_grad():
        y_pred = model(X_test).cpu().numpy()

    print("==== NN推論結果 ====")

    # 全体指標
    mse_all = mean_squared_error(y_test, y_pred)
    r2_all  = r2_score(y_test, y_pred)
    rmse_all = np.sqrt(mse_all)
    print(f"Test  R²  = {r2_all:.6f}")
    print(f"Test  MSE = {mse_all:.6f}")
    print(f"      RMSE = {rmse_all:.6f}")

    # mm系/deg系 に分けた RMSE
    label_list = []
    for i in range(4):
        num = i+1
        for ax in ("dx","dy","dz","tip","tilt"):
            label_list.append(f"{ax}{num}")


    mm_idx  = [i for i,l in enumerate(label_list) if l.startswith(("dx","dy","dz"))]
    deg_idx = [i for i,l in enumerate(label_list) if l.startswith(("tip","tilt"))]

    rmse_mm  = np.sqrt(mean_squared_error(y_test[:, mm_idx],  y_pred[:, mm_idx]))
    rmse_deg = np.sqrt(mean_squared_error(y_test[:, deg_idx], y_pred[:, deg_idx]))
    print(f"→ mm系  RMSE:  {rmse_mm:.6f} mm")
    print(f"→ deg系 RMSE: {rmse_deg:.6f} deg")

    # 表示用ラベルを作成
    axes      = ["dx","dy","dz","tip","tilt"]
    labels = [f"{ax}{i+1}" for i in range(n_surfaces) for ax in axes]

    idx = 0   # ここを他の番号に変えれば好きなサンプルを表示できる

    true_vals = y_test_np[idx]
    pred_vals = y_pred[idx]

    # 一覧出力
    for label, t, p in zip(labels, true_vals, pred_vals):
        unit = "mm" if label.startswith(("dx","dy","dz")) else "deg"
        print(f"L1 {label}: True = {t:.3f} {unit}, Pred = {p:.3f} {unit}")



if __name__ == "__main__":
    #時間計測
    t1 = time.time()

    multiprocessing.freeze_support()   # PyInstallerなどで必要な場合のみ
    main()

    print("経過時間：", time.time() - t1)
