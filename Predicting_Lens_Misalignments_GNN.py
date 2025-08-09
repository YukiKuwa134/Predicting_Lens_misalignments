# -*- coding: utf-8 -*-
"""

@author: Y.Kuwahara

機械学習によるレンズ組み立てズレの推定

Predicting Lens Misalignments
GNN使う

"""

# モジュールインポート
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader 
from torch.utils.data import random_split

from sklearn.metrics import mean_squared_error, r2_score
import shap

import multiprocessing
import time
import matplotlib.pyplot as plt


# ---- 定数設定 ----
zernike_kou = 120
fields = (-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0)  #像高比
wavelength = 0.55 #μm

n_surfaces   = 13       # 面数
coeff_per_sf = 120      # 1面あたりのZernike係数（120個×視野1 =120）

# エレメント 面マップ
ELEM_SURF_MAP = [
    (1, 2),       # L1 単レンズ
    (3, 4, 5),    # L2–L3 バルサム面 → 3 面まとめて平行移動
    (7, 8, 9),     # L4-L5
    (10, 11)      # L6
]

# Dataset定義
#GNN
class LensGNN(torch.nn.Module):
    def __init__(self, in_channels=120, hidden=64, n_outputs=20,dropout=0.2):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden,       hidden)
        #self.conv3 = GCNConv(hidden,       hidden)   # 好みで層追加

        self.readout = global_mean_pool 
        self.head    = nn.Linear(hidden, n_outputs)
        self.drop  = nn.Dropout(dropout)     

    def forward(self, x, edge_index, batch):
        if edge_index.device != x.device:          
            edge_index = edge_index.to(x.device)

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        #x = F.relu(self.conv3(x, edge_index))

        x = global_mean_pool(x, batch)   # (B, hidden) ← ここが重要
        x = self.drop(x)
        return self.head(x)     


class LensGraphDataset(InMemoryDataset):
    def __init__(self, X_path, y_path, transform=None):
        super().__init__('.', transform)
        X = np.load(X_path).astype(np.float32)   # (N, 1080)
        y = np.load(y_path).astype(np.float32)   # (N, 20)

        n_sf = n_surfaces                       # 13
        # データの総特徴量数をノード数で割る
        coeff_per_sf = X.shape[1] // n_sf       # 1080 // 13 = 83  (要確認: 整数で割り切れるか)

        # 正しくは「視野数 × zernike_kou」で reshape
        n_fields = len(fields)  # 9
        assert X.shape[1] == n_fields * zernike_kou,f"1080 != {n_fields}×{zernike_kou}"

        # __getitem__ を使うだけなので data_list は不要
        self.X = X
        self.y = y


    def __len__(self):
         return len(self.X)

    def __getitem__(self, idx):

        zi = self.X[idx]                  # (1080,)
        yi = self.y[idx]                  # (20,)

        # 正しくは (視野数, zernike_kou)
        node_attr = torch.from_numpy(
            zi.reshape(len(fields), zernike_kou)
        ).float()     

        # 2) エッジ情報（固定ならクラス変数などで保持
        # edge_index: 双方向線形＋自己ループ
        edge_index = build_edge_index(len(fields)).long()

        # 3) ラベルを tensor 化   ★★ この行が質問の箇所 ★★
        label = torch.tensor(yi, dtype=torch.float).unsqueeze(0)

        # 4) Data オブジェクトを作成
        data = Data(x=node_attr,edge_index=edge_index,y=label)

        return data

def build_edge_index(n=13):
    # 0-1-2-…-12 線形＋双方向＋自己ループ
    src = torch.arange(n-1)
    dst = src + 1
    edge_index = torch.cat([
        torch.stack([src, dst], dim=0),
        torch.stack([dst, src], dim=0),
        torch.stack([torch.arange(n), torch.arange(n)], dim=0)  # self loop
    ], dim=1)
    return edge_index



#=====================================================================================
# main関数
#
#=====================================================================================
def main():

    # ハイパーパラメータ・デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 128
    lr = 1e-3
    max_epochs = 100

    #debug
    X = np.load("DoubleGauss_wavefront_data_100k.npy")
    y = np.load("DoubleGauss_misalignments_100k.npy")
    print("X shape:", X.shape)   # 期待 → (50000, 1080) など
    print("y shape:", y.shape)   # 期待 → (50000,   20)

# ---------------- データ前処理 ----------------
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(X)           # X = (50000,1080)
    X_scaled = scaler.transform(X).astype('float32')
    np.save("DoubleGauss_wavefront_scaled_100k.npy", X_scaled)


    # DataLoader
    dataset = LensGraphDataset("DoubleGauss_wavefront_scaled_100k.npy",
                            "DoubleGauss_misalignments_100k.npy")


    dataset_len = len(dataset)
    print("Total :", len(dataset))

    # 例：訓練80%・検証10%・テスト10%
    test_ratio = 0.10
    val_ratio  = 0.10

    # int() すると切り捨てなので max(1, …) で最低 1 にする
    n_test = max(1, int(dataset_len * test_ratio))
    n_val  = max(1, int(dataset_len * val_ratio))

    # 残りをすべて train に
    n_train = dataset_len - n_test - n_val
    if n_train <= 0:
        # データが少なすぎる場合は val/test を再調整
        n_test = max(1, dataset_len // 5)    # 20% を目安に
        n_val  = max(1, dataset_len // 5)
        n_train = dataset_len - n_test - n_val

    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(0)
    )

    # ======================================
    # ★ DataLoader 作成
    # ======================================
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False)

    #モデル 作成
    model = LensGNN(in_channels=120, hidden=64, n_outputs=20).to(device)

    # ----------------------------------------------------------
    sample = dataset[0].to(device)

    node_count = sample.x.size(0)           # ノード数 (例: 9)
    batch_vec  = torch.zeros(
        node_count,
        dtype=torch.long,
        device=device
    )     

    out = model(sample.x.to(device), sample.edge_index.to(device),batch_vec)


    # ---------------- モデルと学習 ----------------
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-4, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    #---- スケジューラの定義 ----
    # OneCycleLR 定義
#    total_steps = max_epochs * len(train_loader)
#    scheduler = OneCycleLR(
#        optimizer,
#        max_lr=3e-3,            # サイクルのピーク学習率
#        total_steps=total_steps,
#        pct_start=0.3,          # 学習率上昇に使う割合
#        anneal_strategy='cos',
#        final_div_factor=1e4    # 最終 lr = max_lr / final_div_factor
#    )

    # ↓ ここに挿入 ↓
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',         # val_loss が小さくなる方向を良しとする
        factor=0.5,         # lr *= 0.5
        patience=5,         # 5 epoch 改善がなければ lr を下げる
#        verbose=True
    )
    best_val_r2 = -1e9
    epochs_no_imp = 0
    patience = 10

    # Training Loop with EarlyStopping
    best_val_loss = float('inf')
    epochs_no_improve = 0

    # 学習ループ直前にindex準備
    # ラベルリストと同じ順番で mm/deg のインデックスを作成
    label_list = [f"{ax}{i+1}"
                for i in range(len(ELEM_SURF_MAP))   # 4
                for ax in ("dx","dy","dz","tip","tilt")]
    mm_idx  = [i for i,l in enumerate(label_list) if l.startswith(("dx","dy","dz"))]
    deg_idx = [i for i,l in enumerate(label_list) if l.startswith(("tip","tilt"))]

    train_losses, val_losses, val_r2s = [], [], []

    train_time_start = time.time()

#---- 学習 ----
    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss = 0.0

        for data in train_loader:

            data = data.to(device, non_blocking=True)  
            optimizer.zero_grad()
            out   = model(data.x, data.edge_index, data.batch)
            loss  = criterion(out, data.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running_loss += loss.item() * data.num_graphs

        train_loss = running_loss / len(train_loader.dataset)   # ←平均化

        # -------- 検証 --------
        model.eval()

        # ← ここで初期化
        val_loss = 0.0
        val_r2   = 0.0

        val_loss, y_trues, y_preds = 0.0, [], []
        with torch.no_grad():
            for data in val_loader:
                data   = data.to(device)
                out  = model(data.x, data.edge_index, data.batch)

                # MSE の累積
                val_loss += F.mse_loss(out, data.y, reduction="sum").item()

                # R² の累積
                # ここではバッチごとに r2_score を計算して合計し、
                # 最後に平均をとる想定です
                y_true = data.y.cpu().numpy().ravel()
                y_pred = out .cpu().numpy().ravel()
                val_r2 += r2_score(y_true, y_pred)

        val_loss /= len(val_loader.dataset)
#        y_trues   = np.concatenate(y_trues, 0)
#        y_preds   = np.concatenate(y_preds, 0)
#        val_r2    = r2_score(y_trues, y_preds)
        val_r2   /= len(val_loader)

        # ログ表示など…
        print(f"Epoch {epoch:3d}  Train MSE={train_loss:.5f}  "
            f"Val MSE={val_loss:.5f}  Val R²={val_r2:.4f}")

        print(f"Validation — loss: {val_loss:.4f}, R²: {val_r2:.4f}")

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_r2s.append(val_r2)

        scheduler.step(val_loss)

        # -------- Early-Stopping --------
        best_val_r2, epochs_no_imp = -1e9, 0
        if val_r2 > best_val_r2 + 1e-4:  # 改善を閾値付きで判定
            best_val_r2 = val_r2
            torch.save(model.state_dict(), "best_gnn.pt")
            epochs_no_imp = 0
        else:
            epochs_no_imp += 1
            if epochs_no_imp >= patience:
                break

    print("学習経過時間：", time.time() - train_time_start)

    # --- 学習曲線の描画 ---
    plt.figure()
    plt.plot(train_losses, label="Train MSE")
    plt.plot(val_losses,   label="Val MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(val_r2s, label="Val R²")
    plt.xlabel("Epoch")
    plt.ylabel("R²")
    plt.legend()
    plt.show()


#---- 推論に切り替え ----
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    y_preds, y_trues = [], []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            pred = model(data.x, data.edge_index, data.batch)
            y_preds.append(pred.cpu())
            y_trues.append(data.y.cpu())

    y_pred = torch.cat(y_preds).numpy()
    y_test = torch.cat(y_trues).numpy()


    print("==== GNN推論結果 ====")

    # 全体指標
    mse_all = mean_squared_error(y_test, y_pred)
    r2_all  = r2_score(y_test, y_pred)
    rmse_all = np.sqrt(mse_all)
    print(f"Test  R²  = {r2_all:.6f}")
    print(f"Test  MSE = {mse_all:.6f}")
    print(f"      RMSE = {rmse_all:.6f}")

    # [mm]系/[deg]系 に分けた RMSE
    label_list = []
    for i in range(len(ELEM_SURF_MAP)):
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
    labels = [f"{ax}{i+1}" for i in range(len(ELEM_SURF_MAP)) for ax in axes]

    # 比較したいサンプルのインデックス
    idx = 0   # ここを他の番号に変えれば好きなサンプルを表示

    true_vals = y_test[idx]
    pred_vals = y_pred[idx]

    # 一覧出力
    for label, t, p in zip(labels, true_vals, pred_vals):
        unit = "mm" if label.startswith(("dx","dy","dz")) else "deg"
        print(f"{label}: True = {t:.3f} {unit}, Pred = {p:.3f} {unit}")

    np.save("true_vals.npy", np.stack(true_vals, axis=0))
    np.save("pred_vals.npy", np.stack(pred_vals, axis=0))



if __name__ == "__main__":
    #時間計測
    t1 = time.time()

    #multiprocessing.freeze_support()   # PyInstaller などで必要な場合のみ
    main()

    print("総経過時間：", time.time() - t1)
