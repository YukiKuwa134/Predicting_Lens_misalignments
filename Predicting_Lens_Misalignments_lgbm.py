# -*- coding: utf-8 -*-
"""

@author: Y.Kuwahara

機械学習によるレンズ組み立てズレの推定

Predicting Lens Misalignments
LightGBM

"""

# モジュールインポート
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from optiland import wavefront, zernike
from optiland.optic import Optic

import cupy as cp
import time
from tqdm import tqdm
import joblib

# ---- 定数設定 ----
zernike_kou = 120
field = (-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0)  #像高比
wavelength = 0.55 #μm

#時間計測
t1 = time.time()

# ---- Data loading ----------
X = np.load("DoubleGauss_wavefront_data.npy").reshape(-1, len(field)*zernike_kou)
y = pd.read_csv("DoubleGauss_misalignments.csv").to_numpy()
best_lgbm = joblib.load("best_lgbm_params.pkl")

n_elems = y.shape[1] // 5
mm_idx  = [i*5 + j for i in range(n_elems) for j in (0,1,2)]
deg_idx = [i*5 + j for i in range(n_elems) for j in (3,4)]

labels = [
    f"L{i+1} {'d'+ax if ax in ('x','y','z') else ax}{i+1}"
    for i in range(n_elems)
    for ax in ("x","y","z","tip","tilt")
]

X_tr, X_te, y_tr, y_te = train_test_split(
     X, y, test_size=0.2, random_state=0, shuffle=True
 )

y_tr_orig = y_tr.copy()
y_te_orig = y_te.copy()

# ---- 出力yを一度だけスケーリング ---
scaler = StandardScaler()
y_tr_scaled = scaler.fit_transform(y_tr)
y_te_scaled = scaler.transform(y_te)

# 特徴名リスト
feature_names = [f"Z{i}" for i in range(X_tr.shape[1])]

# NumPy → DataFrame に変換
X_tr = pd.DataFrame(X_tr, columns=feature_names)
X_te = pd.DataFrame(X_te, columns=feature_names)


base_xgb = XGBRegressor(
    tree_method='gpu_hist',
    predictor='gpu_predictor',
    n_estimators=500,
    max_depth=6,
    learning_rate=0.1,
    verbosity=0,
    eval_metric='rmse'
)

estimators = []
for idx in tqdm(range(y_tr.shape[1]), desc="Training XGB outputs"):
    m = XGBRegressor(
        tree_method='gpu_hist',
        predictor='gpu_predictor',
        n_estimators=500,
        max_depth=6,
        learning_rate=0.1,
        verbosity=0,
        eval_metric='rmse'
    )
    m.fit(
        X_tr,
        y_tr_scaled[:, idx],
        eval_set=[(X_te, y_te_scaled[:, idx])],
        verbose=False
    )
    estimators.append(m)

# 学習済みモデルを MultiOutputRegressor に詰め替え
multi_xgb = MultiOutputRegressor(base_xgb, n_jobs=1)
multi_xgb.estimators_ = estimators

# 予測→逆スケール
y_pred_scaled = multi_xgb.predict(X_te)           # スケール済み予測
y_pred = scaler.inverse_transform(y_pred_scaled)  # 元の単位に戻す

# テストセット評価
mse_test = mean_squared_error(
    y_te_orig,      # スケーリング前の正解 y
    y_pred,         # 逆スケールにしたテスト予測 y
    multioutput='uniform_average'
)
r2_test = r2_score(
    y_te_orig,
    y_pred,
    multioutput='uniform_average'
)
print(f"Test  R²  = {r2_test:.6f}")
print(f"Test  MSE = {mse_test:.6f}")

# mm/deg 別 データセット
rmse_mm_test = np.sqrt(mean_squared_error(
    y_te_orig[:, mm_idx],
    y_pred[:,   mm_idx],
    multioutput='uniform_average'
))
rmse_deg_test = np.sqrt(mean_squared_error(
    y_te_orig[:, deg_idx],
    y_pred[:,   deg_idx],
    multioutput='uniform_average'
))
print(f"→ Test mm系 RMSE: {rmse_mm_test:.4f} mm")
print(f"→ Test deg系 RMSE: {rmse_deg_test:.4f} deg")

#---- LightGBM ----
base_lgbm = LGBMRegressor(
    **best_lgbm,
    device='gpu',
    force_col_wise=True,
    verbosity=-1
)


estimators = []
for idx in tqdm(range(y_tr_scaled.shape[1]), desc="Training LGB output"):
    m = LGBMRegressor(
        **best_lgbm,
        device='gpu',
        force_col_wise=True,
        verbosity=-1
    )
    m.fit(X_tr, y_tr_scaled[:, idx])
    estimators.append(m)

# 学習済みモデルを入れる
multi_lgbm = MultiOutputRegressor(base_lgbm, n_jobs=-1)
multi_lgbm.estimators_ = estimators

#予測
y_pred_lgbm_scaled = multi_lgbm.predict(X_te)
y_pred_lgbm        = scaler.inverse_transform(y_pred_lgbm_scaled)

mse = mean_squared_error(
    y_te_orig,
    y_pred,
    multioutput='uniform_average'
)
r2 = r2_score(
    y_te_orig,
    y_pred,
    multioutput='uniform_average'
)
print(f"XGB  R²  = {r2:.6f}")
print(f"XGB  MSE = {mse:.6f}")


# ── LightGBM の予測（スケール→逆スケール） ──
y_pred_lgbm_scaled = multi_lgbm.predict(X_te)
y_pred_lgbm        = scaler.inverse_transform(y_pred_lgbm_scaled)


# ── テストセット評価 ──
mse_lgbm = mean_squared_error(
     y_te_orig, y_pred_lgbm, multioutput='uniform_average'
)
r2_lgbm = r2_score(
    y_te_orig, y_pred_lgbm, multioutput='uniform_average'
)
print(f"LGBM  R²  = {r2_lgbm:.6f}")
print(f"LGBM  MSE = {mse_lgbm:.6f}")


#---- エレメント 面マップ ----
ELEM_SURF_MAP = [
    (1, 2),       # L1
    (3, 4, 5),    # L2–L3
    (7, 8, 9),    # L4-L5
    (10, 11)      # L6
]

class DoubleGauss(Optic):

    def __init__(self):
        super().__init__()

    def reset(self):
        super()._initialize_attributes()
        self._build_surfaces()
        self._setup_defaults()

    def _build_surfaces(self):
        # レンズデータ
        self.add_surface(index=0, radius=np.inf, thickness=np.inf)
        self.add_surface(index=1, radius=56.20238, thickness=8.75, material="N-SSK2")
        self.add_surface(index=2, radius=152.28580, thickness=0.5)
        self.add_surface(index=3, radius=37.68262, thickness=12.5, material="N-SK2")
        self.add_surface(
            index=4,
            radius=np.inf,
            thickness=3.8,
            material=("F5", "schott"),
        )
        self.add_surface(index=5, radius=24.23130, thickness=16.369445)
        self.add_surface(index=6, radius=np.inf, thickness=13.747957, is_stop=True)
        self.add_surface(
            index=7,
            radius=-28.37731,
            thickness=3.8,
            material=("F5", "schott"),
        )
        self.add_surface(index=8, radius=np.inf, thickness=11, material="N-SK16")
        self.add_surface(index=9, radius=-37.92546, thickness=0.5)
        self.add_surface(index=10, radius=177.41176, thickness=7, material="N-SK16")
        self.add_surface(index=11, radius=-79.41143, thickness=61.487536)
        self.add_surface(index=12) #像面

    def _setup_defaults(self):
        self.set_aperture(aperture_type="imageFNO", value=5)
        self.set_field_type(field_type="angle")

        self.add_field(y=0)
        self.add_field(y=10)
        self.add_field(y=14)

        self.add_wavelength(value=0.4861)
        self.add_wavelength(value=0.5876, is_primary=True)
        self.add_wavelength(value=0.6563)

    # 軸ずれ
    def decenter(self, elem_idx, dx=0, dy=0, dz=0):
        for s in ELEM_SURF_MAP[elem_idx]:
            cs = self.surface_group.surfaces[s].geometry.cs
            cs.x += dx;  cs.y += dy;  cs.z += dz

    # tiptilt
    def tiptilt(self, elem_index: int, tip_deg: float, tilt_deg: float):
        #tip_deg, tilt_deg は [deg]。
        # ラジアンに変換
        tip  = np.deg2rad(tip_deg)
        tilt = np.deg2rad(tilt_deg)

        for s in ELEM_SURF_MAP[elem_index]:
            cs = self.surface_group.surfaces[s].geometry.cs
            cs.rx += tip; cs.ry += tilt


    def randomly_misalign(self,delta_pos_max: float = 0.3,delta_tilt_max: float = 0.5):
        """
        4エレメント × (dx,dy,dz, tip, tilt) = 20 要素を返す。
        tip, tilt は [deg]。
        """
        # 並進偏芯 分布 (4×3)
        delta_pos  = np.random.uniform(-delta_pos_max,delta_pos_max,size=(4, 3))
        # Tip/Tilt 分布 (4×2) ← deg 単位の乱数
        delta_tilt = np.random.uniform(-delta_tilt_max,delta_tilt_max,size=(4, 2))

        # 4×5 の行列にまとめ
        shifts = np.hstack((delta_pos, delta_tilt))  # shape = (4,5)

        # 各エレメントに注入
        for i, (dx, dy, dz, tip_deg, tilt_deg) in enumerate(shifts):
            self.decenter(i,dx,dy,dz)
            self.tiptilt(i,tip_deg,tilt_deg)

        # 一次元化して返す (1×20)
        return shifts.ravel().reshape(1, -1)

    def fit_zernike(self, fields=field, wl=wavelength, n_terms=zernike_kou):
        coeffs = []
        for fy in fields:
            z = wavefront.ZernikeOPD(
                    self,
                    field=(0, fy),
                    wavelength=wl,
                    zernike_type="standard",
                    num_terms=n_terms,
            )
            coeffs.append(z.coeffs)
        return np.hstack(coeffs).reshape(1, -1)    # shape (1, 108)


lens = DoubleGauss()

# ── 新規サンプル（単一データ）での予測・表示 ──
X_new_df = lens.fit_zernike(fields=field, n_terms=zernike_kou)
X_new_df = pd.DataFrame(X_new_df, columns=feature_names)

# 単一サンプル予測
y_pred_new_scaled = multi_xgb.predict(X_new_df)
y_pred_new        = scaler.inverse_transform(y_pred_new_scaled)

# ── New Data の作成 ──
lens.reset()
y_new = lens.randomly_misalign()             # shape=(1,20) 真値取得
X_new = lens.fit_zernike(fields=field, n_terms=zernike_kou)
X_new = X_new.reshape(1, -1)

# 予測
y_pred_new_scaled = multi_xgb.predict(X_new)
y_pred_new        = scaler.inverse_transform(y_pred_new_scaled)

# 単一サンプル評価
from sklearn.metrics import mean_squared_error
mse_single = mean_squared_error(y_new, y_pred_new)
print(f"\nPerformance on New Data:")
print(f"MSE(single) = {mse_single:.6f}")

# true_vs_pred 部分では y_pred_new を使う
for i, label in enumerate(labels):
    true_value = y_new[0, i]
    pred_value = y_pred_new[0, i]
    measure    = label.split()[1]
    if measure.startswith(('dx','dy','dz')):
        unit, tv, pv = 'mm', true_value, pred_value
    else:
        unit, tv, pv = 'deg',true_value, pred_value
    print(f"{label}: True = {tv:.3f} {unit}, Pred = {pv:.3f} {unit}")



# レンズ構成見たいときは
#lens.draw()


#時間測定
t2 = time.time()
# 経過時間を表示
elapsed_time = t2-t1
print(f"経過時間：{elapsed_time}")

# 散布図
plt.figure(figsize=(6,6))

plt.scatter(y_new.flatten(), y_pred_new.flatten(), alpha=0.7, edgecolors="k")
plt.xlabel("True value")
plt.ylabel("Predicted value")
plt.title("New-data True vs Pred (all 20 outputs)")
plt.plot([-0.5,0.5],[-0.5,0.5],"r--")
plt.grid()
plt.show()
