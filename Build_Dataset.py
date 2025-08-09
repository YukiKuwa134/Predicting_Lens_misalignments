# -*- coding: utf-8 -*-
"""

@author: Y.Kuwahara

教師データ生成



"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from optiland.optic import Optic
from optiland import wavefront as wf
import multiprocessing
from joblib import Parallel, delayed , parallel_backend
from file_io import Lens

zernike_kou = 120
fields = (-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0) 
wl = 0.55 #[μm]


def build_dataset(json_path: str,n_samples:int,delta_pos,delta_deg):

    lens = Lens(json_path)

    #---- 摂動設定 ----
    np.random.seed(42)

    # ランダム生成数
    num_samples = n_samples


    cols       = [f'{ax}{i+1}'           # 列名 dx1..dz4
                for i in range(len(lens.ELEM_SURF_MAP)) # ←ここのELEM_SURF_MAP汎用化したい
                for ax in ('dx','dy','dz','tip','tilt')]
    Z_stack    = []               # 波面係数 (N,108) を溜めるリスト
    rows       = []               # 偏芯量 (N,12) を溜めるリスト
    rng        = np.random.default_rng(42)


    shifts_all = np.empty((num_samples, 4, 5), dtype=np.float32)
    for i in range(num_samples):
        # --- 1) 並進偏芯 (dx,dy,dz) の分布 ---
        if i < int(num_samples * 0.8):
            shifts_all[i,:,:3] = rng.uniform(-delta_pos, delta_pos, (4, 3))
        else:
            low, high = delta_pos * 0.8, delta_pos * 1.2
            shifts_all[i,:,:3] = rng.uniform(low, high, (4, 3))
        # --- 2) Tip/Tilt (tx,ty) の分布 ---
        shifts_all[i,:,3:] = rng.uniform(-delta_deg, delta_deg, (4, 2))

    cpu_core = multiprocessing.cpu_count()

    # --- 並列実行 ---
    results = Parallel(n_jobs=cpu_core, verbose=10, batch_size=64)(
        delayed(make_sample)(i,shifts_all[i],json_path)
        for i in range(num_samples)
    )

    # 落ちるならバックエンド処理に切り替える
    #with parallel_backend('threading', n_jobs=cpu_core):
    #    results = Parallel(verbose=10)(
    #        delayed(make_sample)(i, shifts_all[i])
    #        for i in range(num_samples)
    #    )

    # --- 結果のアンパック ---
    # results は [(Z0, s0), (Z1, s1), …] のリスト
    Z_stack, rows = zip(*results)

    #---- データ出力 ----
    np.save("wavefront_data.npy", np.stack(Z_stack, axis=0))

    np.save("misalignments.npy", np.stack(rows, axis=0))
    pd.DataFrame(np.stack(rows, axis=0), columns=cols).to_csv("misalignments.csv", index=False)



def make_sample(i_loop,shifts,json_path: str):
    lens = Lens(json_path)  # 毎サンプル新規インスタンス

    for i, (dx,dy,dz,tx,ty) in enumerate(shifts):
        lens.decenter(i, dx,dy,dz)
        lens.tiptilt(i,tx,ty)

    # CPU用
    Z = np.hstack([
        wf.ZernikeOPD(lens.optic,
                      field=(0,f), wavelength=wl,
                      zernike_type='standard', num_terms=zernike_kou
        ).coeffs
        for f in fields        #  像高比
    ])

    return Z, shifts.ravel().astype(np.float32)

def main():

    # 例: JSON ファイル設計データを渡して 20000 サンプル作成 
    build_dataset("example_design_iotest.json", n_samples=50000,delta_pos = 0.35,delta_deg = 0.5)
    print("---- proc_end ----")

    return

if __name__ == "__main__":

    main()
