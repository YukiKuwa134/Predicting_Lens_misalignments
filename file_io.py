# -*- coding: utf-8 -*-
"""

@author: Y.Kuwahara

機械学習によるレンズ組み立てズレの推定
file_io
設計データの読み込み
        ↓
レンズクラスの作成

"""

import numpy as np
from optiland.optic import Optic
from optiland.fileio import load_optiland_file
from optiland.materials.material import Material
from optiland.materials.ideal    import IdealMaterial

class Lens:
    """
    JSON を読み込んで Optic インスタンスを保持し、
    add_field / add_wavelength / set_aperture などの
    共通設定をセットするラッパー。
    """


    def __init__(self, json_path: str):
        # .JSONから読み込んだOpticを保持
        self.optic: Optic = load_optiland_file(json_path)

        # optic: Optic インスタンス
        total_surf_n = len(self.optic.surface_group.surfaces)
        surfaces = self.optic.surface_group.surfaces
        stop_idx = self.optic.surface_group.stop_index
        last_idx = len(surfaces) - 1

        element_map = []
        current = []

        # エレメント 面マップを作る
        total_surf_n = len(self.optic.surface_group.surfaces)
        for i in range(total_surf_n):
            if i == 0:
                # 物体面
                pass
            elif i == total_surf_n-1:
                # 像面
                pass
            else:
                # 
                if i == self.optic.surface_group.stop_index:
                    # 絞り面
                    pass
                else:
                    # ガラス面か空気面か
                    material = self.optic.surface_group.surfaces[i]
                    try:
                        mat_post = material.material_post.name
                    except:
                        mat_post = None

                    try:
                        mat_pre  = material.material_pre.name
                    except:
                        mat_pre = None
                   

                    if mat_post == None :
                        # Air
                        if current:
                            current.append(i)
                            element_map.append(tuple(current))
                            
                            current = []
                    else:
                        # glass
                        current.append(i)

        # 例: [(1,2), (3,4,5), (7,8,9), (10,11)]
        self.ELEM_SURF_MAP = element_map      

        # 共通のデフォルト設定をかける
        #self._setup_defaults()

    def _setup_defaults(self):
        # 例: F-NO, 視野, 波長をデフォルトで追加
        self.optic.set_aperture(aperture_type="imageFNO", value=5)
        self.optic.set_field_type(field_type="angle")
        for y in (0, 10, 14):
            self.optic.add_field(y=y)
        for wl, primary in ((0.4861, False), (0.5876, True), (0.6563, False)):
            self.optic.add_wavelength(value=wl, is_primary=primary)

    # もし追加で便利なデバッグ用メソッドが要れば書いておく
    def list_surfaces(self):
        return [
            (s.geometry.radius, s.thickness, s.material)
            for s in self.optic.surface_group.surfaces
        ]

    def decenter(self, elem_idx: int, dx=0, dy=0, dz=0):
        """elem_idx 番目のレンズをデセンタリング"""
        for s in self.ELEM_SURF_MAP[elem_idx]:
            cs = self.optic.surface_group.surfaces[s].geometry.cs
            cs.x += dx; cs.y += dy; cs.z += dz

    def tiptilt(self, elem_idx: int, tip_deg: float, tilt_deg: float):
        """elem_idx 番目のレンズをチルト・チップ"""
        tip  = np.deg2rad(tip_deg)
        tilt = np.deg2rad(tilt_deg)
        for s in self.ELEM_SURF_MAP[elem_idx]:
            cs = self.optic.surface_group.surfaces[s].geometry.cs
            cs.rx += tip; cs.ry += tilt
    
    # エレメント 面マップ
    ELEM_SURF_MAP = []
