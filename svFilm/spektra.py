# -*- coding: utf-8 -*-
r"""真胶片卷 —— 把 spektrafilm（Andrea Volpato）的物理链接进我们的 L2。

## 为什么
09-14 SV 定的方向：**丢弃作者线，全部用真卷**。
理由（他原话）：「**作者线们本来就是用真胶片拍的**」——
即：作者线是**从真胶片成片上量的二手**（还隔着小红书压缩图），
而 spektrafilm 是**数据表 + 光谱测量的一手**。**丢中介，用一手。**

## 它是什么
`spektrafilm` 不用 LUT，而是**重建整条光化学管线**：
**负片曝光 → 显影 → 印相 → 扫描**，全部从 `density_curves`(H&D) + `log_sensitivity`(光谱) + 染料密度推出来。

## 它自带什么（所以我们对应的层要关）
| 真卷自带 | 我们原来那层 |
|---|---|
| H&D 特征曲线（256 点） | L2 的**影调曲线** |
| `dir_couplers`（层内/层间抑制） | **彩度**（我们手工做的 chroma_p/s） |
| `grain`（分通道颗粒，蓝最大） | **颗粒层** |
| `halation`（R 最强 ⇒ 红橙，物理推导） | **黑柔/Halation** |

⇒ 用真卷时：**我们的影调曲线 / 颜色三块 / 空间层全部关掉**（不然是两套叠一起）。

## 保留什么
- **入口**（解码 + settle + 入口曲线 + clip_guard）—— 决定曝光的起点
- **锚点**（脸提亮）—— "人好看"
- **降噪**（传感器噪声，胶片没有）
- **L3 肤色** / **L4 护栏**

## ⚠ 许可
spektrafilm 的 profiles 与生成的 LUT = **CC BY-SA 4.0 + 自定义前言**
（署名 + 相同方式共享）。用了它，我们仓库的分发也要跟着这条。

## ⚠ 落点谁定
spektrafilm **不负责"好看"** —— 它的 `auto_exposure` 只是"让负片正确曝光"。
要让画面落在我们要的落点上，得动 **`enlarger.print_exposure`**（印相曝光倍率）。
⚠ **它越大越暗**（负片逻辑）。落点由调用方（`pipeline`）算好再传进来。
"""
from __future__ import annotations

import os
import sys
import threading

import numpy as np

from . import config as C

_LOCK = threading.Lock()
_PARAMS = {}          # (film, print) -> 已 init 的 params（**复用**，init 不便宜）
_SF = [None]          # 懒加载的 spektrafilm 模块


def _sf():
    """懒加载 spektrafilm（**注意**：必须能找到它的 src 目录）。"""
    if _SF[0] is None:
        root = os.environ.get('SPEKTRAFILM_ROOT') or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            '_tools', 'spektrafilm', 'src')
        if os.path.isdir(root) and root not in sys.path:
            sys.path.insert(0, root)
        from spektrafilm.runtime import init_params, simulate    # noqa: E402
        _SF[0] = (init_params, simulate)
    return _SF[0]


# 我们对外用的卷名 → spektrafilm 的 (负片 profile, 相纸 profile)
STOCK_MAP = {
    'portra400':     ('kodak_portra_400',     'kodak_portra_endura'),
    'fuji_c200':     ('fujifilm_c200',        'fujifilm_crystal_archive_typeii'),
    'pro400h':       ('fujifilm_pro_400h',    'fujifilm_crystal_archive_typeii'),
    'ektar100':      ('kodak_ektar_100',      'kodak_endura_premier'),
    'cinestill800t': ('kodak_vision3_500t',   'kodak_2383'),
    # 备着（不选就不会加载）：
    'portra160':     ('kodak_portra_160',     'kodak_portra_endura'),
    'portra800':     ('kodak_portra_800',     'kodak_portra_endura'),
    'gold200':       ('kodak_gold_200',       'kodak_endura_premium'),
    'ultramax400':   ('kodak_ultramax_400',   'kodak_endura_premium'),
    'xtra400':       ('fujifilm_xtra_400',    'fujifilm_crystal_archive_typeii'),
    'velvia100':     ('fujifilm_velvia_100',  'kodak_supra_endura'),
    'provia100f':    ('fujifilm_provia_100f', 'kodak_supra_endura'),
    'vision3_250d':  ('kodak_vision3_250d',   'kodak_2383'),
    'vision3_50d':   ('kodak_vision3_50d',    'kodak_2383'),
    'verita200d':    ('kodak_verita_200d',    'kodak_2383'),
}

# 出厂默认的几个（按"人手最常要的"排）
DEFAULT_STOCKS = ('portra400', 'fuji_c200', 'pro400h', 'ektar100', 'cinestill800t')


def _get_params(film, printp):
    key = (film, printp)
    with _LOCK:
        p = _PARAMS.get(key)
        if p is None:
            init_params, _ = _sf()
            p = init_params(film_profile=film, print_profile=printp)
            _PARAMS[key] = p
        return p


def has(stock_name):
    return stock_name in STOCK_MAP


def render(lin, stock_name, cfg=C, print_exposure=None):
    r"""喂**场景线性**（入口交出来的 `lin`），出**显示域**。

    `print_exposure` = 落点旋钮（印相曝光倍率）。**越大越暗**。
      不传就用配置里的 `SPEK_PRINT_EXPOSURE`（默认 1.0 = 让它自己定，通常偏暗）。
    """
    film, printp = STOCK_MAP[stock_name]
    _init_params, simulate = _sf()
    p = _get_params(film, printp)

    def _apply_kwargs():
        r"""把「出厂关着、我们打开」的那几项从 cfg 推进 params。

        ⚠ 必须在 `simulate` **之前**逐次设置（params 是复用的），跑完由 `finally` 还原。
        ⚠ `simulate()` 默认每次跑 `digest_params`；我们这些字段**不在 digest 的覆盖名单里**
          （preview_mode 清的是 lens_blur/grain/unsharp；lut_mode 清的是 spatial/boost_ev/
           white·black_correction/unsharp）⇒ 只要 `settings.preview_mode` 与 `debug.lut_mode`
          都是 False，我们设的值就能活到管线里。
        """
        # ① 印相曲线变形：改相纸曲线**形状**（保 D(0)/D_max/每层 A）
        if getattr(cfg, 'SPEK_MORPH', False):
            from spektrafilm.utils.morph_curves import PrintCurvesMorphParams
            p.print_render.density_curves_morph = PrintCurvesMorphParams(
                active=True,
                gamma_factor=float(getattr(cfg, 'SPEK_MORPH_GAMMA', 1.0)),
                gamma_factor_fast=float(getattr(cfg, 'SPEK_MORPH_FAST', 1.0)),
                gamma_factor_slow=float(getattr(cfg, 'SPEK_MORPH_SLOW', 1.0)),
                developer_exhaustion=float(getattr(cfg, 'SPEK_MORPH_EXHAUST', 0.0)),
            )
        # ② 柔光：SV 选「A」⇒ 挂**放大机**（印相 raw 域、颗粒形成之前 ⇒ 颗粒保锐）
        _fam = getattr(cfg, 'SPEK_DIFFUSION_FAMILY', 'black_pro_mist')
        _stg = float(getattr(cfg, 'SPEK_DIFFUSION_STRENGTH', 0.0))
        _scl = float(getattr(cfg, 'SPEK_DIFFUSION_SCALE', 1.0))
        p.enlarger.diffusion_filter.active = bool(getattr(cfg, 'SPEK_DIFFUSION_ENLARGER', False)) and _stg > 0
        p.enlarger.diffusion_filter.filter_family = _fam
        p.enlarger.diffusion_filter.strength = _stg
        p.enlarger.diffusion_filter.spatial_scale = _scl
        p.camera.diffusion_filter.active = bool(getattr(cfg, 'SPEK_DIFFUSION_CAMERA', False)) and _stg > 0
        p.camera.diffusion_filter.filter_family = _fam
        p.camera.diffusion_filter.strength = _stg
        p.camera.diffusion_filter.spatial_scale = _scl
        # ③ Halation 的高光增亮（**入口 RAW 域**重建过曝高光 = 晕圈的燃料）
        #    ⚠ `protect_ev` 必须一起降下来，否则门槛够不到、boost 是空操作。
        p.film_render.halation.boost_ev = float(getattr(cfg, 'SPEK_BOOST_EV', 0.0))
        p.film_render.halation.boost_range = float(getattr(cfg, 'SPEK_BOOST_RANGE', 0.3))
        p.film_render.halation.protect_ev = float(getattr(cfg, 'SPEK_BOOST_PROTECT_EV', 4.0))
        # ④ 预闪（不放底片、片基光直打相纸）= 加法偏置 ⇒ 提黑位、降对比
        p.enlarger.preflash_exposure = float(getattr(cfg, 'SPEK_PREFLASH', 0.0))
        p.enlarger.preflash_y_filter_shift = float(getattr(cfg, 'SPEK_PREFLASH_Y_SHIFT', 0.0))
        p.enlarger.preflash_m_filter_shift = float(getattr(cfg, 'SPEK_PREFLASH_M_SHIFT', 0.0))
        # ⑤ 扫描白平衡 / 黑位校正（⚠ 会连带改印相曝光以保中灰 ⇒ 可能挪落点）
        p.scanner.white_correction = bool(getattr(cfg, 'SPEK_SCAN_WHITE_CORR', False))
        p.scanner.black_correction = bool(getattr(cfg, 'SPEK_SCAN_BLACK_CORR', False))
        # ⑥ 像差模糊（⚠ 单位不同；`enlarger.lens_blur` 是死参数，设了也无效果）
        p.camera.lens_blur_um = float(getattr(cfg, 'SPEK_CAMERA_LENS_BLUR_UM', 0.0))
        p.scanner.lens_blur = float(getattr(cfg, 'SPEK_SCANNER_LENS_BLUR', 0.0))
        p.enlarger.lens_blur = float(getattr(cfg, 'SPEK_ENLARGER_LENS_BLUR', 0.0))

    # 每次 simulate 前按需改（params 对象是复用的 ⇒ 跑完要**还回去**）
    with _LOCK:
        old_pe = p.enlarger.print_exposure
        old_ae = p.camera.auto_exposure
        old_np = p.enlarger.normalize_print_exposure
        old_dc = p.film_render.dir_couplers.amount
        # 还原用的深拷贝：⑥ 那几项要恢复原对象，最省事是**先拍快照**（只拍我们会碰的）
        import copy as _copy
        _snap = {
            'morph': _copy.copy(p.print_render.density_curves_morph),
            'enl_diff': _copy.copy(p.enlarger.diffusion_filter),
            'cam_diff': _copy.copy(p.camera.diffusion_filter),
            'boost_ev': p.film_render.halation.boost_ev,
            'boost_range': p.film_render.halation.boost_range,
            'protect_ev': p.film_render.halation.protect_ev,
            'preflash': p.enlarger.preflash_exposure,
            'preflash_y': p.enlarger.preflash_y_filter_shift,
            'preflash_m': p.enlarger.preflash_m_filter_shift,
            'white_corr': p.scanner.white_correction,
            'black_corr': p.scanner.black_correction,
            'cam_blur': p.camera.lens_blur_um,
            'scan_blur': p.scanner.lens_blur,
            'enl_blur': p.enlarger.lens_blur,
        }
        try:
            # ★★ 浓淡旋钮（09-14 SV 选「A」）：层间抑制 = 彩度的物理来源。
            #   1.0 是出厂物理值；我们实测全批偏高 79%（12.13 vs 鹿井 6.79）⇒ 取 `SPEK_COUPLERS`。
            p.film_render.dir_couplers.amount = float(
                getattr(cfg, 'SPEK_COUPLERS', old_dc))
            # ★★ 出厂关着的暗房/光学效果（09-14 SV「都打开」）
            _apply_kwargs()
            if print_exposure is not None:
                p.camera.auto_exposure = False
                p.enlarger.normalize_print_exposure = False
                p.enlarger.print_exposure = float(print_exposure)
            else:
                p.camera.auto_exposure = bool(getattr(cfg, 'SPEK_AUTO_EXPOSURE', True))
                p.enlarger.normalize_print_exposure = True
            out = simulate(np.clip(lin, 0.0, None), p, print_timings=False)
        finally:
            p.enlarger.print_exposure = old_pe
            p.camera.auto_exposure = old_ae
            p.enlarger.normalize_print_exposure = old_np
            p.film_render.dir_couplers.amount = old_dc
            # 把上面打开的还回去（params 复用 ⇒ 不还原会污染下一卷/下一张）
            p.print_render.density_curves_morph = _snap['morph']
            p.enlarger.diffusion_filter = _snap['enl_diff']
            p.camera.diffusion_filter = _snap['cam_diff']
            p.film_render.halation.boost_ev = _snap['boost_ev']
            p.film_render.halation.boost_range = _snap['boost_range']
            p.film_render.halation.protect_ev = _snap['protect_ev']
            p.enlarger.preflash_exposure = _snap['preflash']
            p.enlarger.preflash_y_filter_shift = _snap['preflash_y']
            p.enlarger.preflash_m_filter_shift = _snap['preflash_m']
            p.scanner.white_correction = _snap['white_corr']
            p.scanner.black_correction = _snap['black_corr']
            p.camera.lens_blur_um = _snap['cam_blur']
            p.scanner.lens_blur = _snap['scan_blur']
            p.enlarger.lens_blur = _snap['enl_blur']
    return np.clip(np.asarray(out, np.float64), 0.0, 1.0)


def fit_print_exposure(lin, stock_name, target_L50, cfg=C, lo=0.05, hi=3.0, iters=9):
    """二分找 `print_exposure`，让出图**画面中位**落在 `target_L50`。

    ⚠ **print_exposure 越大越暗**（负片逻辑）⇒ 太亮要**增大**它（我第一版写反过，收敛到边界）。
    """
    from . import color
    def _med(pe):
        d = render(lin, stock_name, cfg, print_exposure=pe)
        lab = color.to_lab(d)
        return float(np.median(lab[..., 0]))
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        if _med(mid) > target_L50:
            lo = mid          # 太亮 ⇒ 增大 pe（变暗）
        else:
            hi = mid
    return (lo + hi) / 2.0
