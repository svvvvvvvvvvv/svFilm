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
    # 每次 simulate 前按需改（params 对象是复用的 ⇒ 要**还回去**，见下）
    with _LOCK:
        old_pe = p.enlarger.print_exposure
        old_ae = p.camera.auto_exposure
        old_np = p.enlarger.normalize_print_exposure
        try:
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
