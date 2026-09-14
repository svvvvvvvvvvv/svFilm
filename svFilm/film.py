# -*- coding: utf-8 -*-
"""真胶片的**成色模型** —— 09-14 SV「按你的判断顺序都做了」时的三块。

三块各自独立、都可在 config 里开关：

  【丙】`crosstalk`    —— 颜色串扰。三层乳剂的染料会互相吸收 ⇒ 一个 3×3 矩阵，
                          而且**按亮度分三段用不同强度**（中间调最弱）。
                          出处 LIMO（`_debug/_rs/FilmEmulation.metal` STAGE E）。
  【甲】`layer_speeds` —— 分通道响应。三层乳剂感光度不同 ⇒ `rgb = rgb ** (1/speeds)`。
                          出处 LIMO STAGE B（`FilmPreset.layerSpeeds`）。
  【乙】`density`      —— **完整密度引擎**：log 曝光 → R/G/B **各自一条密度曲线** →
                          `透射率 = 10^(−密度)` → 用该胶片自己的分通道底色归一化 → 负片翻正。
                          出处 Emulsifier（`_debug/_rs/emul/engine.py`）；
                          曲线数据是真的 **Kodak Portra 400** 实测（`data/portra400_char.csv`，101 点）。

为什么这三块值得单独成层：我们原来的颜色全在 **Lab 感知色空间里"事后加偏移"**
（暗部偏色、亮部偏色是手工设的两个数）。真胶片的色偏是**从三条密度曲线和片基底色里长出来的**
—— 这两个不是同一件事。
"""
import os

import numpy as np

from . import color
from . import config as C

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

_cache = {}


# ============================== 曲线数据 ==============================

def load_curve(name='portra400'):
    """读一条真胶片的感光曲线。返回 dict（带缓存）。

    `logE` 是 log10 曝光量；`r/g/b` 三条是**密度**（越大越黑）；`d_min/d_max` 是该胶片自己的
    分通道底片密度（**这三条不一样 = 负片那片橙色片基**）。
    """
    if name in _cache:
        return _cache[name]
    import csv
    import json
    key = str(name).strip().lower()
    csv_path = os.path.join(_DATA, '%s_char.csv' % key)
    js_path = os.path.join(_DATA, 'stocks_measured', '%s.json' % key)
    if not os.path.exists(csv_path):
        raise FileNotFoundError('没有这条曲线：%s' % csv_path)
    le, ch = [], [[], [], []]
    with open(csv_path, newline='', encoding='utf-8') as f:
        rd = csv.reader(f)
        head = next(rd)
        idx = [head.index(k) for k in ('LogE', 'R', 'G', 'B')]
        for row in rd:
            if not row or not row[0].strip():
                continue
            le.append(float(row[idx[0]]))
            for i in range(3):
                ch[i].append(float(row[idx[1 + i]]))
    d = dict(name=key, logE=np.array(le, np.float64),
             r=np.array(ch[0]), g=np.array(ch[1]), b=np.array(ch[2]))
    dmat = np.stack([d['r'], d['g'], d['b']], -1)
    d['d_min'] = dmat.min(0)                      # 分通道底片密度（片基 + 橙罩）
    d['d_max'] = dmat.max(0)
    if os.path.exists(js_path):
        j = json.load(open(js_path, encoding='utf-8'))
        an = j.get('density_anchors') or {}
        if an.get('d_min'):
            d['d_min'] = np.array([an['d_min'][k] for k in 'rgb'], np.float64)
            d['d_max'] = np.array([an['d_max'][k] for k in 'rgb'], np.float64)
        d['film_name'] = j.get('film_name')
    _cache[name] = d
    return d


# ============================== 【丙】颜色串扰 ==============================

# 3×3 串扰核（对角线"自己减多少"、非对角"串过去多少"）。数值出自 LIMO STAGE E 的那个矩阵。
_CROSSTALK_K = np.array([
    [0.075, 0.038, 0.015],
    [0.023, 0.053, 0.030],
    [0.015, 0.045, 0.090],
], np.float64)


def crosstalk(rgb, amount=0.38, crossovers=(0.25, 0.55, 0.88), cfg=C):
    r"""【丙】颜色串扰：三层乳剂的染料互相吸收。

    `amount` 是总强度（0~1，LIMO 给 Portra 400 标的是 **0.38**）；
    **强度按亮度分三段**：暗部 ×0.6、中间调 ×(1−35%)、亮部 ×1.3
    （物理上是 T 颗粒在不同曝光区的散射不一样）。
    """
    if amount <= 1e-4:
        return rgb
    x, y, z = (float(crossovers[0]), float(crossovers[1]), float(crossovers[2]))
    lum = color.luma(rgb)
    s = lambda e0, e1, v: np.clip((v - e0) / max(e1 - e0, 1e-6), 0.0, 1.0) ** 2 * (3 - 2 * np.clip((v - e0) / max(e1 - e0, 1e-6), 0.0, 1.0))  # noqa: E731
    mid_red = s(0.20, 0.35, lum) * s(0.70, 0.55, lum)     # 中间调掩膜（第二项是反向的）
    mid_red = np.clip(mid_red, 0.0, 1.0)
    ct = (amount * s(0.0, x, lum) * 0.6
          + amount * (s(x, y, lum) - s(x, y, lum) * mid_red * 0.35)
          + amount * s(y, z, lum) * 1.3)
    ct = np.clip(ct, 0.0, 1.0)[..., None, None]              # (H, W, 1, 1)
    m = np.eye(3).reshape(1, 1, 3, 3) - ct * _CROSSTALK_K.reshape(1, 1, 3, 3)
    return np.einsum('...ij,...j->...i', m, rgb)


# ============================== 【甲】分通道响应 ==============================

def layer_speeds(rgb, speeds=(0.96, 1.0, 1.03), strength=1.0, cfg=C):
    r"""【甲】三层乳剂感光度不同 ⇒ `rgb = rgb ** (1/speeds)`。

    `speeds` 出自 LIMO `FilmPreset.layerSpeeds`（Portra 400 = **0.96 / 1.0 / 1.03**：
    红层稍慢、蓝层稍快 ⇒ 亮部偏暖、暗部偏冷，正是 Portra 的性格）。
    `strength` 把 3 个 speed 朝 1.0 插值（0 = 恒等）。
    """
    sp = 1.0 + (np.asarray(speeds, np.float64) - 1.0) * float(np.clip(strength, 0.0, 2.0))
    if np.allclose(sp, 1.0):
        return rgb
    return np.maximum(rgb, 1e-9) ** (1.0 / sp.reshape(1, 1, 3))


# ============================== 【乙】密度引擎 ==============================

def _t_at(d, sc, mapped, grid):
    """给定"映射后的 log E"，返回三通道的归一化输出（0~1，未翻转）。"""
    dens = np.array([np.interp(mapped, grid, d[k]) for k in 'rgb'])
    t = 10.0 ** (-dens * sc)
    t_max = 10.0 ** (-np.asarray(d['d_min']) * sc)
    t_min = 10.0 ** (-np.asarray(d['d_max']) * sc)
    return (t - t_min) / np.maximum(t_max - t_min, 1e-5)


def _print_offset(d, sc, sc_log=2.0, mid_disp=0.45, mid_lin=0.18):
    r"""★★ **印相曝光（分通道）** —— 真实暗房的 "printer lights"。

    每张相纸印的时候，R/G/B 三个曝光量是**分别调**的（不然负片的橙色片基会让片子一片红）。
    我们按通道各自解一个偏移，让"**中灰进 = 中灰出**"。这样：

      · **位置不动**（不依赖画面内容 ⇒ 不是"把每张都拽到同一个中灰"）
      · 三个通道**形状上的差异全部保留** —— 那才是胶片的性格
        （该亮的偏暖、该暗的偏冷，是曲线长出来的，不是事后加的偏移）

    解的是**绿通道**（测光惯例），三个通道各自保留差异 —— 那正是分通道成色本身。
    """
    x = np.asarray(d['logE'], np.float64)
    center = 0.5 * (float(x.min()) + float(x.max()))
    base = (np.log10(mid_lin) - np.log10(0.18)) * sc_log + center
    target = float(np.clip(mid_disp, 1e-3, 0.999))          # ⚠ f 返回的是**显示域**，靶也要显示域
    inv = 1.0 / 2.2
    t_max = 10.0 ** (-np.asarray(d['d_min']) * sc)
    t_min = 10.0 ** (-np.asarray(d['d_max']) * sc)
    rng = np.maximum(t_max - t_min, 1e-5)
    curve = np.stack([d['r'], d['g'], d['b']], -1)

    out = np.zeros(3)
    for ch in range(3):
        def f(mapped):
            dens = float(np.interp(mapped, x, curve[:, ch]))
            t = 10.0 ** (-dens * sc)
            v = 1.0 - (t - t_min[ch]) / rng[ch]
            return float(np.clip(v, 1e-3, 0.999) ** inv)
        lo, hi = -14.0, 8.0
        for _ in range(80):                    # f 随 mapped 单调递增 ⇒ 二分
            m = 0.5 * (lo + hi)
            if f(m) < target:
                lo = m
            else:
                hi = m
        out[ch] = 0.5 * (lo + hi) - base
    return out


def density(disp, stock='portra400', scalar=None, cfg=C):
    r"""【乙】完整密度引擎（Emulsifier 那套 + 我们补的**印相曝光**）。`disp`(0~1) 进、出。

        ① 线性光 → `log10` = 曝光量
        ② 以中灰 0.18 为锚平移到曲线中心（**不改写位置**），再叠一个**印相曝光**（让中灰守恒）
        ③ R/G/B **各自**插值一条密度曲线
        ④ `透射率 = 10^(−密度 × scalar)`
        ⑤ 用该胶片自己的 `d_min/d_max` 归一化（**分通道** —— 这就是那片橙色片基）
        ⑥ 负片 ⇒ `1 − x`；再上 1/2.2
    """
    d = load_curve(stock)
    sc = float(getattr(cfg, 'DENSITY_SCALAR', 0.60) if scalar is None else scalar)
    sc_log = float(getattr(cfg, 'DENSITY_LOGE_SCALE', 2.0))
    off = _print_offset(d, sc, sc_log)
    lin = np.maximum(color.s2l(np.clip(disp, 0.0, 1.0)), 1e-6)
    log_e = np.log10(lin)
    center = 0.5 * (float(d['logE'].min()) + float(d['logE'].max()))
    mapped = (log_e - np.log10(0.18)) * sc_log + center + off.reshape(1, 1, 3)
    grid = d['logE']
    dens = np.stack([np.interp(mapped[..., 0], grid, d['r']),
                     np.interp(mapped[..., 1], grid, d['g']),
                     np.interp(mapped[..., 2], grid, d['b'])], -1)
    t = 10.0 ** (-dens * sc)
    t_max = (10.0 ** (-np.asarray(d['d_min']) * sc)).reshape(1, 1, 3)
    t_min = (10.0 ** (-np.asarray(d['d_max']) * sc)).reshape(1, 1, 3)
    out = (t - t_min) / np.maximum(t_max - t_min, 1e-5)
    out = 1.0 - out                                   # 负片翻正
    out = np.clip(out, 1e-3, 0.999) ** (1.0 / 2.2)
    return np.clip(out, 0.0, 1.0)


def blend(a, b, w):
    """把 b 按权重 w 混到 a 上（w=0 ⇒ 逐位等于 a）。"""
    w = float(np.clip(w, 0.0, 1.0))
    return a if w <= 1e-6 else a * (1.0 - w) + b * w
