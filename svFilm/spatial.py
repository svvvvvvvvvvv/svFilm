# -*- coding: utf-8 -*-
"""空间域层 —— 颗粒 / 黑柔(Bloom) / Halation。塞不进 LUT 的三件。

为什么必须独立一层：L2 的 .cube 是"逐像素的颜色查找表"，只能装
"颜色对比 + 高光压缩"这类点运算。而下面三件事都是**邻域运算**：

  * 颗粒 Grain    —— 银盐颗粒的随机性，随亮度变化、随区域抑制
  * 黑柔 Bloom    —— 镜头/滤镜像差把亮部往外扩散（Black Pro Mist 的招牌）
  * Halation      —— 光穿过片基被反射回乳剂，在亮部周围散出红橙晕圈

位置：L2 之后、L3 之前。输入输出都是 disp 域。
强度由"卷"给（`stocks.py`），这里只负责算法；`resolve()` 负责把卷的强度
叠到 config 默认上。

三条纪律：
  1) 三个效果都**不改中灰**（加的量都在局部，且乘性/零均值）；
  2) 颗粒用**固定种子**，同一张图每次结果一致（可复现，别用随机种子）；
  3) 掩膜全部用"平滑单调量"，不用硬阈值 —— 硬阈值会在画面里留下边缘。
"""
from __future__ import annotations

import numpy as np

from . import color
from . import config as C


def resolve(cfg=C, stock=None):
    """把卷的空间参数叠到 config 默认上（卷优先），返回三个 dict。"""
    p = dict(
        grain=dict(enable=cfg.GRAIN_ENABLE, amount=cfg.GRAIN_AMOUNT, size=cfg.GRAIN_SIZE,
                   chroma=cfg.GRAIN_CHROMA, skin_suppress=cfg.GRAIN_SKIN_SUPPRESS,
                   detail_suppress=cfg.GRAIN_DETAIL_SUPPRESS, dark_floor=cfg.GRAIN_DARK_FLOOR),
        bloom=dict(enable=cfg.BLOOM_ENABLE, amount=cfg.BLOOM_AMOUNT, radius=cfg.BLOOM_RADIUS,
                   thr_lo=cfg.BLOOM_THR_LO, thr_hi=cfg.BLOOM_THR_HI,
                   warmth=cfg.BLOOM_WARMTH, veil=cfg.BLOOM_VEIL),
        halation=dict(enable=cfg.HALATION_ENABLE, amount=cfg.HALATION_AMOUNT,
                      radius=cfg.HALATION_RADIUS, thr_lo=cfg.HALATION_THR_LO,
                      thr_hi=cfg.HALATION_THR_HI, color=list(cfg.HALATION_COLOR),
                      radius_ratios=list(cfg.HALATION_RADIUS_RATIOS)),
    )
    if stock:
        sp = stock.get('spatial') or {}
        for k, v in sp.items():
            if k in p:
                p[k].update(v)
    return p


# ---------------- 基础件 ----------------
def _blur(a, sigma):
    """高斯模糊（float64 进、float64 出）。

    大半径（>4px）走"先缩图 -> 小核模糊 -> 再放大"：sigma 20 直接算核要 ~170 阶，
    又慢又没必要 —— 辉光本来就是低频，缩一半再糊肉眼看不出差别，快好几倍。
    """
    if sigma is None or sigma <= 0:
        return np.asarray(a, np.float64)
    import cv2
    x = np.ascontiguousarray(a, dtype=np.float32)
    f = 1.0
    if sigma > 4.0:
        f = sigma / 2.0
        h, w = x.shape[:2]
        nw, nh = max(1, int(round(w / f))), max(1, int(round(h / f)))
        x = cv2.resize(x, (nw, nh), interpolation=cv2.INTER_AREA)
        sigma = sigma / f
    y = cv2.GaussianBlur(x, (0, 0), float(sigma))
    if f != 1.0:
        h, w = a.shape[:2]
        y = cv2.resize(y, (w, h), interpolation=cv2.INTER_LINEAR)
    return y.astype(np.float64)


def _local_detail(disp):
    """高频能量（0~1，按 P95 归一）。用来知道"这块本来就纹理多"。"""
    g = color.gray_of(disp)
    hi = g - _blur(g, 2.0)
    n = float(np.percentile(np.abs(hi), 95)) + 1e-9
    return np.clip(np.abs(hi) / n, 0.0, 1.0)


def _skin_mask_fast(disp):
    """肤色软掩膜（0~1）。故意不用 Lab（全图 Lab 要 0.8s），
    用显示域的"红>绿>蓝 + 有彩度"近似 —— 只用来决定"这里少撒点颗粒"，不需要精确。"""
    r = disp[..., 0]
    g = disp[..., 1]
    b = disp[..., 2]
    m = color.smoothstep(r - b, 0.035, 0.150)          # 红明显高于蓝
    m *= color.smoothstep(r - g, 0.010, 0.080)         # 红高于绿
    m *= 1.0 - color.smoothstep(color.sat_hsv(disp), 0.62, 0.92)   # 太艳的（衣服/花）不算
    m *= color.smoothstep(color.gray_of(disp), 0.10, 0.22)         # 太暗不算
    return m


# ---------------- 乙：颗粒 ----------------
def grain(disp, p, cfg=C):
    if not p.get('enable') or p.get('amount', 0.0) <= 0.0:
        return np.clip(disp, 0.0, 1.0), dict(applied=False)

    cur = np.clip(disp, 0.0, 1.0)
    g = color.gray_of(cur)
    h, w = g.shape

    # 亮度包络：中间调最明显，两端收（胶片就是这样，不是均匀撒盐）
    env = np.clip(4.0 * g * (1.0 - g), 0.0, 1.0) ** 0.55
    env *= color.smoothstep(g, float(p.get('dark_floor', 0.03)), float(p.get('dark_floor', 0.03)) + 0.07)
    # ★ 高光端**精确归零**（09-13 调研修 bug）：原来只压 0.70，白墙/天空还留 30% 颗粒在动。
    #   物理：密度饱和区没有可显影的银盐。外面（LIMO `applyGrainAsExposure`、
    #   Emulsifier `grain_mask=(luma^0.5)(1-luma)^1.5`）两端都精确为 0。
    env *= 1.0 - color.smoothstep(g, float(cfg.GRAIN_HI_LO), float(cfg.GRAIN_HI_HI))

    # 区域抑制：脸和高细节处少撒
    if p.get('skin_suppress', 0.0) > 0.0:
        env *= 1.0 - float(p['skin_suppress']) * _skin_mask_fast(cur)
    if p.get('detail_suppress', 0.0) > 0.0:
        env *= 1.0 - float(p['detail_suppress']) * _local_detail(cur)

    # 噪声：单色为主 + 一点彩噪；高斯模糊出"颗粒尺度"，再归一化回 std≈1
    rng = np.random.default_rng(int(cfg.GRAIN_SEED))
    sigma = float(p.get('size', 1.2))
    mono = _blur(rng.normal(0.0, 1.0, (h, w)), sigma)
    mono /= float(mono.std()) + 1e-9
    ck = float(p.get('chroma', 0.0))
    if ck > 0.0:
        col = np.stack([_blur(rng.normal(0.0, 1.0, (h, w)), sigma) for _ in range(3)], axis=-1)
        col /= float(col.std()) + 1e-9
        col -= col.mean(axis=-1, keepdims=True)          # 去掉亮度分量：只留"彩"
        n = mono[..., None] * (1.0 - ck) + col * ck
    else:
        n = mono[..., None]

    # 乘性叠加：黑还是黑（0 乘任何数还是 0），中灰也不会被整体推走
    out = cur * (1.0 + float(p['amount']) * env[..., None] * n)
    out = np.clip(out, 0.0, 1.0)
    return out, dict(applied=True, amount=float(p['amount']),
                     env_mean=float(env.mean()), seed=int(cfg.GRAIN_SEED))


# ---------------- 丙：黑柔 / Bloom ----------------
def bloom(disp, p, cfg=C):
    if not p.get('enable') or p.get('amount', 0.0) <= 0.0:
        return np.clip(disp, 0.0, 1.0), dict(applied=False)

    cur = np.clip(disp, 0.0, 1.0)
    lin = color.s2l(cur)
    g = color.gray_of(cur)
    radius = float(p.get('radius', 22.0))

    # 只有够亮的地方才发光（软阈值，避免硬边）
    m = color.smoothstep(g, float(p.get('thr_lo', 0.74)), float(p.get('thr_hi', 0.93)))
    hot = lin * m[..., None]
    glow = np.stack([_blur(hot[..., i], radius) for i in range(3)], axis=-1)

    # 辉光偏暖一点（镜头/柔光镜的常见表现）
    wa = float(p.get('warmth', 0.0))
    glow = glow * np.array([1.0 + 0.10 * wa, 1.0 + 0.02 * wa, 1.0 - 0.10 * wa])

    out_lin = lin + float(p['amount']) * glow

    # 黑柔特征：整体往"模糊版"靠一点 => 轻微提灰、降对比（Black Pro Mist 那口气）
    veil = float(p.get('veil', 0.0))
    if veil > 0.0:
        base = np.stack([_blur(out_lin[..., i], radius * 0.55) for i in range(3)], axis=-1)
        out_lin = out_lin * (1.0 - veil) + base * veil

    out = np.clip(color.l2s(out_lin), 0.0, 1.0)
    return out, dict(applied=True, amount=float(p['amount']), radius=radius, veil=veil)


# ---------------- 丁：Halation ----------------
def halation(disp, p, cfg=C):
    if not p.get('enable') or p.get('amount', 0.0) <= 0.0:
        return np.clip(disp, 0.0, 1.0), dict(applied=False)

    cur = np.clip(disp, 0.0, 1.0)
    lin = color.s2l(cur)
    g = color.gray_of(cur)

    # 亮部才有能量；晕圈出现在亮区的**外侧**（片基把光散回去）
    bright = color.smoothstep(g, float(p.get('thr_lo', 0.78)), float(p.get('thr_hi', 0.99)))
    src = bright[..., None] * lin

    # ★ 分通道扩散半径（09-13 调研修正）：红光穿透片基散射得最远，蓝光几乎不散。
    #   出处：LIMO `FilmShaderCommon.h` RED/GREEN/BLUE_PENETRATION = 0.88/0.10/0.02；
    #         spektrafilm 也是三通道各自独立的散射 sigma。
    #   ⇒ 红边变成"外圈红、里层偏白"，而不是把亮部整块叠一层橙。
    r0 = float(p.get('radius', 18.0))
    ratios = p.get('radius_ratios') or [1.0, 0.45, 0.15]
    spread = np.stack([_blur(src[..., i], r0 * float(ratios[i])) for i in range(3)], axis=-1)
    ring = spread * (1.0 - np.clip(bright * 1.25, 0.0, 1.0))[..., None]

    col = np.asarray(p.get('color', [1.0, 0.30, 0.12]), np.float64)
    out_lin = lin + float(p['amount']) * ring * col
    out = np.clip(color.l2s(out_lin), 0.0, 1.0)
    return out, dict(applied=True, amount=float(p['amount']), radius=r0,
                     radius_ratios=[float(x) for x in ratios])


# ---------------- 编排 ----------------
def apply(disp, cfg=C, stock=None):
    """按 光学 → 乳剂的物理顺序：Halation（片基）→ Bloom（镜头）→ Grain（银盐）。"""
    p = resolve(cfg, stock)
    cur = np.clip(disp, 0.0, 1.0)
    cur, h_info = halation(cur, p['halation'], cfg)
    cur, b_info = bloom(cur, p['bloom'], cfg)
    cur, g_info = grain(cur, p['grain'], cfg)
    info = dict(halation=h_info, bloom=b_info, grain=g_info,
                any=bool(h_info['applied'] or b_info['applied'] or g_info['applied']),
                stock=(stock or {}).get('name'))
    return cur, info
