# -*- coding: utf-8 -*-
"""降噪层 —— 收拾"RAW 提亮后暗部冒出来的色斑与噪点"。

位置：**L1 修正之后、L2 风格之前**。
为什么在这：噪点在原始 RAW 里本来就存在，是 L1 把暗部提亮才把它放大到看得见的；
放在 L1 之后，正好收拾"被放大的那一份"，而且不会干扰 L1 的分位判据。

三条原则（都是被上一代管线坑过总结出来的）：
  1. **只在暗部 + 平坦区下手**：亮部本来就干净，边缘/纹理区一降噪就糊成塑料。
  2. **保边**：用引导滤波（guided filter），以 L* 当引导图 —— 结构在哪它就跟到哪。
  3. **近似零均值**：不是"把像素换成平滑值"，而是"在原值上加回一部分 (平滑值 − 原值)"，
     并且按掩膜加权 ⇒ 全图均值/分位几乎不动（中灰、黑位不漂移，selftest 有这条不变量）。

色度（a*/b*）比亮度下手更重：肉眼嫌脏的"色斑"几乎全在色度通道，亮度通道留细节。
LUT 装不下这个（它是空间域算子），所以单独一层。
"""
from __future__ import annotations

import numpy as np

from . import color


def _guided(I, p, r, eps):
    """引导滤波：以 I 为引导、p 为输入。O(1)/像素（全用盒滤波）。"""
    import cv2
    k = (int(r) * 2 + 1, int(r) * 2 + 1)
    I = I.astype(np.float32)
    p = p.astype(np.float32)
    mI = cv2.boxFilter(I, -1, k, normalize=True)
    mp = cv2.boxFilter(p, -1, k, normalize=True)
    cI = cv2.boxFilter(I * I, -1, k, normalize=True)
    cIp = cv2.boxFilter(I * p, -1, k, normalize=True)
    var = cI - mI * mI
    cov = cIp - mI * mp
    a = cov / (var + float(eps))
    b = mp - a * mI
    ma = cv2.boxFilter(a, -1, k, normalize=True)
    mb = cv2.boxFilter(b, -1, k, normalize=True)
    return ma * I + mb


def _grad_mag(L):
    """L* 的梯度幅值（先轻微模糊，免得把噪点本身当边缘）。"""
    import cv2
    g = cv2.GaussianBlur(L.astype(np.float32), (0, 0), 1.2)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def apply(disp, cfg):
    """disp(0~1) -> (disp', info)。按 config 的 DENOISE_* 走；不开就原样返回。"""
    if not getattr(cfg, 'DENOISE_ENABLE', False):
        return disp, dict(applied=False)

    R = int(getattr(cfg, 'DENOISE_RADIUS', 4))
    eps = float(getattr(cfg, 'DENOISE_EPS', 90.0))
    k_l = float(np.clip(getattr(cfg, 'DENOISE_LUMA', 0.50), 0.0, 1.0))
    k_c = float(np.clip(getattr(cfg, 'DENOISE_CHROMA', 0.90), 0.0, 1.0))

    lab = color.to_lab(np.clip(disp, 0.0, 1.0))
    L = lab[..., 0].astype(np.float32)

    # 亮度掩膜：暗部（L* 低）才有噪点；高光本来就干净，别去动皮肤高光
    w_dark = 1.0 - color.smoothstep(L, cfg.DENOISE_DARK_LO, cfg.DENOISE_DARK_HI)

    # 平坦掩膜分两套：亮度怕糊细节（门槛紧），色度只管色斑（门槛松）。
    # 为什么分开：a*/b* 的噪点本身不产生 L* 梯度，用亮度的边缘门槛去卡色度会"该降的降不了"。
    mag = _grad_mag(L)
    w_flat_l = 1.0 - color.smoothstep(mag, cfg.DENOISE_EDGE_LO, cfg.DENOISE_EDGE_HI)
    w_flat_c = 1.0 - color.smoothstep(mag, cfg.DENOISE_EDGE_LO_C, cfg.DENOISE_EDGE_HI_C)
    w_l = (w_dark * w_flat_l).astype(np.float32)
    w_c = (w_dark * w_flat_c).astype(np.float32)

    # 亮度：自引导（eps 大一点 ⇒ 更保结构），只补一点点
    Ls = _guided(L, L, R, eps)
    lab[..., 0] = L + (k_l * w_l) * (Ls - L)
    # 色度：以 L* 为引导（色斑要贴着结构平滑，但不能跨边缘串色）
    for ch in (1, 2):
        ch_in = lab[..., ch].astype(np.float32)
        cs = _guided(L, ch_in, R, eps)
        lab[..., ch] = ch_in + (k_c * w_c) * (cs - ch_in)

    out = np.clip(color.from_lab(lab), 0.0, 1.0)
    info = dict(applied=True, radius=R, luma=k_l, chroma=k_c,
                mask_mean=float(w_c.mean()), mask_luma_mean=float(w_l.mean()),
                mask_max=float(w_c.max()))
    return out, info
