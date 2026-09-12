# -*- coding: utf-8 -*-
"""L3 —— 局部层。默认只做一件事：肤色保护。

风格层会把颜色往胶片方向收，肤色是最容易被收坏的地方。
这里拿 L1 修正完的成片当参考，哪里是肤色、就把那里的色度按比例还给参考值。
明度用当前的，色度用参考的 —— 只还"色"，不还"亮"。

（真正的大局：局部后面还会长柔化/颗粒/黑柔，那是空间域的活，
  用独立一层做，不塞进颜色层。）
"""
import numpy as np

from . import color
from . import config as C


def skin_mask(disp):
    """软掩膜 0~1：色相 5~55 度、有彩度、明度在中段。"""
    lab = color.to_lab(np.clip(disp, 0.0, 1.0))
    h = color.hue_deg(lab)
    c = color.chroma(lab)
    L = lab[..., 0]
    m = color.smoothstep(h, 2.0, 14.0) * (1.0 - color.smoothstep(h, 46.0, 66.0))
    m *= color.smoothstep(c, 5.0, 13.0) * (1.0 - color.smoothstep(c, 70.0, 95.0))
    m *= color.smoothstep(L, 12.0, 22.0) * (1.0 - color.smoothstep(L, 86.0, 95.0))
    return m


def apply(ref_disp, disp, cfg=C):
    """ref_disp = L1 修正后的成片；disp = 当前（过完风格）的成片。"""
    if not cfg.SKIN_PROTECT or cfg.SKIN_PROTECT_STRENGTH <= 0.0:
        return np.clip(disp, 0.0, 1.0), dict(skin_cov=0.0)

    cur = np.clip(disp, 0.0, 1.0)
    ref = np.clip(ref_disp, 0.0, 1.0)
    m = skin_mask(cur)                     # 只这一次是全图 Lab
    cov = float(np.mean(m))
    if cov < 1e-4:
        return cur, dict(skin_cov=0.0)

    # 只对掩膜覆盖到的像素做 Lab 往返（全图做要 3 趟，这里是 1 趟全图 + 2 趟稀疏）
    sel = m > 1e-3
    if not np.any(sel):
        return cur, dict(skin_cov=cov)
    lab_c = color.to_lab(cur[sel])
    lab_r = color.to_lab(ref[sel])
    w = (m[sel] * cfg.SKIN_PROTECT_STRENGTH)[:, None]
    lab_c[:, 1:] = lab_c[:, 1:] * (1.0 - w) + lab_r[:, 1:] * w
    out = cur.copy()
    out[sel] = np.clip(color.from_lab(lab_c), 0.0, 1.0)
    return out, dict(skin_cov=cov)
