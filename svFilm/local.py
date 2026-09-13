# -*- coding: utf-8 -*-
"""L3 —— 局部层。默认做两件事，都只碰色度（不改明度）：

  ① 肤色保护 —— 风格层会把颜色往胶片方向收，肤色最容易被收坏。
     拿 L1 修正完的成片当参考，哪里是肤色、就把那里的色度按比例还给参考值。
     明度用当前的，色度用参考的 —— 只还"色"，不还"亮"。

  ② 肤色正向达标 —— 只在"保护"补不回来时起作用（09-13 SV 拍板「全」档 + 过亮门）。
     目标 = 大师脸区实测中位（a* 16.3 / b* 18.5，`_debug/analysis/skin_master_ruler.md`）。
     **逐像素**：每个皮肤像素自己看"离目标还差多少"，差多少补多少；**只向上补**；
     单像素位移设上限（a* 8 / b* 12，护唇妆/腮红）；**两轴都低才补**（某轴已高过目标
     就完全不碰 —— 挡粉衣服/木色被顺手推暖）；过亮门**逐像素**收力
     （L* ≤ 73.3 全强度 / ≥ 83.7 不补，逆光高调脸的高光部分自己不补、暗部照补）。

（真正的大局：局部后面还会长柔化/颗粒/黑柔，那是空间域的活，
  用独立一层做，不塞进颜色层。）
"""
import numpy as np

from . import color
from . import config as C


def skin_mask(disp):
    """软掩膜 0~1：色相 5~55 度、有彩度、明度在中段。门限见 `config.SKIN_MASK_*`（P1-3）。"""
    lab = color.to_lab(np.clip(disp, 0.0, 1.0))
    h = color.hue_deg(lab)
    c = color.chroma(lab)
    L = lab[..., 0]
    _hlo = tuple(getattr(C, 'SKIN_MASK_HUE_LO', (2.0, 14.0)))
    _hhi = tuple(getattr(C, 'SKIN_MASK_HUE_HI', (46.0, 66.0)))
    _clo = tuple(getattr(C, 'SKIN_MASK_C_LO', (5.0, 13.0)))
    _chi = tuple(getattr(C, 'SKIN_MASK_C_HI', (70.0, 95.0)))
    _llo = tuple(getattr(C, 'SKIN_MASK_L_LO', (12.0, 22.0)))
    _lhi = tuple(getattr(C, 'SKIN_MASK_L_HI', (86.0, 95.0)))
    m = color.smoothstep(h, _hlo[0], _hlo[1]) * (1.0 - color.smoothstep(h, _hhi[0], _hhi[1]))
    m *= color.smoothstep(c, _clo[0], _clo[1]) * (1.0 - color.smoothstep(c, _chi[0], _chi[1]))
    m *= color.smoothstep(L, _llo[0], _llo[1]) * (1.0 - color.smoothstep(L, _lhi[0], _lhi[1]))
    return m


def _protect(ref_disp, disp, cfg):
    """肤色保护：色度按 SKIN_PROTECT_STRENGTH 往 L1 的参考值还。"""
    cur = np.clip(disp, 0.0, 1.0)
    ref = np.clip(ref_disp, 0.0, 1.0)
    m = skin_mask(cur)                     # 只这一次是全图 Lab
    cov = float(np.mean(m))
    if cov < float(getattr(C, 'SKIN_PROTECT_MIN_COV', 1.0e-4)):
        return cur, dict(skin_cov=0.0)
    sel = m > float(getattr(C, 'SKIN_PROTECT_SEL', 1.0e-3))
    if not np.any(sel):
        return cur, dict(skin_cov=cov)
    lab_c = color.to_lab(cur[sel])
    lab_r = color.to_lab(ref[sel])
    w = (m[sel] * cfg.SKIN_PROTECT_STRENGTH)[:, None]
    lab_c[:, 1:] = lab_c[:, 1:] * (1.0 - w) + lab_r[:, 1:] * w
    out = cur.copy()
    out[sel] = np.clip(color.from_lab(lab_c), 0.0, 1.0)
    return out, dict(skin_cov=cov)


def skin_floor(disp, cfg=C):
    """肤色正向达标：**逐像素**把"低于大师脸的皮肤"抬到档上。只动色度。

    每个皮肤像素自己算"离目标还差多少"，差多少补多少；三个门决定它使多大劲：
      · 软掩膜 m           —— 越像皮肤越使劲；
      · 过亮门（逐像素）    —— 亮度超过大师脸区分布上端就收力（逆光/高调脸不硬补）；
      · 两轴都低门          —— 某一轴已经高过目标（粉衣服/木色/唇妆）就完全不碰。

    ⚠ 不用"全图皮肤中位"这类整体统计量。试过，不行：掩膜会把粉衣服/木色/路面算进来
    （实测一张 15.3% 的像素被判成皮肤），中位被拉低之后**人脸会被推过头**
    （实测 b* 冲到 20.4，目标 18.5）。逐像素做就没有这个问题。
    """
    if not getattr(cfg, 'SKIN_FLOOR', False):
        return np.clip(disp, 0.0, 1.0), dict(applied=False, reason='off')

    d = np.clip(disp, 0.0, 1.0)
    m = skin_mask(d)
    if int((m > 0.5).sum()) < int(getattr(C, 'SKIN_MASK_MIN_PX', 200)):
        return d, dict(applied=False, reason='no_skin')
    sel = m > float(getattr(C, 'SKIN_MASK_SEL', 0.02))

    lab = color.to_lab(d[sel])
    a, b, L = lab[:, 1], lab[:, 2], lab[:, 0]
    aT, bT = cfg.SKIN_FLOOR_A, cfg.SKIN_FLOOR_B

    gate_L = 1.0 - color.smoothstep(L, cfg.SKIN_FLOOR_L_LO, cfg.SKIN_FLOOR_L_HI)
    excess = np.maximum(a - aT, b - bT)
    both_low = 1.0 - color.smoothstep(excess, 0.0, cfg.SKIN_FLOOR_EXCESS)
    w = m[sel] * gate_L * both_low

    info = dict(applied=False, a_med=float(np.median(a)), b_med=float(np.median(b)),
                L_med=float(np.median(L)), gate=float(np.mean(gate_L)),
                touched=float(np.mean(w > 1e-3)))
    if float(np.max(w)) < 1e-4:
        info['reason'] = 'nothing_to_do'
        return d, info

    lab[:, 1] = a + np.clip(aT - a, 0.0, cfg.SKIN_FLOOR_A_MAX) * w
    lab[:, 2] = b + np.clip(bT - b, 0.0, cfg.SKIN_FLOOR_B_MAX) * w
    out = d.copy()
    out[sel] = np.clip(color.from_lab(lab), 0.0, 1.0)
    info['applied'] = True
    return out, info


def apply(ref_disp, disp, cfg=C):
    """ref_disp = L1 修正后的成片；disp = 当前（过完风格 + 空间域）的成片。"""
    out = np.clip(disp, 0.0, 1.0)
    info = {}
    if cfg.SKIN_PROTECT and cfg.SKIN_PROTECT_STRENGTH > 0.0:
        out, pinfo = _protect(ref_disp, out, cfg)
        info.update(pinfo)
    else:
        info['skin_cov'] = 0.0
    if getattr(cfg, 'SKIN_FLOOR', False):
        out, finfo = skin_floor(out, cfg)
        info['skin_floor'] = finfo
    return out, info
