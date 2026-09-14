# -*- coding: utf-8 -*-
r"""L3 —— 局部层。两件事，顺序 = **色度 → 保护**：
  ① 肤色保护 —— 风格层会把颜色往胶片方向收，肤色最容易被收坏。
     拿 L1 修正完的成片当参考，哪里是肤色、就把那里的色度按比例还给参考值。
     明度用当前的，色度用参考的 —— 只还"色"，不还"亮"。

  ② 肤色正向达标 —— 只在"保护"补不回来时起作用（09-13 SV 拍板「全」档 + 过亮门）。
     目标 = 大师脸区实测中位（a* 16.3 / b* 18.5，`_debug/analysis/skin_master_ruler.md`）。
     **逐像素**：每个皮肤像素自己看"离目标还差多少"，差多少补多少；**只向上补**；
     单像素位移设上限（a* 8 / b* 12，护唇妆/腮红）；**两轴都低才补**（某轴已高过目标
     就完全不碰 —— 挡粉衣服/木色被顺手推暖）；过亮门**逐像素**收力
     （L* ≤ 73.3 全强度 / ≥ 83.7 不补，逆光高调脸的高光部分自己不补、暗部照补）。

  ⚠ **09-14 SV：「把脸部立体感的部分删掉」** —— 原来还有第 ③ 件事「脸」（`face_tone`：
  提亮 + 立体感 A1），**已整段删除**。原因：A1 按**整个脸框**放大明暗、且**只封亮部不封暗部**
  ⇒ 框里的眉毛/眼睛/睫毛被越压越黑（实测 DSCF0830 最暗一档 5.2 → 2.3）。
  位置（提亮）早在同一天就搬去 `io.anchor_ev`（一条全局曲线）了。

（真正的大局：局部后面还会长柔化/颗粒/黑柔，那是空间域的活，
  用独立一层做，不塞进颜色层。）
"""
import numpy as np

from . import color
from . import face
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


def white_micro(disp, cfg=C, amount=0.0):
    r"""★ 09-14 SV 选「丙」：给**白区**补回一点中尺度微反差。

    为什么单独写这一道：`_debug/lab_ours_vs_lujing.py` 量到我们的**白区层次**
    （近白 + 低彩那块的中尺度局部对比 σ4）只有 **1.34**，鹿井那边 **4.69**。
    逐层追踪查明：**我们真正磨掉的只有 0.41**（入口 1.75 → 成片 1.34），
    而且**磨得最多的是空间层（黑柔）**；剩下 3.35 是**内容差**
    （鹿井的"白"是天空/阳光下的白墙，天生有层次）。

    ⇒ 所以这道是**"照大师的数值去补"**，不是修 bug。**amount 可调**：
        0.0 = 不动（白区层次保持 1.34）
        ~0.5 = 折中
        1.0 = 追满（把白区层次往 4.69 推）

    ⚠ 只动 **L\***，不碰 a/b（不脏色）；只落在**白区掩膜**里（近白 + 低彩，羽化）；
    并且**高光不许吹白**（`WHITE_MICRO_TOP` 封顶）。
    """
    if amount <= 1e-4:
        return disp, dict(applied=False, reason='off', amount=amount)
    import cv2
    lab = color.to_lab(np.clip(disp, 0.0, 1.0))
    L = lab[..., 0]
    C = np.sqrt(lab[..., 1] ** 2 + lab[..., 2] ** 2)
    L_lo = float(getattr(cfg, 'WHITE_MICRO_L', 82.0))       # 白区下界（L*）
    C_hi = float(getattr(cfg, 'WHITE_MICRO_C', 12.0))       # 白区彩度上限
    sel = ((L >= L_lo) & (C <= C_hi)).astype(np.float32)
    if float(sel.mean()) < 1e-4:
        return disp, dict(applied=False, reason='no_white', amount=amount)
    _h, _w = L.shape
    _sig = max(3.0, float(getattr(cfg, 'WHITE_MICRO_FEATHER_REL', 0.02)) * _w)
    w = cv2.GaussianBlur(sel, (0, 0), _sig)
    _sig2 = float(getattr(cfg, 'WHITE_MICRO_SIGMA', 4.0))
    detail = L.astype(np.float32) - cv2.GaussianBlur(L.astype(np.float32), (0, 0), _sig2)
    _mx = float(getattr(cfg, 'WHITE_MICRO_MAX', 6.0))
    gain = float(np.clip(amount, 0.0, 2.0)) * float(getattr(cfg, 'WHITE_MICRO_AMT', 1.6))
    d = np.clip(detail * gain, -_mx, _mx)
    top = float(getattr(cfg, 'WHITE_MICRO_TOP', 99.0))
    L2 = np.clip(L + w * d, 0.0, top)
    lab[..., 0] = L2
    out = np.clip(color.from_lab(lab), 0.0, 1.0)
    return out, dict(applied=True, amount=amount, white_cov=float(sel.mean()),
                     gain=gain, sigma=_sig2, top=top)


def face_depth(disp, cfg=C):
    r"""★ 09-14 重做「脸的层次」—— **只作用皮肤 + 暗部有底 + 死区**。

    旧的那版（`face_tone`，同日删除）翻车在三条（`_debug/lab_face_dark.py` 实测）：
      ① 力道**按整个脸框**下 ⇒ 框里的皮肤、眉毛、眼睛、**头发一起被拉开**；
      ② **只封亮部**（`FACE_TOP_CAP` L\*97）、**暗部一个底都没有** ⇒ 本来就深的眉毛一拉贴死
         （0830 最暗一档 5.2 → **2.3**，压掉一半）；
      ③ `k` 由**皮肤**跨度算 —— 皮肤被黑柔+颗粒压平 ⇒ k 变大 ⇒ 皮肤没拉够、眉毛被拉过头。
    这一版一条对一条地改：
      ① 权重 = **真分割的 `face_skin`**（羽化）⇒ 眉毛/眼睛/头发**一个像素不碰**；
      ② 亮部封顶之外**再补一条暗部下限**（`FACE_BOT_CAP`）；
      ③ **死区**（`FACE_DEPTH_DEAD`）：跨度够就不动；
      ④ 排在**空间层之后**（黑柔+颗粒才是压平脸的主力，得在它后面补）；
      ⑤ 只**放大已有**的明暗（A1），**不编光** —— SV 拍的是逆光/明暗交界，脸本身有明暗。
    靶 `FACE_TGT_SPAN` = 鹿井 26 张的「脸内部跨度」p25 = 35。
    ⇒ ★ 理由：**"跨度"是形状量、不是位置量** —— 位置跨场景不可比，**形状可以抄**（跟影调同一条线）。
    """
    if not bool(getattr(cfg, 'FACE_DEPTH_ENABLE', False)):
        return disp, dict(applied=False, reason='off')
    import cv2
    from . import face as _face
    d = np.clip(disp, 0.0, 1.0)
    try:
        sk = np.asarray(_face.parse(d)['masks']['face_skin'], np.float32)
    except Exception as e:                                   # noqa: BLE001
        return disp, dict(applied=False, reason='parse_fail', err=str(e)[:60])
    sel = sk > 0.5
    if int(sel.sum()) < int(getattr(cfg, 'FACE_DEPTH_MIN_PX', 300)):
        return disp, dict(applied=False, reason='no_skin', n=int(sel.sum()))
    lab = color.to_lab(d)
    L = lab[..., 0].astype(np.float64)
    Ls = float(np.median(L[sel]))
    span = float(np.percentile(L[sel], 90) - np.percentile(L[sel], 10))
    tgt = float(getattr(cfg, 'FACE_TGT_SPAN', 35.0))
    dead = float(getattr(cfg, 'FACE_DEPTH_DEAD', 0.90))
    if span >= tgt * dead:
        return disp, dict(applied=False, reason='span_ok', span=span, target=tgt)
    k = float(np.clip(tgt / max(span, 1e-6), 1.0, float(getattr(cfg, 'FACE_SPAN_KMAX', 2.0))))
    Lx = Ls + k * (L - Ls)
    top = float(getattr(cfg, 'FACE_TOP_CAP', 97.0))
    bot = float(getattr(cfg, 'FACE_BOT_CAP', 12.0))
    Lx = np.where(Lx > L, np.minimum(Lx, np.maximum(L, top)), Lx)      # 变亮：封顶
    Lx = np.where(Lx < L, np.maximum(Lx, np.minimum(L, bot)), Lx)      # 变暗：封底
    _h, _w = L.shape
    sig = max(2.0, float(getattr(cfg, 'FACE_DEPTH_FEATHER_REL', 0.02)) * _w)
    w = np.clip(cv2.GaussianBlur(sk, (0, 0), sig), 0.0, 1.0)
    lab[..., 0] = np.clip(L + w * (Lx - L), 0.0, 100.0)
    out = np.clip(color.from_lab(lab), 0.0, 1.0)
    return out, dict(applied=True, span_before=round(span, 2), span_target=tgt,
                     k=round(k, 3), skin_px=int(sel.sum()), face_med=round(Ls, 1),
                     top=top, bot=bot)


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
    # ★ 09-14 SV 选「C」：白区微反差（补回被黑柔磨掉的那一层）
    _wm = float(getattr(cfg, 'WHITE_MICRO', 0.0) or 0.0)
    out, winfo = white_micro(out, cfg, _wm)
    info['white_micro'] = winfo
    # ★★ 09-14 重做「脸的层次」：排在**空间层之后**（黑柔+颗粒才是压平脸的主力）
    out, dinfo = face_depth(out, cfg)
    info['face_depth'] = dinfo
    return out, info
