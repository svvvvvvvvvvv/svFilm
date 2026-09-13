# -*- coding: utf-8 -*-
"""L4 —— 护栏层。只做"不许超过"，永远不做"必须等于"。

"必须等于某个值"是上一代管线最贵的错误：为了让成片达到某个靶，
它会去洗淡衣服、拉爆高光。护栏只划上限，达到了就收手。
"""
import numpy as np

from . import color
from . import config as C


def _white_frac(disp):
    return float(np.mean(color.gray_of(disp) >= (254.0 / 255.0)))


def _solve_scale(disp, cap, lo=None, hi=1.0, iters=None):
    """二分求一个整体缩放系数，使死白占比 <= cap。只往小找。（下界/迭代数见 config，P1-3）"""
    if lo is None:
        lo = float(getattr(C, 'GUARD_SOLVE_LO', 0.55))
    if iters is None:
        iters = int(getattr(C, 'GUARD_SOLVE_ITERS', 14))
    lin = color.s2l(np.clip(disp, 0.0, 1.0))
    if _white_frac(disp) <= cap:
        return 1.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        cand = color.l2s(np.clip(lin * mid, 0.0, None))
        if _white_frac(np.clip(cand, 0.0, 1.0)) <= cap:
            lo = mid
        else:
            hi = mid
    return lo


def _cap_chroma(disp, cap):
    lab = color.to_lab(np.clip(disp, 0.0, 1.0))
    c90 = float(np.percentile(color.chroma(lab), 90))
    if c90 <= cap or c90 <= C.NOISE_FLOOR:
        return disp, False
    f = float(np.clip(cap / c90, 0.0, 1.0))
    lab[..., 1] *= f
    lab[..., 2] *= f
    return np.clip(color.from_lab(lab), 0.0, 1.0), True


def enforce(disp, cfg=C):
    """返回 (disp_out, info)。info 记录每趟做了什么，便于回溯。"""
    out = np.clip(disp, 0.0, 1.0)
    log = []
    for _ in range(max(1, cfg.GUARD_MAX_PASS)):
        touched = False

        wf = _white_frac(out)
        if cfg.CAP_WHITE_FRAC is not None and wf > cfg.CAP_WHITE_FRAC:
            k = _solve_scale(out, cfg.CAP_WHITE_FRAC)
            if k < 0.999:
                out = np.clip(color.l2s(np.clip(color.s2l(out) * k, 0.0, None)), 0.0, 1.0)
                log.append('scale %.4f (dead-white %.3f%% -> %.3f%%)'
                           % (k, wf * 100.0, _white_frac(out) * 100.0))
                touched = True

        if cfg.CAP_CHROMA_C90 is not None:
            out, did = _cap_chroma(out, cfg.CAP_CHROMA_C90)
            if did:
                log.append('chroma cap %.1f' % cfg.CAP_CHROMA_C90)
                touched = True

        if not touched:
            break

    return out, dict(actions=log,
                     dead_white_frac=_white_frac(out),
                     chroma_c90=color.chroma_c90(out))
