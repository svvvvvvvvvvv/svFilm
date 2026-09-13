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


def _soft_cap_L(L, cap, soft):
    r"""超过 `cap` 的部分**平滑收回**，不是一刀切。

        excess = max(L − cap, 0)
        L_out  = L − excess + soft · tanh(excess / soft)

    三条性质（都是"不许超过"要的）：
      · excess = 0 ⇒ L_out = L（**暗的一律不动**，只压不提）
      · 在 cap 处**值与斜率都连续** ⇒ 不会压出台阶 / 断层
        （对应 SV 手册那句「发丝边上还能看出一丝丝分层，不是一条死白线」）
      · 无论多亮，收回来后**永远到不了 cap + soft** ⇒ 是真上限，不是渐近无限的软塌
    """
    e = np.clip(np.asarray(L, np.float64) - float(cap), 0.0, None)
    s = max(float(soft), 1e-6)
    return L - e + s * np.tanh(e / s)


def _feather(m, rel, hw):
    """掩膜羽化：高斯 σ = `rel` × **画面短边**。不羽化会在掩膜边界压出一道接缝。"""
    import cv2
    sig = max(1.0, float(rel) * float(min(hw)))
    return np.clip(cv2.GaussianBlur(np.asarray(m, np.float32), (0, 0), sig), 0.0, 1.0)


def cap_face_bg(disp, cfg=C, masks=None):
    r"""L4 的两道「不许超过」：**脸 ≤ `FACE_CAP_L`**、**背景 ≤ 脸上限 − `BG_CAP_REL_L`**。

    设计主线（SV 09-14 原话）＝「先锚点人脸会好看的亮度，再去算背景该有的亮度」：
    **脸是锚**（定死的 68），**背景由脸推算**（68 − 17 = 51），两个都是**上限**：
      · **只压不提** —— 暗的一律不动（所以"背景一个像素不碰"那条指**提亮**，这里不破）
      · **软压不硬切** —— `_soft_cap_L`，压不出断层
      · **中间地带（头发 / 衣服 / 身体皮肤）不动** —— 头发本来就该深，压它没意义
    """
    d = np.clip(disp, 0.0, 1.0)
    info = dict(applied=False)
    if not bool(getattr(cfg, 'CAP_FACE_BG_ENABLE', True)):
        info['reason'] = 'off'
        return d, info
    if masks is None:                                   # 调用方没给 ⇒ 自己解析一次（慢一点，但接口干净）
        try:
            from . import face
            masks = face.parse(d)['masks']
        except Exception as e:                          # noqa: BLE001 缺依赖/没脸 ⇒ 优雅降级
            info.update(reason='no_mask', err='%s: %s' % (type(e).__name__, e))
            return d, info

    cap_face = float(getattr(cfg, 'FACE_CAP_L', 68.0))
    cap_bg = cap_face - float(getattr(cfg, 'BG_CAP_REL_L', 17.0))
    soft = float(getattr(cfg, 'CAP_SOFT_L', 6.0))
    rel_f = float(getattr(cfg, 'CAP_FEATHER_REL', 0.006))
    hw = d.shape[:2]

    wf = _feather(masks.get('face_skin', 0.0), rel_f, hw)
    wb = _feather(masks.get('bg', 0.0), rel_f, hw)
    if float(np.max(wf)) < 1e-3 and float(np.max(wb)) < 1e-3:
        info['reason'] = 'empty_mask'
        return d, info

    lin = color.s2l(d)
    L = color.L_of_lin(color.Y_of(lin))
    Lc = L + wf * (_soft_cap_L(L, cap_face, soft) - L) \
           + wb * (_soft_cap_L(L, cap_bg, soft) - L)

    out = d.copy()
    wsel = (wf > 1.0e-6) | (wb > 1.0e-6)
    if np.any(wsel):
        out[wsel] = color.retone_L(lin[wsel], Lc[wsel])
    info.update(applied=True, cap_face=cap_face, cap_bg=cap_bg, soft=soft,
                face_cov=float(np.mean(wf)), bg_cov=float(np.mean(wb)),
                face_L=float(np.median(L[wf > 0.5])) if np.any(wf > 0.5) else None,
                bg_L=float(np.median(L[wb > 0.5])) if np.any(wb > 0.5) else None)
    return out, info


def enforce(disp, cfg=C, masks=None):
    """返回 (disp_out, info)。info 记录每趟做了什么，便于回溯。"""
    # ★ 两道「不许超过」放在最前：先把脸/背景的过曝收掉，
    #   后面的全局死白/彩度护栏往往就不必再动了。
    out, cap_info = cap_face_bg(disp, cfg, masks=masks)
    log = []
    if cap_info.get('applied'):
        log.append('face/bg cap (face<=%.1f bg<=%.1f)' % (cap_info['cap_face'],
                                                           cap_info['cap_bg']))
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
                     chroma_c90=color.chroma_c90(out),
                     face_bg_cap=cap_info)
