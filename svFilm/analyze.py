# -*- coding: utf-8 -*-
"""L0 —— 分析层。只读，绝不改像素。

这一层存在的唯一理由：把"这张图现在长什么样"变成一组数，
后面几层照着这组数决定做什么。判据全部是分位（百分位），
不用均值 —— 均值会被画面内容（大块天空 / 大块暗背景）拖走。
"""
import numpy as np

from . import color
from . import config as C

# 分析用的分位点（0.2% 定黑点，99.8% 定白点，与主流自动曝光一致）
PCTS = (C.PCT_BLACK, 25.0, C.PCT_MID, 75.0, C.PCT_KNEE, C.PCT_WHITE)


def analyze(lin, disp, kind='raw'):
    """lin/disp: (H,W,3) float64。返回纯 dict（可 json 化）。"""
    Y = color.Y_of(lin)
    g = color.gray_of(disp)                     # 显示域灰度 0~1
    L = color.L_of_lin(Y)

    lin_p = color.pct_of(Y, PCTS)
    gray_p = color.pct_of(g, PCTS)
    L_p = color.pct_of(L, PCTS)

    gm = gray_p[C.PCT_MID]
    gw = gray_p[C.PCT_WHITE]
    gb = gray_p[C.PCT_BLACK]

    # 死白：显示域 >= 254/255 的占比
    dead_white = float(np.mean(g >= (254.0 / 255.0)))
    near_white = float(np.mean(g >= 0.90))

    # 有效曝光偏差（相对上限护栏），**只用来做报告**，不用来决定"要不要提亮"
    ev_est = float(np.log2(max(gm, C.NOISE_FLOOR) / C.TGT_MID))

    # 决策：中灰只当**过亮护栏**（"亮得离谱就压回来"），不当提亮靶。
    # ⚠ 09-13 起护栏线 = **GUARD_MID**（大师逐图『中位 L*』的 P95 = 87.0 → 显示域 0.854），
    #   不再是 TGT_MID（灰阶 140）。原因：中位数是**内容量**（画面里暗的东西占多少），
    #   不是曝光量 —— 大师全体有**一半**的片子中位在 L*58 以上，拿 140 当线等于把正常亮片压闷
    #   （园岭实测：入口已经把中位送到贴住相机 161，L1 又把它拽回 138）。
    #   欠曝该在入口按 baseline exposure 补（见 io.load_raw）。
    if gm > C.GUARD_MID:
        decision = 'compress'          # 真的亮得离谱 -> 压回来（这就是"救过曝"）
    elif gm < C.TGT_MID - C.MID_DEADZONE:
        decision = 'below'             # 偏暗：入口补过就不动；入口补不了才允许兜底提亮
    else:
        decision = 'hold'

    return {
        'kind': kind,
        'decision': decision,
        'gray_pcts': gray_p,        # 显示域灰度 0~1
        'gray255': {k: v * 255.0 for k, v in gray_p.items()},
        'L_pcts': L_p,
        'lin_pcts': lin_p,
        'gray_black': gb,
        'gray_mid': gm,
        'gray_white': gw,
        'dead_white_frac': dead_white,
        'near_white_frac': near_white,
        'chroma_c90': color.chroma_c90(disp),
        'ev_est': ev_est,
    }


def summarize(rep):
    """一行给人看的摘要（讲人话，不甩代号）。

    ⚠ 这一行的每一个数都是**原片**（L0 analyze 的输入）的，不是成片的。
    汇报时必须带「原片」二字，否则会被读成"我拿到的图死白这么多"。
    """
    return (
        "原片 中灰 {:.0f}/255 (L*{:.0f}) | 黑点 {:.0f} | 白点 {:.0f} | "
        "死白 {:.2%} | 彩度P90 {:.1f} | {}".format(
            rep['gray_mid'] * 255, rep['L_pcts'][C.PCT_MID],
            rep['gray_black'] * 255, rep['gray_white'] * 255,
            rep['dead_white_frac'], rep['chroma_c90'], rep['decision'],
        )
    )
