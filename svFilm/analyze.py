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

    # 有效曝光偏差（相对中灰靶），只用来做报告/限幅，不用来决定"要不要动"
    ev_est = float(np.log2(max(gm, C.NOISE_FLOOR) / C.TGT_MID))

    # 决策：超中灰靶 + 死区才动手；其余一律不动（含"本来就暗"的图）
    if gm > C.TGT_MID + C.MID_DEADZONE:
        decision = 'compress'
    elif gm < C.TGT_MID - C.MID_DEADZONE:
        decision = 'below'
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
    """一行给人看的摘要（讲人话，不甩代号）"""
    return (
        "中灰 {:.0f}/255 (L*{:.0f}) | 黑点 {:.0f} | 白点 {:.0f} | "
        "死白 {:.2%} | 彩度P90 {:.1f} | {}".format(
            rep['gray_mid'] * 255, rep['L_pcts'][C.PCT_MID],
            rep['gray_black'] * 255, rep['gray_white'] * 255,
            rep['dead_white_frac'], rep['chroma_c90'], rep['decision'],
        )
    )
