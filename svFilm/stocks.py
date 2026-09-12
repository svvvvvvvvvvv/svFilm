# -*- coding: utf-8 -*-
"""胶片卷表 —— 一个"卷"就是一份数据：颜色性格 + 空间效果强度。

为什么要做成表：换卷不该改代码，只该改数据。所以这里全是纯数据，
`style.py` 拿它去调颜色，`spatial.py` 拿它去调颗粒/黑柔/Halation。

一个卷包含两组东西，分得很清楚：
  * 颜色（归 L2）：色交叉矩阵 / 暗部色调 / 亮部色调 / 彩度 / 明度对比
  * 空间（归 spatial，LUT 装不下）：颗粒 / 黑柔 Bloom / Halation

`neutral` 是"什么都不做"的基准（恒等），用来做 A/B 的对照栏。
不选卷（stock=None）时用 config.py 里的默认，行为和以前一致。

想要新卷：照抄一个 dict 改数，`stocks.py` 加一行就行 —— 代码不用动。
"""
from __future__ import annotations

import copy

# 短名 -> 中文名 + 一句话人话说明（汇报时用它，别甩英文代号）
_LABEL = {
    'neutral': ('中性基准', '不风格化，原样放行（做 A/B 对照用）'),
    'portra400': ('柯达 Portra 400', '人像卷。肤色暖、高光柔、颗粒细、对比低 —— 最不挑人'),
    'pro400h': ('富士 Pro 400H', '通透偏青绿、低对比、高光更亮，日系那口气就是这个'),
    'fuji_c200': ('富士 C200', '消费负片。偏青绿、对比略高、颗粒看得出来，便宜卷的味道'),
    'ektar100': ('柯达 Ektar 100', '彩度和锐度都最猛的一卷，颗粒几乎看不见，适合风景'),
    'cinestill800t': ('电影卷 800T', '灯光片。日光下整体偏蓝，高光会散出红橙光晕（Halation）'),
    'air': ('日系空气感', '淡、亮、低彩，几乎无颗粒 —— 通透而不是浓'),
}


def _c(matrix, shadow, hilight, chroma, contrast, tint_lo=0.18, tint_hi=0.75):
    """颜色那组：色交叉矩阵(线性域 3x3) + 暗/亮分裂色调(显示域加性) + 彩度 + 对比。"""
    return dict(matrix=matrix, shadow_tint=shadow, hilight_tint=hilight,
                chroma=chroma, contrast=contrast, tint_lo=tint_lo, tint_hi=tint_hi)


def _s(grain=None, bloom=None, halation=None):
    """空间那组：三个效果各自的强度/尺度；没给的用 config 默认（默认是关）。"""
    d = {}
    if grain:
        d['grain'] = grain
    if bloom:
        d['bloom'] = bloom
    if halation:
        d['halation'] = halation
    return d


_ID = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

TABLE = {
    'neutral': dict(
        name='neutral', label=_LABEL['neutral'][0], desc=_LABEL['neutral'][1],
        color=_c(_ID, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 1.00, 1.00),
        spatial=_s(),
    ),

    'portra400': dict(
        name='portra400', label=_LABEL['portra400'][0], desc=_LABEL['portra400'][1],
        # 红绿微抬、蓝微收 = 暖肤色；负片交调很轻
        color=_c([[1.020, -0.014, -0.004],
                  [-0.010, 1.014, -0.004],
                  [-0.004, -0.010, 1.006]],
                 [0.000, 0.002, 0.010],      # 暗部微冷
                 [0.012, 0.004, -0.008],     # 亮部微暖
                 0.96, 0.98),
        spatial=_s(
            grain=dict(enable=True, amount=0.022, size=1.1, chroma=0.18,
                       skin_suppress=0.68, detail_suppress=0.30),
            bloom=dict(enable=True, amount=0.075, radius=22.0, thr_lo=0.74, thr_hi=0.93,
                       warmth=0.35, veil=0.045),
        ),
    ),

    'pro400h': dict(
        name='pro400h', label=_LABEL['pro400h'][0], desc=_LABEL['pro400h'][1],
        color=_c([[0.992, -0.006, 0.008],
                  [-0.006, 1.010, 0.000],
                  [0.000, 0.004, 0.998]],
                 [0.000, 0.008, 0.012],      # 暗部青绿
                 [0.002, 0.006, 0.006],      # 亮部偏青，通透
                 0.94, 0.96),
        spatial=_s(
            grain=dict(enable=True, amount=0.018, size=1.1, chroma=0.16,
                       skin_suppress=0.65, detail_suppress=0.30),
            bloom=dict(enable=True, amount=0.085, radius=24.0, thr_lo=0.72, thr_hi=0.92,
                       warmth=0.20, veil=0.040),
        ),
    ),

    'fuji_c200': dict(
        name='fuji_c200', label=_LABEL['fuji_c200'][0], desc=_LABEL['fuji_c200'][1],
        color=_c([[0.985, -0.010, 0.016],
                  [-0.008, 1.018, -0.004],
                  [-0.004, 0.008, 0.986]],
                 [0.000, 0.010, 0.014],
                 [0.004, 0.006, 0.002],
                 1.02, 1.05),
        spatial=_s(
            grain=dict(enable=True, amount=0.032, size=1.3, chroma=0.24,
                       skin_suppress=0.55, detail_suppress=0.25),
            bloom=dict(enable=True, amount=0.060, radius=20.0, thr_lo=0.76, thr_hi=0.94,
                       warmth=0.15, veil=0.030),
        ),
    ),

    'ektar100': dict(
        name='ektar100', label=_LABEL['ektar100'][0], desc=_LABEL['ektar100'][1],
        color=_c([[1.045, -0.020, -0.012],
                  [-0.018, 1.030, -0.006],
                  [-0.010, -0.016, 1.030]],
                 [0.004, 0.000, 0.008],
                 [0.010, 0.000, -0.004],
                 1.18, 1.12),
        spatial=_s(
            grain=dict(enable=True, amount=0.012, size=0.9, chroma=0.12,
                       skin_suppress=0.70, detail_suppress=0.35),
            bloom=dict(enable=True, amount=0.045, radius=18.0, thr_lo=0.80, thr_hi=0.96,
                       warmth=0.20, veil=0.020),
        ),
    ),

    'cinestill800t': dict(
        name='cinestill800t', label=_LABEL['cinestill800t'][0], desc=_LABEL['cinestill800t'][1],
        # 钨丝灯平衡：日光下蓝通道被压、红通道相对保留 → 整体偏蓝青
        color=_c([[0.965, 0.008, 0.022],
                  [-0.004, 1.002, 0.000],
                  [0.010, 0.004, 0.972]],
                 [0.000, 0.006, 0.018],      # 暗部蓝
                 [0.014, 0.004, -0.006],     # 亮部暖（红晕的底子）
                 1.05, 1.02),
        spatial=_s(
            grain=dict(enable=True, amount=0.038, size=1.5, chroma=0.28,
                       skin_suppress=0.50, detail_suppress=0.20),
            bloom=dict(enable=True, amount=0.090, radius=22.0, thr_lo=0.74, thr_hi=0.93,
                       warmth=0.30, veil=0.040),
            # 这一卷的招牌：高光往外散红橙晕圈
            halation=dict(enable=True, amount=0.130, radius=18.0,
                          thr_lo=0.78, thr_hi=0.99, color=[1.000, 0.300, 0.120]),
        ),
    ),

    'air': dict(
        name='air', label=_LABEL['air'][0], desc=_LABEL['air'][1],
        color=_c([[1.004, -0.002, -0.002],
                  [-0.002, 1.004, -0.002],
                  [-0.002, -0.002, 1.004]],
                 [0.000, 0.002, 0.006],
                 [0.004, 0.004, 0.004],
                 0.86, 0.95, tint_lo=0.25, tint_hi=0.85),
        spatial=_s(
            grain=dict(enable=True, amount=0.008, size=1.0, chroma=0.10,
                       skin_suppress=0.70, detail_suppress=0.40),
            bloom=dict(enable=True, amount=0.055, radius=26.0, thr_lo=0.78, thr_hi=0.94,
                       warmth=0.10, veil=0.025),
        ),
    ),
}

NAMES = list(TABLE.keys())


def names():
    return list(NAMES)


def get(name):
    """取一份卷（深拷贝，防止调用方改到表）。name=None/'' -> None。"""
    if not name:
        return None
    k = str(name).strip().lower()
    if k not in TABLE:
        raise KeyError('没有这个卷: %s（可选：%s）' % (name, ', '.join(NAMES)))
    return copy.deepcopy(TABLE[k])


def label_of(name):
    """中文名，汇报用。"""
    if not name:
        return 'config 默认'
    return TABLE[str(name).strip().lower()]['label']


def color_params(cfg, stock):
    """把卷的颜色参数叠到 config 默认上（卷优先）；stock=None 时就是 config 默认。"""
    p = dict(matrix=cfg.FILM_MATRIX, shadow_tint=cfg.FILM_SHADOW_TINT,
             hilight_tint=cfg.FILM_HILIGHT_TINT, tint_lo=cfg.TINT_LO,
             tint_hi=cfg.TINT_HI, chroma=cfg.CHROMA_SCALE, contrast=cfg.CONTRAST)
    if stock:
        c = stock.get('color') or {}
        for k in p:
            if k in c:
                p[k] = c[k]
    return p
