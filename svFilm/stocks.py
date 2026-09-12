# -*- coding: utf-8 -*-
r"""胶片卷表 —— 一个"卷"就是一份数据：颜色性格 + 空间效果强度。

为什么要做成表：换卷不该改代码，只该改数据。所以这里全是纯数据，
`style.py` 拿它去调颜色，`spatial.py` 拿它去调颗粒/黑柔/Halation。

一个卷包含两组东西，分得很清楚：
  * 颜色（归 L2）：a*/b* 偏移、暗部/亮部 b* 分离、彩度(p, s)、明度对比
  * 空间（归 spatial，LUT 装不下）：颗粒 / 黑柔 Bloom / Halation

`neutral` 是"什么都不做"的基准（恒等），用来做 A/B 的对照栏。
不选卷（stock=None）时用 config.py 里的默认，行为和以前一致。

想要新卷：照抄一个 dict 改数，`stocks.py` 加一行就行 —— 代码不用动。

════════════════════════════════════════════════════════════════
**出处（v0.2.1 起：颜色数值是从数据量出来的，不是手编的）**

* **卷名 / 方向** 继承 `E:\WorkBuddy\摄影助手\大师作品\胶片卷映射与分组策略.md`
  （09-10，SV 授权命名；见 `_debug/master_doc_0912.md` §五"大师九条线"）。
* **颜色数值** = 由 `_debug/calib_stocks_from_masters.py` 从 **1170 张大师成片**
  （`_debug/analysis/master_resurvey.json`）量出来的，口径「取神不取形」：
  **卷 = 这条作者线相对"大师全体中位"的性格偏移**。
  逐卷证据见 `效果debug/<日期>/卷标定_大师颜色聚类/卷标定报告.md`。
* **空间数值** = 手写底子 × 数据相对微调（噪声混了 ISO/降噪/压缩，只能当相对信号）。
* **`pro400h` 没标定**（数据里没有"青绿粉彩"那条线）→ `calibrated=False`，仍是手写近似。
  ⚠ 我方中性路径与大师平均还有系统性差（偏暖 2.5 个 b*）—— 这条**没有**塞进卷里，
  单独列在标定报告 §四等拍板。
════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import copy

# 短名 -> 中文名 + 一句话人话说明（汇报时用它，别甩英文代号）
_LABEL = {
    'neutral': ('中性基准', '不风格化，原样放行（做 A/B 对照用）'),
    'portra400': ('柯达 Portra 400', '"鹿井"那条线：整体微绿、暗部回暖、亮部收一点彩 —— 最不挑人'),
    'pro400h': ('富士 Pro 400H', '青绿通透、低对比（⚠ 未标定，手写近似；数据里没有这条线）'),
    'fuji_c200': ('富士 C200', '"石田真澄"那条线：低饱和、略平、暗部回暖 —— 日常淡调'),
    'ektar100': ('柯达 Ektar 100', '"川岛小鸟(仿拍)"那条线：整体最暖最浓、暗部平、亮部反压冷'),
    'cinestill800t': ('电影卷 800T', '"MasashiWakui"那条线：亮部爆暖 + 强雾 + 红橙晕圈(Halation)'),
    'air': ('日系空气感', '"酒井貴弘"那条线：亮调为主、暗部微暖、彩度略高于基准'),
}

_ID = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

# 把"量出来的数"翻成 L2 颜色模型能吃的字段
#   a / b       整体 a*/b* 偏移        ← 尺子：a*中位 / b*中位
#   b_sh / b_hi 暗部(L*≤P25)/亮部(L*≥P90) 额外 b*  ← 尺子：暗部b* / 亮部b*（冷暖分离）
#   chroma_p/s  彩度 gamma 与倍率      ← 尺子：彩度中位 + 彩度P90（形状 + 量级）
#   contrast    明度对比（只动 L*）    ← 尺子：反差 span90
def _c(a=0.0, b=0.0, b_sh=0.0, b_hi=0.0, chroma_p=1.0, chroma_s=1.0, chroma_ref=20.0,
       contrast=1.0, tint_lo=25.0, tint_hi=90.0, matrix=None):
    return dict(matrix=(matrix or _ID), a=a, b=b, b_sh=b_sh, b_hi=b_hi,
                chroma_p=chroma_p, chroma_s=chroma_s, chroma_ref=chroma_ref,
                contrast=contrast, tint_lo=tint_lo, tint_hi=tint_hi)


def _s(grain=None, bloom=None, halation=None):
    """空间那组：三个效果各自的强度/尺度；没给的用 config 默认（默认是关）。"""
    d = {}
    for k, v in (('grain', grain), ('bloom', bloom), ('halation', halation)):
        if v:
            vv = dict(v)
            vv.setdefault('enable', True)
            d[k] = vv
    return d


TABLE = {
    'neutral': dict(
        name='neutral', label=_LABEL['neutral'][0], desc=_LABEL['neutral'][1],
        color=_c(), spatial=_s(),
        source=None, calibrated=True,   # 恒等 = 无需标定
    ),

    # ── 以下五卷：颜色由 calib_stocks_from_masters.py 从大师线量出（09-12） ──
    'portra400': dict(
        name='portra400', label=_LABEL['portra400'][0], desc=_LABEL['portra400'][1],
        color=_c(a=-0.48, b=-0.74, b_sh=+1.33, b_hi=+0.22,
                 chroma_p=1.041, chroma_s=0.999, contrast=1.055),
        spatial=_s(
            grain=dict(amount=0.0297, size=1.1, chroma=0.18,
                       skin_suppress=0.68, detail_suppress=0.30),
            bloom=dict(amount=0.0752, radius=22.0, thr_lo=0.74, thr_hi=0.93,
                       warmth=0.35, veil=0.0248),
        ),
        source='鹿井', calibrated=True,
    ),

    # ⚠ 唯一没标定的一卷：数据里没有"青绿粉彩"那条作者线 → 手写近似（路 A 色卡标定才能名副其实）
    'pro400h': dict(
        name='pro400h', label=_LABEL['pro400h'][0], desc=_LABEL['pro400h'][1],
        color=_c(a=-0.50, b=-1.60, b_sh=-1.20, b_hi=-0.40,
                 chroma_p=1.020, chroma_s=0.900, contrast=0.970),
        spatial=_s(
            grain=dict(amount=0.018, size=1.1, chroma=0.16,
                       skin_suppress=0.65, detail_suppress=0.30),
            bloom=dict(amount=0.085, radius=24.0, thr_lo=0.72, thr_hi=0.92,
                       warmth=0.20, veil=0.040),
        ),
        source=None, calibrated=False,
    ),

    'fuji_c200': dict(
        name='fuji_c200', label=_LABEL['fuji_c200'][0], desc=_LABEL['fuji_c200'][1],
        color=_c(a=-0.44, b=-0.73, b_sh=+1.36, b_hi=+0.73,
                 chroma_p=0.963, chroma_s=0.822, contrast=1.034),
        spatial=_s(
            grain=dict(amount=0.0355, size=1.3, chroma=0.24,
                       skin_suppress=0.55, detail_suppress=0.25),
            bloom=dict(amount=0.0610, radius=20.0, thr_lo=0.76, thr_hi=0.94,
                       warmth=0.15, veil=0.0199),
        ),
        source='石田真澄', calibrated=True,
    ),

    'ektar100': dict(
        name='ektar100', label=_LABEL['ektar100'][0], desc=_LABEL['ektar100'][1],
        color=_c(a=+0.00, b=+2.81, b_sh=+0.46, b_hi=-2.81,
                 chroma_p=1.107, chroma_s=1.202, contrast=1.040),
        spatial=_s(
            grain=dict(amount=0.0123, size=0.9, chroma=0.12,
                       skin_suppress=0.70, detail_suppress=0.35),
            bloom=dict(amount=0.0503, radius=18.0, thr_lo=0.80, thr_hi=0.96,
                       warmth=0.20, veil=0.0149),
        ),
        source='川岛小鸟(仿拍)', calibrated=True,
    ),

    'cinestill800t': dict(
        name='cinestill800t', label=_LABEL['cinestill800t'][0], desc=_LABEL['cinestill800t'][1],
        color=_c(a=+0.00, b=-0.75, b_sh=-0.00, b_hi=+2.90,
                 chroma_p=0.944, chroma_s=0.878, contrast=0.940),
        spatial=_s(
            grain=dict(amount=0.0476, size=1.5, chroma=0.28,
                       skin_suppress=0.50, detail_suppress=0.20),
            bloom=dict(amount=0.0777, radius=22.0, thr_lo=0.74, thr_hi=0.93,
                       warmth=0.30, veil=0.1200),
            # 这一卷的招牌：高光往外散红橙晕圈（作者线雾量 0.32 = 全场最高，坐实）
            halation=dict(amount=0.130, radius=18.0,
                          thr_lo=0.78, thr_hi=0.99, color=[1.000, 0.300, 0.120]),
        ),
        source='MasashiWakui', calibrated=True,
    ),

    'air': dict(
        name='air', label=_LABEL['air'][0], desc=_LABEL['air'][1],
        color=_c(a=+0.00, b=-0.75, b_sh=+0.84, b_hi=+0.75,
                 chroma_p=1.030, chroma_s=1.078, contrast=1.049),
        spatial=_s(
            grain=dict(amount=0.0062, size=1.0, chroma=0.10,
                       skin_suppress=0.70, detail_suppress=0.40),
            bloom=dict(amount=0.0626, radius=26.0, thr_lo=0.78, thr_hi=0.94,
                       warmth=0.10, veil=0.0183),
        ),
        source='酒井貴弘', calibrated=True,
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
    """把卷的颜色参数叠到 config 默认上（卷优先）；stock=None 时就是 config 默认。

    字段和"大师那把尺子"一一对应：
      a / b       整体 a*/b* 偏移      ←→ 尺子的 a*中位 / b*中位
      b_sh / b_hi 暗部/亮部额外 b*     ←→ 尺子的 暗部b* / 亮部b*（冷暖分离）
      chroma_p/s  彩度 gamma 与倍率    ←→ 尺子的 彩度中位 / 彩度P90
      contrast    明度对比             ←→ 尺子的 反差 span90
    """
    p = dict(matrix=cfg.FILM_MATRIX,
             a=cfg.COL_A, b=cfg.COL_B, b_sh=cfg.COL_B_SH, b_hi=cfg.COL_B_HI,
             tint_lo=cfg.COL_TINT_LO, tint_hi=cfg.COL_TINT_HI,
             chroma_p=cfg.CHROMA_P, chroma_s=cfg.CHROMA_S, chroma_ref=cfg.CHROMA_REF,
             contrast=cfg.CONTRAST)
    if stock:
        c = stock.get('color') or {}
        for k in p:
            if k in c:
                p[k] = c[k]
    return p
