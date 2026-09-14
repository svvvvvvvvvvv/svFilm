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
  （09-10，SV 授权命名；见 `../_debug/master_doc_0912.md` §五"大师九条线"）。
* **颜色数值** = 由 `../_debug/calib_stocks_from_masters.py` 从 **1170 张大师成片**
  （`../_debug/analysis/master_resurvey.json`）量出来的，口径「取神不取形」：
  **卷 = 这条作者线相对"大师全体中位"的性格偏移**。
  逐卷证据见 `效果debug/<日期>/卷标定_大师颜色聚类/卷标定报告.md`。
* **空间数值** = 手写底子 × 数据相对微调（噪声混了 ISO/降噪/压缩，只能当相对信号）。
* **`pro400h` 没标定**（数据里没有"青绿粉彩"那条线）→ `calibrated=False`，仍是手写近似。
  ⚠ 我方中性路径与大师平均还有系统性差（偏暖 2.5 个 b*）—— 这条**没有**塞进卷里，
  改由 `config.BASE_TABLE` 的「基准成色」负责（`BASE_FULL` 已让探针四项全落大师带）。
  两者相加才是最终颜色：`cfg 默认 → 基准 → 卷`。
════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import copy

# 短名 -> 中文名 + 一句话人话说明（汇报时用它，别甩英文代号）
_LABEL = {
    'neutral': ('中性基准', '不风格化，原样放行（做 A/B 对照用）'),
    # ★★ 09-14 SV 定：**丢弃作者线，全部用真卷**（他的理由：「作者线们本来就是用真胶片拍的」
    #   ⇒ 那只是**从真胶片成片上量的二手**（还隔着小红书压缩图），而真卷是**一手**）。
    'portra400': ('柯达 Portra 400', '真卷：Kodak Portra 400 负片 + Portra Endura 相纸（spektrafilm 物理链）'),
    'pro400h': ('富士 Pro 400H', '真卷：Fujifilm Pro 400H + Crystal Archive Type II'),
    'fuji_c200': ('富士 C200', '真卷：Fujifilm C200 + Crystal Archive Type II'),
    'ektar100': ('柯达 Ektar 100', '真卷：Kodak Ektar 100 + Endura Premier'),
    'cinestill800t': ('电影卷 800T', '真卷：Kodak Vision3 500T + 2383 印片（Cinestill 800T 就是它去碳层）'),
}

_ID = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

# 把"量出来的数"翻成 L2 颜色模型能吃的字段
#   a / b       整体 a*/b* 偏移        ← 尺子：a*中位 / b*中位
#   b_sh / b_hi 暗部(L*≤P25)/亮部(L*≥P90) 额外 b*  ← 尺子：暗部b* / 亮部b*（冷暖分离）
#   chroma_p/s  彩度 gamma 与倍率      ← 尺子：彩度中位 + 彩度P90（形状 + 量级）
#   contrast    明度对比（只动 L*）    ← 尺子：反差 span90
def _c(a=0.0, b=0.0, b_sh=0.0, b_hi=0.0, chroma_p=1.0, chroma_s=1.0, chroma_ref=20.0,
       contrast=1.0, tint_lo=25.0, tint_hi=90.0, matrix=None, chroma_ends=None,
       tone_curve=None, tone_toe=None, tone_lift=None, tone_shoulder=None,
       film_color_w=None, density_stock=None):
    """`chroma_ends` = 抽色饱和（两端掉彩量）。None = 继承 config.CHROMA_ENDS；
    显式给 0.0 = 关（`neutral` 靠它保持恒等）。

    `tone_curve` / `tone_toe` / `tone_lift` / `tone_shoulder` = 胶片影调曲线（见 config.TONE_*）。
    None = 继承 config；显式给值 = 这一卷自己的影调性格（`neutral` 显式关掉以保恒等）。

    `film_color_w` = **真胶片成色三块**（丙串扰 / 甲分通道 / 乙密度引擎，见 `film.py`）的总强度。
    None = 继承 `config.FILM_COLOR_W`；**`neutral` 显式给 0.0** ⇒ 回到逐位恒等（A/B 对照底要干净）。
    `density_stock` = 乙 用哪条实测曲线（None = `config.DENSITY_STOCK`）。
    """
    return dict(matrix=(matrix or _ID), a=a, b=b, b_sh=b_sh, b_hi=b_hi,
                chroma_p=chroma_p, chroma_s=chroma_s, chroma_ref=chroma_ref,
                contrast=contrast, tint_lo=tint_lo, tint_hi=tint_hi,
                chroma_ends=chroma_ends,
                tone_curve=tone_curve, tone_toe=tone_toe, tone_lift=tone_lift,
                tone_shoulder=tone_shoulder,
                film_color_w=film_color_w, density_stock=density_stock)


def _s(grain=None, bloom=None, halation=None):
    """空间那组：三个效果各自的强度/尺度；没给的用 config 默认（默认是关）。

    ⚠ **bloom 不在这里给 `amount`**（09-13 SV 定档「化开 0.15」）：加性辉光与化开必须
    **成对相等**才能量守恒（`config.BLOOM_AMOUNT == config.BLOOM_SPREAD`），所以强度统一由
    `config.py` 给；卷只保留 `radius / thr_* / warmth / veil` 这些**性格**参数。
    """
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
        color=_c(chroma_ends=0.0, tone_curve=False, film_color_w=0.0),
        # ↑ 恒等：抽色饱和 + 影调曲线 + **真胶片成色三块** 全关掉，保 A/B 对照底干净
        spatial=_s(),
        source=None, calibrated=True,   # 恒等 = 无需标定
    ),


    # ══ 五个真卷（09-14 SV：「丢弃作者线，全部用真卷」）══════════════════════════
    # 作者线是**从真胶片成片上量的二手**（还隔着小红书压缩图）；真卷是**数据表 + 光谱测量的一手**。
    # 组装：**我们只出「入口 + 锚点 + 降噪 + 肤色 + 护栏」，胶片性格整段交给真卷**。
    'portra400': dict(
        name='portra400', label=_LABEL['portra400'][0], desc=_LABEL['portra400'][1],
        # ★★ 真卷：走 spektrafilm 的物理链（负片 → 印相 → 扫描）。
        #   颜色 / 影调 / 颗粒 / halation **全部由它自带** ⇒ 我们的 color/spatial 保持恒等、
        #   且 pipeline 会**跳过** L1 影调 + L2 颜色 + 空间层。详见 `spektra.py`。
        spek=dict(film='kodak_portra_400', print='kodak_portra_endura',
                  pe=0.58),   # 落点标定：让五卷都落在中位 ~55
        color=_c(), spatial=_s(),
        source='spektrafilm', calibrated='物理',   # 负片+相纸官配
    ),
    'fuji_c200': dict(
        name='fuji_c200', label=_LABEL['fuji_c200'][0], desc=_LABEL['fuji_c200'][1],
        # ★★ 真卷：走 spektrafilm 的物理链（负片 → 印相 → 扫描）。
        #   颜色 / 影调 / 颗粒 / halation **全部由它自带** ⇒ 我们的 color/spatial 保持恒等、
        #   且 pipeline 会**跳过** L1 影调 + L2 颜色 + 空间层。详见 `spektra.py`。
        spek=dict(film='fujifilm_c200', print='fujifilm_crystal_archive_typeii',
                  pe=1.41),   # 落点标定：让五卷都落在中位 ~55
        color=_c(), spatial=_s(),
        source='spektrafilm', calibrated='物理',   # 
    ),
    'pro400h': dict(
        name='pro400h', label=_LABEL['pro400h'][0], desc=_LABEL['pro400h'][1],
        # ★★ 真卷：走 spektrafilm 的物理链（负片 → 印相 → 扫描）。
        #   颜色 / 影调 / 颗粒 / halation **全部由它自带** ⇒ 我们的 color/spatial 保持恒等、
        #   且 pipeline 会**跳过** L1 影调 + L2 颜色 + 空间层。详见 `spektra.py`。
        spek=dict(film='fujifilm_pro_400h', print='fujifilm_crystal_archive_typeii',
                  pe=0.76),   # 落点标定：让五卷都落在中位 ~55
        color=_c(), spatial=_s(),
        source='spektrafilm', calibrated='物理',   # 
    ),
    'ektar100': dict(
        name='ektar100', label=_LABEL['ektar100'][0], desc=_LABEL['ektar100'][1],
        # ★★ 真卷：走 spektrafilm 的物理链（负片 → 印相 → 扫描）。
        #   颜色 / 影调 / 颗粒 / halation **全部由它自带** ⇒ 我们的 color/spatial 保持恒等、
        #   且 pipeline 会**跳过** L1 影调 + L2 颜色 + 空间层。详见 `spektra.py`。
        spek=dict(film='kodak_ektar_100', print='kodak_endura_premier',
                  pe=0.81),   # 落点标定：让五卷都落在中位 ~55
        color=_c(), spatial=_s(),
        source='spektrafilm', calibrated='物理',   # 
    ),
    'cinestill800t': dict(
        name='cinestill800t', label=_LABEL['cinestill800t'][0], desc=_LABEL['cinestill800t'][1],
        # ★★ 真卷：走 spektrafilm 的物理链（负片 → 印相 → 扫描）。
        #   颜色 / 影调 / 颗粒 / halation **全部由它自带** ⇒ 我们的 color/spatial 保持恒等、
        #   且 pipeline 会**跳过** L1 影调 + L2 颜色 + 空间层。详见 `spektra.py`。
        spek=dict(film='kodak_vision3_500t', print='kodak_2383',
                  pe=1.18),   # 落点标定：让五卷都落在中位 ~55
        color=_c(), spatial=_s(),
        source='spektrafilm', calibrated='物理',   # Cinestill 800T = Vision3 500T 去碳层
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


# ---------------- 基准成色（不属于任何卷） ----------------
def base_names():
    return list(C_BASE_NAMES)


def resolve_base(cfg, base=None):
    """基准成色的解析 + 归一成 L2 能吃的字段。

    base 可以是预设名（见 config.BASE_TABLE）或 dict；None 时取 config.BASE。
    键用 b_ 前缀，避免和卷的字段名混淆。
    """
    name = base if base is not None else getattr(cfg, 'BASE', None)
    tb = getattr(cfg, 'BASE_TABLE', {}) or {}
    if isinstance(name, dict):
        d = name
        name = d.get('name') or 'custom'
    else:
        key = str(name or 'BASE_NONE').strip().upper()
        if key not in tb:
            key = 'BASE_NONE'
        d = tb.get(key, {})
        name = key
    return dict(name=name,
                label=d.get('label') or name,
                desc=d.get('desc') or '',
                b_a=float(d.get('a', 0.0) or 0.0),
                b_b=float(d.get('b', 0.0) or 0.0),
                b_b_sh=float(d.get('b_sh', 0.0) or 0.0),
                b_b_hi=float(d.get('b_hi', 0.0) or 0.0),
                b_chroma_p=float(d.get('chroma_p', 1.0) or 1.0),
                b_chroma_s=float(d.get('chroma_s', 1.0) or 1.0),
                b_contrast=float(d.get('contrast', 1.0) or 1.0),
                b_fog=float(d.get('fog', 0.0) or 0.0))


def base_label(cfg, base=None):
    """基准成色的中文名，汇报用。None/未指定 → 取 config.BASE；都是恒等 → 汇报成"无"。"""
    r = resolve_base(cfg, base)
    if abs(r['b_b']) + abs(r['b_a']) + abs(r['b_fog']) < 1e-9 and \
            abs(r['b_chroma_p'] - 1.0) < 1e-9 and abs(r['b_chroma_s'] - 1.0) < 1e-9 and \
            abs(r['b_contrast'] - 1.0) < 1e-9:
        return '无'
    return r['label']


C_BASE_NAMES = ['BASE_NONE', 'BASE_FOG', 'BASE_DEYELLOW', 'BASE_FULL']


def color_params(cfg, stock, base=None):
    """把「基准成色」再叠「卷」的颜色参数，得出 L2 最终要用的那一组数。

    顺序（路 B 定的口径，别混）：
      cfg 默认  →  基准成色（中性路径对齐大师平均）  →  卷（相对大师平均的性格偏移）

    叠加规则：
      * 偏移类（a / b / b_sh / b_hi）：**相加**。卷量的是"相对大师平均的偏移"，
        基准负责把我们的中性路径挪到"大师平均"上，两者相加才落在作者线上。
      * 彩度类（chroma_p / chroma_s）与对比（contrast）：**相乘**。
      * fog（雾）只有基准有，卷不用管。

    字段 ↔ 尺子对应：
      a / b       整体 a*/b* 偏移      ←→ 尺子的 a*中位 / b*中位
      b_sh / b_hi 暗部/亮部额外 b*     ←→ 尺子的 暗部b* / 亮部b*（冷暖分离）
      chroma_p/s  彩度 gamma 与倍率    ←→ 尺子的 彩度中位 / 彩度P90
      contrast    明度对比             ←→ 尺子的 反差 span90
      fog         线性光域黑位抬升      ←→ 尺子的 黑位 / 雾量
    """
    b = resolve_base(cfg, base)
    p = dict(matrix=cfg.FILM_MATRIX,
             a=cfg.COL_A + b['b_a'], b=cfg.COL_B + b['b_b'],
             b_sh=cfg.COL_B_SH + b['b_b_sh'], b_hi=cfg.COL_B_HI + b['b_b_hi'],
             tint_lo=cfg.COL_TINT_LO, tint_hi=cfg.COL_TINT_HI,
             chroma_p=cfg.CHROMA_P * b['b_chroma_p'],
             chroma_s=cfg.CHROMA_S * b['b_chroma_s'],
             chroma_ref=cfg.CHROMA_REF,
             contrast=cfg.CONTRAST * b['b_contrast'],
             chroma_ends=float(getattr(cfg, 'CHROMA_ENDS', 0.0) or 0.0),
             # ★ 胶片影调曲线（L2）：影子在风格层，所以也在 cfg → 基准 → 卷 这条链上。
             #   基准不动它（基准只管"把我方中性路径对齐大师平均"），卷可以覆盖（各卷自带影调性格）。
             tone_curve=bool(getattr(cfg, 'TONE_CURVE', False)),
             tone_toe=float(getattr(cfg, 'TONE_TOE', 0.0) or 0.0),
             tone_lift=float(getattr(cfg, 'TONE_LIFT', 0.0) or 0.0),
             tone_shoulder=float(getattr(cfg, 'TONE_SHOULDER', 0.0) or 0.0),
             fog=b['b_fog'],
             # ★ 真胶片成色三块（丙/甲/乙）的总强度 + 乙用哪条实测曲线。
             #   None 的卷继承 config ⇒ 只有 `neutral` 显式给 0.0（恒等）。
             film_color_w=float(getattr(cfg, 'FILM_COLOR_W', 1.0)),
             density_stock=getattr(cfg, 'DENSITY_STOCK', 'portra400'))
    if stock:
        c = stock.get('color') or {}
        for k in ('matrix', 'tint_lo', 'tint_hi', 'chroma_ref'):
            if k in c:
                p[k] = c[k]
        if c.get('chroma_ends') is not None:            # 抽色饱和：卷可显式覆盖（0 = 关）
            p['chroma_ends'] = float(c['chroma_ends'])
        # 影调曲线：卷可整组覆盖（给 None 就是"这一卷不要影调曲线"）
        if c.get('tone_curve') is not None:
            p['tone_curve'] = bool(c['tone_curve'])
        for k in ('tone_toe', 'tone_lift', 'tone_shoulder'):
            if c.get(k) is not None:
                p[k] = float(c[k])
        for k in ('a', 'b', 'b_sh', 'b_hi'):
            if k in c:
                p[k] = p[k] + c[k]                     # 偏移相加
        for k in ('chroma_p', 'chroma_s', 'contrast'):
            if k in c:
                p[k] = p[k] * c[k]                     # 乘性相加
        # ★ 真胶片成色三块：卷可显式给（`neutral` = 0.0 ⇒ 恒等）
        if c.get('film_color_w') is not None:
            p['film_color_w'] = float(c['film_color_w'])
        if c.get('density_stock'):
            p['density_stock'] = str(c['density_stock'])
    return p
