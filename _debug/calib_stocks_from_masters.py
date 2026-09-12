# -*- coding: utf-8 -*-
"""从大师作品把"卷"解出来 —— 路 B 的核心脚本。

不手填颜色数字：所有卷的颜色参数都由两个数据集解出
  A) `_debug/analysis/master_resurvey.json` —— 1172 张大师成片的颜色指纹（同一把尺子）
  B) 我方基线指纹（`_debug/lab_ours_fingerprint.py` 量的，14 张探针、neutral 卷）

映射关系（09-10《胶片卷映射与分组策略》已定案，SV 授权命名）：
  卷名          ← 大师作者线         取神点
  portra400     ← 鹿井               原始调色靶（暖底人像）
  fuji_c200     ← 石田真澄           muted 日常低饱和
  ektar100      ← 川岛小鸟(仿拍)      浓彩撞色
  cinestill800t ← MasashiWakui       夜景橙青 split-tone
  air           ← 酒井貴弘           高调清透（Portra 过曝）
  pro400h       ← （无）             数据里没有"青绿粉彩"这条线 → 保持未标定，明确标出来

## 口径：卷 = 这条线相对"大师全体平均"的性格偏移（「取神不取形」）

为什么不用"绝对落点"（= 直接拿我方基线减这条线）：
我方基线和大师平均之间**还有系统性差**（我方偏暖 2.5 个 b*、彩度形状偏陡 0.6）。
那是**中性路径**的课题，不该由每个卷各修一遍 —— 否则以后改中性路径，所有卷全废。
所以卷只带"这条线自己的性格"，绝对差单独列出来给 SV 拍板（见报告 §四）。

解法（都是闭式，不用迭代）：
  色偏   a = Δa*；b = Δb*；b_sh = Δ暗部b* − b；b_hi = Δ亮部b* − b    （Δ = 这条线 − 大师平均）
  彩度   C' = s·Cref·(C/Cref)^p 是**逐像素单调**函数 ⇒ 分位跟着走；
         两个方程（彩度中位、彩度P90）解 (p, s)，精确
  对比   contrast 由反差 span90 的比值给（小幅度，限幅 0.94~1.10）
  空间   在手写底子上做**相对微调**（噪声混了 ISO/降噪/压缩，只能当相对信号）

用法：
  PY _debug/calib_stocks_from_masters.py --ours "D:/照片库/svFilm效果/效果debug/2026-09-12/卷标定_我方指纹/基线14.json"
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from svFilm import paths, stocks  # noqa: E402

MASTER_JSON = 'E:/WorkBuddy/摄影助手/_debug/analysis/master_resurvey.json'

# 卷 ← 作者线（09-10 定案）
MAP = {
    'portra400': '鹿井',
    'fuji_c200': '石田真澄',
    'ektar100': '川岛小鸟(仿拍)',
    'cinestill800t': 'MasashiWakui',
    'air': '酒井貴弘',
    'pro400h': None,
}

CN = {'portra400': '柯达 Portra 400', 'pro400h': '富士 Pro 400H', 'fuji_c200': '富士 C200',
      'ektar100': '柯达 Ektar 100', 'cinestill800t': '电影卷 800T', 'air': '日系空气感',
      'neutral': '中性基准'}

KEYS = ['L50', 'span90', 'black', 'c50', 'c90', 'gray_pct', 'a_med', 'b_med',
        'b_sh', 'b_hi', 'fade_lin', 'noise', 'hi90']

CREF = 20.0


def load_masters():
    d = json.load(open(MASTER_JSON, encoding='utf-8'))
    rows = [r for r in d['rows'] if r.get('c50') and r.get('a_med') is not None]
    byg = {}
    for r in rows:
        byg.setdefault(r['group'], []).append(r)
    return rows, byg


def med(rows, k):
    v = [float(r[k]) for r in rows if r.get(k) is not None and np.isfinite(float(r[k]))]
    return float(np.median(v)) if v else float('nan')


def fingerprint(rows):
    f = {k: med(rows, k) for k in KEYS}
    f['c_ratio'] = f['c90'] / max(f['c50'], 1e-6)
    f['split'] = f['b_hi'] - f['b_sh']
    f['n'] = len(rows)
    return f


def solve_chroma(c50o, c90o, c50t, c90t, cref=CREF, p_lim=(0.55, 1.8), s_lim=(0.60, 1.8)):
    """C' = s·cref·(C/cref)^p 单调 ⇒ 分位跟着走，两个方程解 (p, s)，精确。"""
    if min(c50o, c90o, c50t, c90t) <= 1e-6:
        return 1.0, 1.0, c50o, c90o
    if abs(math.log(c90o / c50o)) < 1e-6:
        return 1.0, 1.0, c50o, c90o
    p = math.log(c90t / c50t) / math.log(c90o / c50o)
    p_c = float(np.clip(p, *p_lim))
    s = c50t / (cref * (c50o / cref) ** p_c)
    s_c = float(np.clip(s, *s_lim))
    get50 = s_c * cref * (c50o / cref) ** p_c
    get90 = s_c * cref * (c90o / cref) ** p_c
    return p_c, s_c, get50, get90


def fmt(v, n=2):
    return ('%+.' + str(n) + 'f') % v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ours', required=True, help='我方基线指纹 json（lab_ours_fingerprint.py 产出）')
    ap.add_argument('--purpose', default='卷标定_大师颜色聚类')
    ap.add_argument('--root', default='D:/照片库/svFilm效果')
    ap.add_argument('--date', default=None)
    ap.add_argument('--cref', type=float, default=CREF)
    ap.add_argument('--baseline-stock', default='neutral')
    a = ap.parse_args()

    rows, byg = load_masters()
    allf = fingerprint(rows)
    ours = json.load(open(a.ours, encoding='utf-8'))
    our = ours['summary'][a.baseline_stock]
    our = dict(our)
    for k in KEYS:                     # 老版本基线 json 可能缺字段（hi90 等）
        our.setdefault(k, float('nan'))

    # 手写底子（空间只做相对微调，不改"有无"；色调边界沿用旧值）
    hand, hand_col = {}, {}
    for nm in MAP:
        st = stocks.TABLE.get(nm) or {}
        hand[nm] = st.get('spatial') or {}
        hand_col[nm] = st.get('color') or {}

    print('大师 %d 张 / %d 组；我方基线 %s' % (len(rows), len(byg), a.baseline_stock))
    print('\n[锚点] 大师全体中位: a*%s b*%s 暗部b*%s 亮部b*%s | 彩度50 %.2f 彩度90 %.2f 形状%.2f 中性%.0f%% 反差%.0f 雾%.4f'
          % (fmt(allf['a_med']), fmt(allf['b_med']), fmt(allf['b_sh']), fmt(allf['b_hi']),
             allf['c50'], allf['c90'], allf['c_ratio'], allf['gray_pct'] * 100,
             allf['span90'], allf['fade_lin']))
    print('[锚点] 我方基线    : a*%s b*%s 暗部b*%s 亮部b*%s | 彩度50 %.2f 彩度90 %.2f 形状%.2f 中性%.0f%% 反差%.0f 雾%.4f'
          % (fmt(our['a_med']), fmt(our['b_med']), fmt(our['b_sh']), fmt(our['b_hi']),
             our['c50'], our['c90'], our['c_ratio'], our['gray_pct'] * 100,
             our['span90'], our['fade_lin']))

    out = {'anchors': {'masters': allf, 'ours': our}, 'stocks': {}}
    lines = []
    lines.append('# 卷标定报告（路 B：从大师作品量出来）\n')
    lines.append('> 数据：`master_resurvey.json`（1170 张大师成片）+ 我方基线（14 张探针、neutral 卷）。')
    lines.append('> 尺子：`svFilm/metrics.py`（与大师同一把，已用 24 张原图交叉校验，颜色项中位差 ≤0.03）。')
    lines.append('> 映射：09-10《胶片卷映射与分组策略》已定案（SV 授权命名）。')
    lines.append('> **口径**：卷 = 这条线**相对大师全体平均**的性格偏移（取神不取形）。')
    lines.append('> 我方基线与大师平均之间的系统性差单独列在 §四，不塞进卷里。\n')
    lines.append('## 一、锚点\n')
    lines.append('| | a*中位 | b*中位 | 暗部b* | 亮部b* | 彩度中位 | 彩度P90 | 彩度形状 | 近中性占比 | 反差 | 黑位 | 颗粒 | 雾量 | 高调占比 |')
    lines.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for nm, f in (('大师全体中位（= 基准成色）', allf), ('我方基线（未套卷）', our)):
        lines.append('| %s | %s | %s | %s | %s | %.2f | %.2f | %.2f | %.0f%% | %.0f | %.1f | %.2f | %.4f | %.1f%% |'
                     % (nm, fmt(f['a_med']), fmt(f['b_med']), fmt(f['b_sh']), fmt(f['b_hi']),
                        f['c50'], f['c90'], f['c_ratio'], f['gray_pct'] * 100,
                        f['span90'], f['black'], f['noise'], f['fade_lin'], f['hi90'] * 100))
    lines.append('\n## 二、一卷一卷量出来\n')
    lines.append('| 卷 ← 作者线 | 整体a* | 整体b* | 暗部b* | 亮部b* | 彩度p | 彩度s | 对比 | 颗粒 | 黑柔 | 雾量 |')
    lines.append('|---|---|---|---|---|---|---|---|---|---|---|')
    detail = []
    print('\n' + '=' * 112)
    for name, author in MAP.items():
        if author is None:
            out['stocks'][name] = dict(source=None, calibrated=False, hand_written=True)
            print('%-14s ← （数据里没有对应作者线）→ 保持手写，calibrated=False' % name)
            continue
        rs = byg[author]
        t = fingerprint(rs)

        # ① 色度：线相对大师平均
        d_a = t['a_med'] - allf['a_med']
        b = t['b_med'] - allf['b_med']
        b_sh = (t['b_sh'] - allf['b_sh']) - b
        b_hi = (t['b_hi'] - allf['b_hi']) - b

        # ② 彩度：把我方彩度按"线的相对形状与量级"缩放
        t50 = our['c50'] * (t['c50'] / max(allf['c50'], 1e-6))
        t90 = our['c90'] * (t['c90'] / max(allf['c90'], 1e-6))
        p_c, s_c, got50, got90 = solve_chroma(our['c50'], our['c90'], t50, t90, a.cref)

        # ③ 对比
        contrast = float(np.clip(1.0 + (t['span90'] / max(allf['span90'], 1e-6) - 1.0) * 0.8, 0.94, 1.10))

        # ④ 空间：手写底子上做相对微调
        hg = hand[name].get('grain') or {}
        hb = hand[name].get('bloom') or {}
        hh = hand[name].get('halation') or {}
        g_k = float(np.clip(t['noise'] / max(allf['noise'], 1e-6), 0.70, 1.35))
        b_k = float(np.clip(1.0 + (t['hi90'] - allf['hi90']) * 1.5, 0.75, 1.40))
        v_k = float(np.clip(t['fade_lin'] / max(allf['fade_lin'], 1e-6), 0.55, 3.0))
        g_amt = round(float(hg.get('amount', 0.022)) * g_k, 4)
        b_amt = round(float(hb.get('amount', 0.075)) * b_k, 4)
        veil = round(float(hb.get('veil', 0.045)) * v_k, 4)
        halo = float(hh.get('amount', 0.0))

        out['stocks'][name] = dict(
            source=author, calibrated=True,
            color=dict(a=round(d_a, 2), b=round(b, 2), b_sh=round(b_sh, 2), b_hi=round(b_hi, 2),
                       chroma_p=round(p_c, 3), chroma_s=round(s_c, 3), chroma_ref=a.cref,
                       contrast=round(contrast, 3),
                       tint_lo=25.0, tint_hi=90.0),   # 对齐尺子：暗部 L*≤P25 / 亮部 L*≥P90
            spatial=dict(
                grain=dict(amount=g_amt, size=hg.get('size', 1.2), chroma=hg.get('chroma', 0.20),
                           skin_suppress=hg.get('skin_suppress', 0.62),
                           detail_suppress=hg.get('detail_suppress', 0.32)),
                bloom=dict(amount=b_amt, veil=veil, radius=hb.get('radius', 22.0),
                           thr_lo=hb.get('thr_lo', 0.74), thr_hi=hb.get('thr_hi', 0.93),
                           warmth=hb.get('warmth', 0.30)),
                halation=dict(amount=round(halo, 4), radius=hh.get('radius', 18.0),
                              thr_lo=hh.get('thr_lo', 0.78), thr_hi=hh.get('thr_hi', 0.99),
                              color=list(hh.get('color', [1.0, 0.30, 0.12]))),
            ),
            evidence=dict(
                target=dict(c50=t['c50'], c90=t['c90'], c_ratio=t['c_ratio'], gray_pct=t['gray_pct'],
                            span90=t['span90'], black=t['black'], a_med=t['a_med'], b_med=t['b_med'],
                            b_sh=t['b_sh'], b_hi=t['b_hi'], noise=t['noise'], hi90=t['hi90']),
                relative=dict(c50=t['c50'] / max(allf['c50'], 1e-6), c90=t['c90'] / max(allf['c90'], 1e-6),
                              span90=t['span90'] / max(allf['span90'], 1e-6),
                              noise=g_k, hi90=b_k, fade=v_k),
                n=int(t['n'])),
        )
        print('%-14s ← %-14s Δa%+5.2f Δb%+5.2f (暗%+5.2f 亮%+5.2f) | 彩度 p=%.3f s=%.3f | 对比%.3f'
              % (name, author, d_a, b, b_sh, b_hi, p_c, s_c, contrast))
        print('     %-14s    彩度50 %.1f→%.1f (线%.1f)   彩度90 %.1f→%.1f (线%.1f)   中性 %.0f%%→线%.0f%%'
              % ('', our['c50'], got50, t['c50'], our['c90'], got90, t['c90'],
                 our['gray_pct'] * 100, t['gray_pct'] * 100))
        print('     %-14s    空间：颗粒 %.4f(x%.2f)  黑柔 %.4f(x%.2f)  雾 %.4f(x%.2f)%s'
              % ('', g_amt, g_k, b_amt, b_k, veil, v_k,
                 '  Halation %.3f（招牌）' % halo if halo > 0 else ''))
        lines.append('| %s ← %s | %s | %s | %s | %s | %.3f | %.3f | %.3f | %.4f | %.4f | %.4f |'
                     % (CN.get(name, name), author, fmt(d_a), fmt(b), fmt(b_sh), fmt(b_hi),
                        p_c, s_c, contrast, g_amt, b_amt, veil))
        detail.append((name, author, t, our, dict(a=d_a, b=b, b_sh=b_sh, b_hi=b_hi, chroma_p=p_c,
                                                  chroma_s=s_c, contrast=contrast, g_amt=g_amt,
                                                  b_amt=b_amt, veil=veil, halo=halo, n=len(rs))))

    lines.append('\n### 逐卷明细（对方量了什么，卷就拿什么）\n')
    for name, author, t, o, q in detail:
        lines.append('#### %s ← %s（n=%d）\n' % (CN.get(name, name), author, q['n']))
        lines.append('| 量 | 我方基线 | 大师平均 | 这条线 | 卷参数 |')
        lines.append('|---|---|---|---|---|')
        lines.append('| 色度 a* | %s | %s | %s | a = %s |' % (fmt(o['a_med']), fmt(allf['a_med']), fmt(t['a_med']), fmt(q['a'])))
        lines.append('| 色度 b* | %s | %s | %s | b = %s |' % (fmt(o['b_med']), fmt(allf['b_med']), fmt(t['b_med']), fmt(q['b'])))
        lines.append('| 暗部 b* | %s | %s | %s | b_sh = %s |' % (fmt(o['b_sh']), fmt(allf['b_sh']), fmt(t['b_sh']), fmt(q['b_sh'])))
        lines.append('| 亮部 b* | %s | %s | %s | b_hi = %s |' % (fmt(o['b_hi']), fmt(allf['b_hi']), fmt(t['b_hi']), fmt(q['b_hi'])))
        lines.append('| 彩度中位 | %.2f | %.2f | %.2f | chroma_p=%.3f chroma_s=%.3f |' % (o['c50'], allf['c50'], t['c50'], q['chroma_p'], q['chroma_s']))
        lines.append('| 彩度P90 | %.2f | %.2f | %.2f | 同上 |' % (o['c90'], allf['c90'], t['c90']))
        r50 = o['c50'] * t['c50'] / max(allf['c50'], 1e-6)
        r90 = o['c90'] * t['c90'] / max(allf['c90'], 1e-6)
        lines.append('| 彩度形状 | %.2f | %.2f | %.2f | → %.2f |' % (o['c_ratio'], allf['c_ratio'], t['c_ratio'], r90 / max(r50, 1e-6)))
        lines.append('| 近中性占比 | %.0f%% | %.0f%% | %.0f%% | 随彩度自然跟 |' % (o['gray_pct'] * 100, allf['gray_pct'] * 100, t['gray_pct'] * 100))
        lines.append('| 反差 | %.0f | %.0f | %.0f | contrast=%.3f |' % (o['span90'], allf['span90'], t['span90'], q['contrast']))
        lines.append('| 颗粒 | %.2f | %.2f | %.2f | 强度 %.4f |' % (o['noise'], allf['noise'], t['noise'], q['g_amt']))
        lines.append('| 高调占比 | %.1f%% | %.1f%% | %.1f%% | 黑柔 %.4f |' % (o['hi90'] * 100, allf['hi90'] * 100, t['hi90'] * 100, q['b_amt']))
        lines.append('| 雾量 | %.4f | %.4f | %.4f | veil=%.4f |' % (o['fade_lin'], allf['fade_lin'], t['fade_lin'], q['veil']))
        lines.append('')

    # 三、未标定
    lines.append('## 三、未标定的一卷（诚实标注）\n')
    lines.append('- **富士 Pro 400H**：数据里**没有**对应的"青绿粉彩"作者线（九条线里没有偏青绿低彩的），')
    lines.append('  所以这一卷保持手写近似，`calibrated=False`。要它名副其实，得走色卡标定（路 A）。')
    lines.append('- **颗粒不标定**：作者线 JPEG 的 `noise` 混了 ISO / 机内降噪 / 网盘压缩，')
    lines.append('  是"这张照片有多糙"而不是"这卷胶片颗粒多粗"。所以颗粒**只做 ±35% 相对微调**，')
    lines.append('  底子仍按真胶片知识手写（Ektar 细、C200 粗）。\n')

    # 四、系统性差（待拍板）
    lines.append('## 四、我方中性路径 vs 大师平均（系统性差，**待拍板**，不属于任何一卷）\n')
    d = {k: our[k] - allf[k] for k in ('a_med', 'b_med', 'b_sh', 'b_hi', 'c50', 'c90', 'c_ratio',
                                       'gray_pct', 'span90', 'black', 'fade_lin')}
    lines.append('| 量 | 我方基线 | 大师平均 | 差 | 人话 |')
    lines.append('|---|---|---|---|---|')
    lines.append('| b*中位 | %s | %s | **%s** | 我方整体**偏黄** %.1f 个 b* |' % (fmt(our['b_med']), fmt(allf['b_med']), fmt(d['b_med']), d['b_med']))
    lines.append('| a*中位 | %s | %s | %s | 洋红/绿方向差得不多 |' % (fmt(our['a_med']), fmt(allf['a_med']), fmt(d['a_med'])))
    lines.append('| 彩度中位 | %.2f | %.2f | %s | 我方略**淡** |' % (our['c50'], allf['c50'], fmt(d['c50'])))
    lines.append('| 彩度形状 | %.2f | %.2f | %s | 我方高彩比中彩**冲得更猛** |' % (our['c_ratio'], allf['c_ratio'], fmt(d['c_ratio'])))
    lines.append('| 近中性占比 | %.0f%% | %.0f%% | %s | 我方"不彩的像素"略多 |' % (our['gray_pct'] * 100, allf['gray_pct'] * 100, fmt(d['gray_pct'] * 100, 0) + 'pp'))
    lines.append('| 反差 | %.0f | %.0f | %s | 我方略平 |' % (our['span90'], allf['span90'], fmt(d['span90'], 0)))
    lines.append('| 黑位 | %.1f | %.1f | %s | 我方黑得**更狠**（真黑 vs 胶片的灰黑） |' % (our['black'], allf['black'], fmt(d['black'], 1)))
    lines.append('| 雾量 | %.4f | %.4f | %s | 我方几乎无雾（L1 把黑点压到 0.01） |' % (our['fade_lin'], allf['fade_lin'], fmt(d['fade_lin'], 4)))
    lines.append('')
    lines.append('**要不要动中性路径？** 三档：')
    lines.append('1. **只加雾**（最小）：让黑位别那么死黑，接近胶片。其他不动。')
    lines.append('2. **退黄 + 加雾**：把整体 b* 从 +3.3 收到 +1 左右，同时加雾。画面会明显"没那么黄、黑位发灰"。')
    lines.append('3. **全对齐**：连彩度形状一起收（形状 3.6 → 3.0）。最接近大师平均，但中性路径的成色会变。')
    lines.append('')
    lines.append('> 注：这三条**现在都没做**，卷里也没塞。等你拍板。\n')

    # 五、大师九条线
    lines.append('## 五、大师九条线的带（验收用）\n')
    lines.append('| 作者 | n | 彩度中位 | 彩度P90 | 形状 | 近中性占比 | 暗部b* | 亮部b* | 冷暖分离 | 黑位 | 颗粒 | 高调占比 | 雾量 |')
    lines.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for g in sorted(byg, key=lambda x: -len(byg[x])):
        f = fingerprint(byg[g])
        lines.append('| %s | %d | %.1f | %.1f | %.2f | %.0f%% | %s | %s | %s | %.1f | %.2f | %.1f%% | %.4f |'
                     % (g, len(byg[g]), f['c50'], f['c90'], f['c_ratio'], f['gray_pct'] * 100,
                        fmt(f['b_sh']), fmt(f['b_hi']), fmt(f['split']), f['black'], f['noise'],
                        f['hi90'] * 100, f['fade_lin']))
    lines.append('')

    pj = paths.debug_path(a.purpose, 'stock_calib.json', a.root, a.date)
    with open(pj, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    pm = paths.debug_path(a.purpose, '卷标定报告.md', a.root, a.date)
    with open(pm, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print('\n-> %s\n-> %s' % (pj, pm))
    return out


if __name__ == '__main__':
    main()
