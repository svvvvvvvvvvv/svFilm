# -*- coding: utf-8 -*-
"""度量层 —— 和"大师作品再学习"那把尺子**完全同口径**的指纹。

为什么单独一个模块：卷的数值要"从数据里量出来"，那就必须有一把**两边都能量**的尺子。
`_debug/analysis/master_resurvey.json` 是用这把尺子量大师的 1172 张（1024 长边、Lab D65）；
这里复刻同一套定义，用来量**我们自己的成片**，两边一比才知道差在哪、要补多少。

定义（照抄 `_debug/lab_master_resurvey.py`，改任何一个都要重跑标定）：
  L{q}    Lab L* 的分位
  c50/c90 Lab 彩度 C 的中位 / P90
  a_med/b_med  Lab a*/b* 中位
  b_sh/b_hi    L* ≤ P25 的像素的 b* 中位 / L* ≥ P90 的 b* 中位  ← 冷暖分离
  split        b_hi − b_sh
  gray_pct     近中性像素占比（C < 8）
  span90       L* P95 − P5（反差）
  black        L* P1（黑位地板）
  noise        平坦区高频 std（颗粒/噪点，0~255 灰度口径）
  fade_lin     线性光域黑位抬升比（雾量）
"""
from __future__ import annotations

import numpy as np

from . import color

NEUTRAL_C = 8.0            # 近中性的彩度阈值（与大师口径一致）
SIZE = 1024                # 测量分辨率（与大师口径一致）


def _resize_u8(u8, size=SIZE):
    h, w = u8.shape[:2]
    m = max(h, w)
    if m <= size:
        return u8
    import cv2
    s = size / float(m)
    return cv2.resize(u8, (max(int(round(w * s)), 1), max(int(round(h * s)), 1)),
                      interpolation=cv2.INTER_AREA)


def _blur255(g255, r):
    """0~255 灰度上做高斯（用 PIL，跟大师那把尺子的实现一致）。"""
    from PIL import Image, ImageFilter
    im = Image.fromarray(np.clip(g255, 0, 255).astype(np.uint8))
    return np.asarray(im.filter(ImageFilter.GaussianBlur(r))).astype(np.float32)


def fingerprint(disp_or_u8):
    """输入 disp(0~1 float) 或 uint8，返回指纹 dict（含 'n_px'）。"""
    a = np.asarray(disp_or_u8)
    if a.dtype != np.uint8:
        u8 = color.display_to_u8(np.clip(a, 0.0, 1.0))
    else:
        u8 = a
    u8 = _resize_u8(u8)
    d = u8.astype(np.float32) / 255.0

    lab = color.to_lab(d.astype(np.float64))
    L, A, B = lab[..., 0], lab[..., 1], lab[..., 2]
    C = np.sqrt(A ** 2 + B ** 2)

    P = {q: float(np.percentile(L, q)) for q in (1, 2, 5, 10, 25, 50, 75, 90, 95, 98, 99)}
    m_sh, m_hi = L <= P[25], L >= P[90]

    g255 = color.gray_of(d.astype(np.float64)) * 255.0
    bl2, bl8 = _blur255(g255, 2.0), _blur255(g255, 8.0)
    hp = g255 - bl2
    flat = (L > 20) & (L < 85) & (np.abs(bl8 - g255) < 6.0)

    Y = (0.2126 * d[..., 0] ** 2.2 + 0.7152 * d[..., 1] ** 2.2 + 0.0722 * d[..., 2] ** 2.2)
    y2, y50 = float(np.percentile(Y, 2)), float(np.percentile(Y, 50))

    out = {('L%d' % q): v for q, v in P.items()}
    out.update(
        Lmean=float(L.mean()), Lstd=float(L.std()),
        span=float(P[98] - P[2]), span90=float(P[95] - P[5]),
        sh1=float((L < 10).mean()), sh20=float((L < 20).mean()),
        hi90=float((L > 90).mean()), hi97=float((L > 97).mean()),
        black=float(P[1]),
        c50=float(np.median(C)), c90=float(np.percentile(C, 90)), cmean=float(C.mean()),
        a_med=float(np.median(A)), b_med=float(np.median(B)),
        b_sh=float(np.median(B[m_sh])) if m_sh.sum() > 50 else 0.0,
        b_hi=float(np.median(B[m_hi])) if m_hi.sum() > 50 else 0.0,
        gray_pct=float((C < NEUTRAL_C).mean()),
        fade_lin=(y2 / max(y50 - y2, 1e-6)),
        noise=float(hp[flat].std()) if flat.sum() > 2000 else float('nan'),
        local=float(hp.std()),
        n_px=int(L.size),
    )
    out['split'] = out['b_hi'] - out['b_sh']
    out['c_ratio'] = out['c90'] / max(out['c50'], 1e-6)
    return out


# ---- 给人看的名字（汇报一律用这个，别甩代号） ----
CN = {
    'L50': 'L*中位', 'span90': '反差', 'c50': '彩度中位', 'c90': '彩度P90',
    'c_ratio': '彩度形状', 'gray_pct': '近中性占比', 'a_med': '色度a*（+红/−绿）',
    'b_med': '色度b*（+黄/−蓝）', 'b_sh': '暗部b*', 'b_hi': '亮部b*',
    'split': '冷暖分离', 'black': '黑位地板', 'fade_lin': '雾量', 'noise': '颗粒',
}

# 做卷时要"落到带里"的四条（大师九人的范围，见 master_doc_0912.md §五）
BAND = {
    'gray_pct': (0.47, 0.75),
    'c_ratio': (2.65, 3.41),
    'c90': (11.7, 25.9),
    'black': (0.0, 13.0),
}


def compare_json(json_path, root, n=24, seed=0):
    """尺子校验：用本模块量大师原图，与既有的 master_resurvey.json 对比。"""
    import json
    import os
    d = json.load(open(json_path, encoding='utf-8'))
    rows = [r for r in d['rows'] if r.get('path') and r.get('c50')]
    rows = [r for r in rows if os.path.exists(r['path'])]
    rng = np.random.default_rng(seed)
    rows = [rows[i] for i in rng.choice(len(rows), min(n, len(rows)), replace=False)]
    keys = ['L50', 'L95', 'span90', 'black', 'c50', 'c90', 'cmean', 'a_med', 'b_med',
            'b_sh', 'b_hi', 'gray_pct', 'fade_lin', 'noise']
    diffs = {k: [] for k in keys}
    from PIL import Image, ImageOps
    for r in rows:
        im = ImageOps.exif_transpose(Image.open(r['path'])).convert('RGB')
        f = fingerprint(np.asarray(im, np.uint8))
        for k in keys:
            if r.get(k) is None or not np.isfinite(r.get(k, np.nan)) or not np.isfinite(f[k]):
                continue
            diffs[k].append(abs(float(f[k]) - float(r[k])))
    print('%-10s %-10s %-10s %s' % ('指标', '中位差', '最大差', '人话'))
    worst = 0.0
    for k in keys:
        v = np.asarray(diffs[k])
        if len(v) == 0:
            continue
        print('%-10s %-10.3f %-10.3f %s' % (k, np.median(v), v.max(), CN.get(k, '')))
        worst = max(worst, float(np.median(v)))
    return worst
