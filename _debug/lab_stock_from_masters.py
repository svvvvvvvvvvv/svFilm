# -*- coding: utf-8 -*-
"""从大师作品的逐图颜色指纹里"量出"胶片卷 —— 路 B。

思路（不碰大师原图，只用已量好的逐图指纹）：
  1) 读 `_debug/analysis/master_resurvey.json`（1172 张，逐张 Lab 颜色 + 影调）
  2) 用**颜色**特征聚类（色度方向 / 冷暖分离 / 彩度形状 / 中性占比），不掺影调
     —— 因为影调已经由 L1 的绝对靶管掉了，卷只该管"颜色性格"
  3) 每族量出中位指纹 → 这就是"这一族长什么样"
  4) 每族的**相对全体中位的偏移**才是可用信号（绝对中位里混着内容差异）

输出（全部走 paths.py）：指纹统计 json + 人话报告 md + 逐族代表图接触表
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from svFilm import paths  # noqa: E402

MASTER_JSON = 'E:/WorkBuddy/摄影助手/_debug/analysis/master_resurvey.json'
MASTER_ROOT = 'E:/WorkBuddy/摄影助手/大师作品'

# 聚类用的颜色特征（全部来自逐图指纹）
FEATS = ['a_med', 'b_med', 'b_sh', 'b_hi', 'c50', 'c90', 'cmean', 'gray_pct']
FEAT_CN = {
    'a_med': '色度a*中位（+红/−绿）', 'b_med': '色度b*中位（+黄/−蓝）',
    'b_sh': '暗部b*（正=暗部暖，负=暗部冷）', 'b_hi': '亮部b*（正=亮部暖）',
    'c50': '彩度中位', 'c90': '彩度P90', 'cmean': '彩度均值',
    'gray_pct': '近中性像素占比', 'split': '冷暖分离(b_hi−b_sh)',
    'c_ratio': '彩度形状(c90/c50)', 'L50': 'L*中位', 'span90': '反差',
    'black': '黑位地板', 'fade_lin': '雾量', 'noise': '颗粒',
}


def load():
    d = json.load(open(MASTER_JSON, encoding='utf-8'))
    rows = d['rows']
    keep = []
    for r in rows:
        if any(r.get(k) is None for k in FEATS):
            continue
        if r.get('c50', 0) <= 0:
            continue
        keep.append(r)
    return keep


def matrix(rows):
    X = np.asarray([[float(r[k]) for k in FEATS] for r in rows], np.float64)
    return X


def zscore(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-9
    return (X - mu) / sd, mu, sd


def kmeans(X, k, seed=0, iters=200):
    rng = np.random.default_rng(seed)
    # k-means++ 初始化
    C = [X[rng.integers(len(X))]]
    for _ in range(k - 1):
        dist = np.min(((X[:, None, :] - np.asarray(C)[None]) ** 2).sum(-1), axis=1)
        p = dist / max(dist.sum(), 1e-12)
        C.append(X[rng.choice(len(X), p=p)])
    C = np.asarray(C, np.float64)
    for _ in range(iters):
        lab = ((X[:, None, :] - C[None]) ** 2).sum(-1).argmin(1)
        newC = np.stack([X[lab == j].mean(0) if np.any(lab == j) else C[j] for j in range(k)])
        if np.allclose(newC, C, atol=1e-8):
            break
        C = newC
    return lab, C


def silhouette(X, lab):
    """简化轮廓系数（抽样，够用）。"""
    n = len(X)
    idx = np.arange(n)
    if n > 600:
        idx = np.random.default_rng(0).choice(n, 600, replace=False)
    S = []
    for i in idx:
        same = (lab == lab[i])
        same[i] = False
        a = np.linalg.norm(X[i] - X[same], axis=1).mean() if same.any() else 0.0
        bs = [np.linalg.norm(X[i] - X[lab == j], axis=1).mean()
              for j in set(lab) if j != lab[i]]
        b = min(bs) if bs else 0.0
        S.append((b - a) / max(a, b, 1e-9))
    return float(np.mean(S))


def fingerprint(rows):
    """一组的颜色指纹（中位 + 派生量）。"""
    f = {}
    for k in FEATS:
        f[k] = float(np.median([float(r[k]) for r in rows]))
    for k in ('L50', 'span90', 'black', 'fade_lin', 'noise', 'split'):
        f[k] = float(np.median([float(r.get(k) or 0.0) for r in rows]))
    f['split'] = f['b_hi'] - f['b_sh']
    f['c_ratio'] = f['c90'] / max(f['c50'], 1e-6)
    f['n'] = len(rows)
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k', type=int, default=0, help='族数；0 = 自动挑')
    ap.add_argument('--purpose', default='卷标定_大师颜色聚类')
    ap.add_argument('--root', default='E:/WorkBuddy/摄影助手')
    ap.add_argument('--date', default=None)
    a = ap.parse_args()

    rows = load()
    X = matrix(rows)
    Z, mu, sd = zscore(X)
    print('样本 %d 张，特征 %d 维' % (len(rows), X.shape[1]))

    if a.k:
        ks = [a.k]
    else:
        ks = list(range(3, 11))
    print('\n%-4s %-10s %-10s  %s' % ('k', '轮廓系数', '族内最小n', '作者纯度(最大族占单一作者的比例)'))
    best = None
    scores = {}
    for k in ks:
        lab, C = kmeans(Z, k, seed=7)
        s = silhouette(Z, lab)
        sizes = [int((lab == j).sum()) for j in range(k)]
        # 作者纯度：每族里出现最多的作者占该族比例，按族大小加权
        groups = np.array([r['group'] for r in rows])
        purity = 0.0
        for j in range(k):
            g = groups[lab == j]
            if len(g) == 0:
                continue
            _, cnt = np.unique(g, return_counts=True)
            purity += (cnt.max() / len(g)) * (len(g) / len(rows))
        scores[k] = (s, lab, C, sizes, purity)
        print('%-4d %-10.3f %-10d %.2f' % (k, s, min(sizes), purity))

    if a.k:
        k = a.k
    else:
        # 自动挑：轮廓系数为主，要求每族至少 40 张
        cand = [(scores[k][0], k) for k in ks if min(scores[k][3]) >= 40]
        k = max(cand)[1] if cand else max(scores, key=lambda kk: scores[kk][0])
    s, lab, C, sizes, purity = scores[k]
    print('\n选中 k=%d（轮廓 %.3f，作者纯度 %.2f）' % (k, s, purity))

    # 族按大小排序重编号（0 = 最大族）
    order = np.argsort([-int((lab == j).sum()) for j in range(k)])
    remap = {int(o): i for i, o in enumerate(order)}
    lab = np.array([remap[int(v)] for v in lab])

    allf = fingerprint(rows)
    out = dict(k=k, silhouette=s, all=allf, clusters=[])
    print('\n全体中位: ' + '  '.join('%s %.2f' % (FEAT_CN[k2].split('（')[0], allf[k2])
                                 for k2 in ('a_med', 'b_med', 'b_sh', 'b_hi', 'c50', 'c90', 'gray_pct')))
    print('\n' + '=' * 100)
    for j in range(k):
        sel = [rows[i] for i in range(len(rows)) if lab[i] == j]
        f = fingerprint(sel)
        groups = list(np.array([r['group'] for r in sel]))
        uni, cnt = np.unique(groups, return_counts=True)
        top = sorted(zip(uni, cnt), key=lambda t: -t[1])
        f['top_authors'] = [[str(u), int(c)] for u, c in top[:5]]
        f['paths'] = [r['path'] for r in sel[:60]]
        out['clusters'].append(f)
        print('族 %-2d  n=%-4d  作者: %s' % (
            j, f['n'], '  '.join('%s %d' % (u, c) for u, c in top[:4])))
        print('     中位L*%5.1f 反差%5.1f | 彩度50 %5.1f  彩度90 %5.1f  形状%4.2f  中性占比%5.1f%%'
              % (f['L50'], f['span90'], f['c50'], f['c90'], f['c_ratio'], f['gray_pct']))
        print('     色度 a*%+5.1f b*%+5.1f | 暗部b*%+6.1f 亮部b*%+6.1f 冷暖分离%+5.1f | 黑位%4.1f 雾量%.3f'
              % (f['a_med'], f['b_med'], f['b_sh'], f['b_hi'], f['split'], f['black'], f['fade_lin']))
    print('=' * 100)

    # 偏移量（相对全体中位）—— 这才是可以搬进卷的信号
    print('\n相对全体中位的偏移（卷要复现的就是这个）：')
    for j, c in enumerate(out['clusters']):
        d = {k2: c[k2] - allf[k2] for k2 in ('a_med', 'b_med', 'b_sh', 'b_hi', 'c50', 'c90', 'gray_pct')}
        print('族 %-2d  Δa%+5.2f Δb%+5.2f  Δb_sh%+6.2f Δb_hi%+6.2f  Δc50%+5.2f Δc90%+5.2f  Δ中性%+5.1f'
              % (j, d['a_med'], d['b_med'], d['b_sh'], d['b_hi'], d['c50'], d['c90'], d['gray_pct']))

    p = paths.debug_path(a.purpose, 'clusters.json', a.root, a.date)
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print('\n-> %s' % p)
    return out


if __name__ == '__main__':
    main()
