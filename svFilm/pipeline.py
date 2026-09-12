# -*- coding: utf-8 -*-
"""编排：L0 分析 -> L1 修正 -> L2 风格 -> ┃空间域┃ -> L3 局部 -> L4 护栏。

层序不是随便定的，它是"标准修图顺序"的直接翻译：
  修正在前、风格在后、局部最后、输出只划上限。

空间域（颗粒/黑柔/Halation）夹在 L2 与 L3 之间：它是"胶片的物理过程"，
发生在颜色定下来之后、局部修补之前。见 spatial.py。
"""
from __future__ import annotations

import os
import time

import numpy as np

from . import analyze, color, config as C, guard, io, local, spatial, stocks, style, tone


class Result:
    __slots__ = ('disp', 'report', 'sample', 'path')

    def __init__(self, disp, report, sample, path):
        self.disp = disp
        self.report = report
        self.sample = sample
        self.path = path

    def save(self, out_path, exif=True):
        return io.save(self.disp, out_path,
                       exif=(self.sample.exif if exif else None),
                       quality=C.JPEG_QUALITY)

    def summary(self):
        r = self.report
        a = r['analyze']
        sp = r.get('spatial') or {}
        tag = []
        if sp.get('grain', {}).get('applied'):
            tag.append('颗粒')
        if sp.get('bloom', {}).get('applied'):
            tag.append('黑柔')
        if sp.get('halation', {}).get('applied'):
            tag.append('Halation')
        return ('{}  [{}]  {}  |  {}  |  修正{}  |  空间{}  |  护栏{}  |  {:.0f}ms'.format(
            self.sample.name, self.sample.kind, analyze.summarize(a),
            stocks.label_of((r.get('style') or {}).get('stock')),
            ('ev%+.2f' % r['tone']['ev_mid']) if r['tone']['applied'] else '未触发',
            ('+'.join(tag) if tag else '无'),
            ('/'.join(r['guard']['actions']) if r['guard']['actions'] else '无'),
            r['ms']))


def _stock_of(stock, cfg):
    """stock 可以是卷名字符串 / 卷 dict / None（None 时取 config.STOCK）。"""
    if stock is None:
        return stocks.get(cfg.STOCK)
    if isinstance(stock, str):
        return stocks.get(stock)
    return stock


def run(path, src=None, max_side=None, cfg=C, out=None, lut=None, keep_stages=False, stock=None):
    t0 = time.perf_counter()
    st = _stock_of(stock, cfg)
    s = io.load(path, max_side or cfg.MAX_SIDE, src=src)

    rep0 = analyze.analyze(s.lin, s.disp, s.kind)                    # L0
    lin1, t_info = tone.correct(s.lin, rep0, cfg)                    # L1
    disp1 = np.clip(color.l2s(np.clip(lin1, 0.0, 1.0)), 0.0, 1.0)

    if lut is None and cfg.LUT_PATH:
        lut = style.cube_read(cfg.LUT_PATH)
    # 锁中灰的参照 = 修正层实际交出来的中灰（不是配置里的靶）
    disp2, s_info = style.apply(disp1, cfg, lut=lut, lock_ref=style.mid_of(disp1), stock=st)
    disp2b, sp_info = spatial.apply(disp2, cfg, stock=st)            # 空间域（颗粒/黑柔/Halation）
    disp3, l_info = local.apply(disp1, disp2b, cfg)                  # L3
    disp4, g_info = guard.enforce(disp3, cfg)                        # L4

    rep = dict(
        camera=s.cam,
        stock=(st or {}).get('name'),
        stock_label=stocks.label_of((st or {}).get('name')),
        analyze=rep0,
        tone=t_info,
        style=s_info,
        spatial=sp_info,
        local=l_info,
        guard=g_info,
        ms=(time.perf_counter() - t0) * 1000.0,
    )
    if keep_stages:
        rep['stages'] = dict(
            base=s.disp,            # 入口归一后的样子（RAW 就是线性直出）
            after_tone=disp1,       # 只做完 L1 影调修正
            after_style=disp2,      # 再过 L2 风格
            after_spatial=disp2b,   # 再过空间域
            after_local=disp3,
        )
    res = Result(disp4, rep, s, path)
    if out:
        res.save(out)
    return res


def run_pair(stem_or_path, prefer='raw', **kw):
    """同一张照片，RAW 与机内 JPG 各跑一遍（用于验证"换底"）。"""
    p = io.find_pair(stem_or_path, prefer='raw')
    a = run(p, src='raw', **kw)
    stem = os.path.splitext(stem_or_path)[0]
    jp = None
    for e in ('.JPG', '.jpg', '.JPEG', '.jpeg'):
        if os.path.exists(stem + e):
            jp = stem + e
            break
    b = run(jp, src='jpg', **kw) if jp else None
    return a, b
