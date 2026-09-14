# -*- coding: utf-8 -*-
"""编排：L0 分析 -> L1 修正 -> L2 风格 -> ┃空间域┃ -> L3 局部 -> L4 护栏。

层序不是随便定的，它是"标准修图顺序"的直接翻译：
  修正在前、风格在后、局部最后、输出只划上限。

空间域（颗粒/黑柔/Halation）夹在 L2 与 L3 之间：它是"胶片的物理过程"，
发生在颜色定下来之后、局部修补之前。见 spatial.py。
层内顺序 = **bloom（镜头）→ halation（乳剂/片基）→ grain（银盐）**
（09-13 晚按 P1-4 改：光学在前，镜头像差发生在乳剂之前）。
"""
from __future__ import annotations

import os
import time

import numpy as np

from . import analyze, color, config as C, denoise, guard, io, local, spatial, spektra, stocks, style, tone


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
        btag = r.get('base_label') or '无'
        b = r.get('entry_bias_ev')
        if b is None:
            etag = '入口无基线'
        else:
            dr = r.get('fuji_dr')
            etag = '入口%+.2fEV%s' % (b, ('(DR%s)' % dr) if dr else '')
        return ('{}  [{}]  {}  |  {}  |  基准{}  |  {}  |  降噪{}  |  修正{}  |  空间{}  |  护栏{}  |  {:.0f}ms'.format(
            self.sample.name, self.sample.kind, analyze.summarize(a), etag, btag,
            stocks.label_of((r.get('style') or {}).get('stock')),
            ('开' if (r.get('denoise') or {}).get('applied') else '关'),
            ('ev%+.2f%s' % (r['tone']['ev_mid'],
                            '(有界兜底)' if r['tone'].get('dark') else ''))
            if r['tone']['applied'] else '未触发',
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


def _entry_bias(sample):
    """入口实际补掉了几档基线曝光（io.load_raw 写在 cam 里）。JPG 路径没有这一项。"""
    if sample.kind != 'raw':
        return None
    v = (sample.cam or {}).get('idt_bias_ev')
    return None if v is None else float(v)




def run(path, src=None, max_side=None, cfg=C, out=None, lut=None, keep_stages=False,
        stock=None, base=None):
    """一次性：**解码 + 跑完整链**。常驻服务请用 `run_from`（跳过解码）。"""
    t0 = time.perf_counter()
    s = io.load(path, max_side or cfg.MAX_SIDE, src=src)
    return run_from(s, cfg=cfg, stock=stock, base=base, out=out, lut=lut,
                    keep_stages=keep_stages, t0=t0, path=path)


def run_from(sample, cfg=C, stock=None, base=None, out=None, lut=None,
             keep_stages=False, t0=None, path=None):
    r"""★ 从**已经 load 好的** sample 起跑 —— 常驻服务的入口。

    ★★ 为什么必须有它：实测 `io.load_raw` **一个人占全链 57%**
    （700 长边：解码 1.911 s / 全链 3.357 s）⇒ **常驻 + 缓存 sample**
    ⇒ 换一次卷从 3.36 s 降到 **1.45 s**（见 README「对外入口」）。
    ⚠ **一份实现、两条入口**：`run()` = `io.load()` + `run_from()` —— 不许各写一套。

    ⚠ 缓存的 sample 必须**同尺寸**（`io.load(..., max_side)` 的产物）；换尺寸要重新 load。
    """
    t0 = time.perf_counter() if t0 is None else t0
    s = sample
    path = path if path is not None else getattr(s, 'path', None)
    st = _stock_of(stock, cfg)

    # ★★ 位置由「脸」的锚点决定（09-14 SV 选「乙」）：由脸算一个曝光偏移，
    #   **把入口那条曲线整体重打**。不是"分区域压脸/压背景"—— 一条曲线、不用掩膜，
    #   背景的亮度是这条曲线算出来的**结果**。
    # ★★ 真卷（`spek=`）**不用锚点**：实测真卷自己就把脸放到 L* 78~86（比我们靶 68 还亮），
    #   再提一遍就是过曝（中位 61 → 83）。⇒ 真卷模式下位置整段交给胶片。
    _pre_spek = (st or {}).get('spek')
    if _pre_spek and not bool(getattr(cfg, 'SPEK_ANCHOR', False)):
        d_ev, anc = 0.0, dict(applied=False, reason='real_stock_真卷自己定曝光')
    else:
        d_ev, anc = io.anchor_ev(s.disp, cfg)
    _on = bool(anc.get('applied'))
    lin_in = io.refocus(s.lin, d_ev, cfg) if _on else s.lin
    if _on:
        # ★★ 09-14 评审修：锚点把入口曲线**重打**了 ⇒ 入口那道裁切护栏（`clip_guard`）
        #   是**在重打之前**算的，结论已经失效 ⇒ 这里**必须再跑一遍**。
        #   不跑的话"提亮救脸"会顺手绕过护栏把背景推爆（实测 0805 背景 75→96、护栏当时算出 k=1.0）。
        lin_in, _gk2 = io.clip_guard(lin_in, cfg)
        anc['clip_guard_k'] = _gk2
        if abs(_gk2 - 1.0) > 1e-9:
            anc['clip_guard_ev'] = float(np.log2(_gk2))
    disp_in = (np.clip(color.l2s(np.clip(lin_in, 0.0, None)), 0.0, 1.0)
               if _on else s.disp)

    rep0 = analyze.analyze(lin_in, disp_in, s.kind)                  # L0
    # L1 影调修正（**只压不提**：提亮交给入口 settle + 脸锚点，兜底提亮那套 09-14 已删）
    # ★★ 09-14 SV：「丢弃作者线，全部用真卷」。
    #   真卷（带 `spek=` 标记）**自带完整 H&D 曲线** ⇒ 我们的 L1 影调修正要**让位**（不然是两条曲线串）。
    _spek = (st or {}).get('spek')
    if _spek:
        lin1, t_info = lin_in, dict(applied=False, reason='real_stock_自带H&D曲线')
        disp1 = disp_in
    else:
        lin1, t_info = tone.correct(lin_in, rep0, cfg)               # L1
        disp1 = np.clip(color.l2s(np.clip(lin1, 0.0, 1.0)), 0.0, 1.0)
    disp1, d_info = denoise.apply(disp1, cfg)                        # 降噪（L1 之后、L2 之前）

    if lut is None and cfg.LUT_PATH:
        lut = style.cube_read(cfg.LUT_PATH)
    # 锁中灰的参照 = 修正层实际交出来的中灰（不是配置里的靶）
    if _spek:
        # ★★ L2 整段换成**真卷**：喂**场景线性**（`lin_in`），出显示域。
        #   它自带 H&D + `dir_couplers`(彩度) + 染料；落点由 `SPEK_PRINT_EXPOSURE` 定。
        # ★ 每卷一个 pe（09-14 标定：不同相纸响应不同 ⇒ 全局一个值会让富士卷偏亮 30 个 L*）
        # ★★ 09-14 新增：再乘一个**逐张微调系数** `SPEK_PE_SHIFT`。
        #   为什么要分开：全局改 `SPEK_PRINT_EXPOSURE` **会被每卷的 pe 盖掉**（实测三档同值）
        #   ⇒ 落点滑杆一直是死的。改成「每卷基准 × 全局系数」后它才真的动得了画面。
        #   用途：救被闪光顶亮的片（1065/1067）—— 只压这一张，别的片不动。
        _base_pe = float(_spek.get('pe') or getattr(cfg, 'SPEK_PRINT_EXPOSURE', 0.55))
        _shift = float(getattr(cfg, 'SPEK_PE_SHIFT', 1.0) or 1.0)
        _pe = _base_pe * _shift
        disp2 = spektra.render(lin_in, st['name'], cfg, print_exposure=_pe)
        s_info = dict(applied=True, how='spektrafilm', stock=st['name'],
                      film=_spek.get('film'), print=_spek.get('print'),
                      print_exposure=_pe, pe_base=_base_pe, pe_shift=_shift,
                      tone_curve=False, film_color_w=0.0)
    else:
        disp2, s_info = style.apply(disp1, cfg, lut=lut, lock_ref=style.mid_of(disp1),
                                    stock=st, base=base)
    # ★ 锚点**收尾**（09-14 SV 选「①」）：L1 那一步把脸放到靶上了，但 **L2 影调曲线又把它抬上去**
    #   （实测 +8.4 L*）⇒ 这里量一次脸、用**全局增益**把它挪回靶 ⇒ **最终脸真的落在靶上**。
    #   只在锚点真的动过（脸偏暗）时才做；仍是一条曲线，不分区。
    if _on and bool(getattr(cfg, 'ANCHOR_FINISH', True)):
        disp2, _fin = io.finish_anchor(disp2, cfg)
        anc['finish'] = _fin
    # ⚠ 09-14 SV：「把脸部立体感的部分删掉」⇒ 原来在 L2 之后 / 空间层之后各插一道
    #   `local.face_tone`（第 4 条「每层护脸」），**已整段删除**。
    disp2r = disp2
    if _spek:
        # ★ 真卷自带的 grain / halation / glare 已经是物理级的（分通道、R 最强 ⇒ 红橙）
        #   ⇒ 我们的空间层**必须让位**，不然是两套颗粒叠一起。
        disp2b, sp_info = disp2, dict(applied=False, reason='real_stock_自带颗粒/halation')
    else:
        disp2b, sp_info = spatial.apply(disp2, cfg, stock=st)        # 空间域（颗粒/黑柔/Halation）
    disp2br = disp2b
    disp3, l_info = local.apply(disp1, disp2b, cfg)                  # L3
    disp4, g_info = guard.enforce(disp3, cfg)                        # L4

    rep = dict(
        camera=s.cam,
        entry_bias_ev=_entry_bias(s),
        fuji_dr=(s.cam or {}).get('fuji_dr'),
        stock=(st or {}).get('name'),
        stock_label=stocks.label_of((st or {}).get('name')),
        base=stocks.resolve_base(cfg, base)['name'],
        base_label=stocks.base_label(cfg, base),
        analyze=rep0,
        tone=t_info,
        denoise=d_info,
        style=s_info,
        spatial=sp_info,
        local=l_info,
        anchor=anc,
        guard=g_info,
        ms=(time.perf_counter() - t0) * 1000.0,
    )
    if keep_stages:
        rep['stages'] = dict(
            base=disp_in,           # 入口归一后的样子（RAW 就是线性直出；锚点已重打过）
            after_tone=disp1,       # 做完 L1 影调修正 + 降噪
            after_style=disp2,      # 再过 L2 风格（含第 4 条护脸那道）
            after_spatial=disp2b,   # 再过空间域（含第 4 条护脸那道）
            after_local=disp3,
            style_raw=disp2r,       # ★ 没护脸的 L2 出口（探针用：看"护脸"到底动了多少）
            spatial_raw=disp2br,    # ★ 没护脸的空间层出口
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
