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

from . import analyze, color, config as C, denoise, guard, io, local, spatial, stocks, style, tone


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


def _face_guard_on(cfg):
    """★ 第 4 条「每层护脸」是否开：整层脸层开着 **且** `FACE_GUARD_LAYERS` 为真。

    关掉（或把 `FACE_ENABLE` 设 False）⇒ 逐位回到"只在 L3 末尾护一道"的老行为。
    """
    return bool(getattr(cfg, 'FACE_ENABLE', False)) and bool(getattr(cfg, 'FACE_GUARD_LAYERS', True))


def _entry_bias(sample):
    """入口实际补掉了几档基线曝光（io.load_raw 写在 cam 里）。JPG 路径没有这一项。"""
    if sample.kind != 'raw':
        return None
    v = (sample.cam or {}).get('idt_bias_ev')
    return None if v is None else float(v)


def _allow_lift(sample, cfg, rep=None):
    """能不能让曝光层（L1）兜底提亮。

    三条，按顺序：
      1) JPG / 关掉 RAW 提亮 ——> 不许（JPG 提亮 = 把被压过的颜色按斜率放大）。
      2) 入口**没补过**基线曝光（非富士 / 读不到 tag）——> 老兜底，允许。
      3) 入口**补过** ——> 09-13 SV 拍板「乙」第 2 步起**不再一刀切**。
         入口换成「零点 + 固定成形」后，成形是固定的 ⇒ **场景本身暗**的图会被忠实压在很低的地方
         （实测 DSCF0547 成片 L*50 只有 17.4，大师·高反差带下沿是 33）。那是**曝光**问题，
         该由曝光层**有界地**补回来。判据 = `analyze` 报的 `dark_lift`（中间调低于带下沿）；
         不满足就照旧不许提 —— 这样"亮场/大反差"那批**逐位不变**。
    """
    if sample.kind != 'raw':
        return bool(cfg.ALLOW_LIFT_JPG)
    if not cfg.ALLOW_LIFT_RAW:
        return False
    b = _entry_bias(sample)
    if b is None or abs(b) <= 1e-6:
        return True
    if not cfg.AUTO_LIFT_ONLY_WHEN_NO_ENTRY_BIAS:
        return True
    # ★ 位置逐张听相机（09-13 深夜）：入口落点已由「机型×DR 实测落点规律」**逐张**定过 ⇒
    #   兜底提亮必须让位，否则同一个"位置"会被补两次（规律本来就把暗片放到相机的位置上了）。
    if (sample.cam or {}).get('entry_settle'):
        return False
    return bool(rep is not None and rep.get('dark_lift'))


def run(path, src=None, max_side=None, cfg=C, out=None, lut=None, keep_stages=False,
        stock=None, base=None):
    t0 = time.perf_counter()
    st = _stock_of(stock, cfg)
    s = io.load(path, max_side or cfg.MAX_SIDE, src=src)

    # ★★ 位置由「脸」的锚点决定（09-14 SV 选「乙」）：由脸算一个曝光偏移，
    #   **把入口那条曲线整体重打**。不是"分区域压脸/压背景"—— 一条曲线、不用掩膜，
    #   背景的亮度是这条曲线算出来的**结果**。
    d_ev, anc = io.anchor_ev(s.disp, cfg)
    _on = bool(anc.get('applied'))
    lin_in = io.refocus(s.lin, d_ev, cfg) if _on else s.lin
    disp_in = (np.clip(color.l2s(np.clip(lin_in, 0.0, None)), 0.0, 1.0)
               if _on else s.disp)

    rep0 = analyze.analyze(lin_in, disp_in, s.kind)                  # L0
    allow = _allow_lift(s, cfg, rep0)
    lin1, t_info = tone.correct(lin_in, rep0, cfg, allow_lift=allow)  # L1
    disp1 = np.clip(color.l2s(np.clip(lin1, 0.0, 1.0)), 0.0, 1.0)
    disp1, d_info = denoise.apply(disp1, cfg)                        # 降噪（L1 之后、L2 之前）

    if lut is None and cfg.LUT_PATH:
        lut = style.cube_read(cfg.LUT_PATH)
    # 锁中灰的参照 = 修正层实际交出来的中灰（不是配置里的靶）
    disp2, s_info = style.apply(disp1, cfg, lut=lut, lock_ref=style.mid_of(disp1),
                                stock=st, base=base)
    # ★ 锚点**收尾**（09-14 SV 选「①」）：L1 那一步把脸放到靶上了，但 **L2 影调曲线又把它抬上去**
    #   （实测 +8.4 L*）⇒ 这里量一次脸、用**全局增益**把它挪回靶 ⇒ **最终脸真的落在靶上**。
    #   只在锚点真的动过（脸偏暗）时才做；仍是一条曲线，不分区。
    if _on and bool(getattr(cfg, 'ANCHOR_FINISH', True)):
        disp2, _fin = io.finish_anchor(disp2, cfg)
        anc['finish'] = _fin
    # ★ 第 4 条「每层护脸」（09-14 SV 拍板「乙」，靶 68）：L2 之后先护一道 ——
    #   影调曲线会把脸拉平；而真正的主力是后面的空间层（**黑柔 + 颗粒**，实测压掉脸跨度 17~31%），
    #   所以空间层之后还要再护一道。靶是**绝对值** ⇒ 后层压下去、下一道就补回来。
    #   两处都复用 `local.face_tone`，不新写动作；关掉 `FACE_GUARD_LAYERS` 即回到老行为。
    if _face_guard_on(cfg):
        disp2r = disp2
        disp2, fg1 = local.face_tone(disp2, cfg)
    else:
        disp2r, fg1 = disp2, dict(applied=False, reason='off')
    disp2b, sp_info = spatial.apply(disp2, cfg, stock=st)            # 空间域（颗粒/黑柔/Halation）
    if _face_guard_on(cfg):
        disp2br = disp2b
        disp2b, fg2 = local.face_tone(disp2b, cfg)
    else:
        disp2br, fg2 = disp2b, dict(applied=False, reason='off')
    disp3, l_info = local.apply(disp1, disp2b, cfg)                  # L3（内部再护一道，保留）
    disp4, g_info = guard.enforce(disp3, cfg)                        # L4

    rep = dict(
        camera=s.cam,
        entry_bias_ev=_entry_bias(s),
        fuji_dr=(s.cam or {}).get('fuji_dr'),
        allow_lift=allow,
        dark_lift=bool(rep0.get('dark_lift')),
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
        face_guard=dict(after_style=fg1, after_spatial=fg2),
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
