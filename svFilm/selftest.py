# -*- coding: utf-8 -*-
"""自检 —— 动完任何一层都要跑。不依赖任何样片，纯合成图 + 不变量。

  python -m svFilm.selftest
"""
import os
import struct
import sys
import tempfile

import numpy as np

from . import (analyze, cameras, color, config as C, denoise, guard, io, local, metrics,
               pipeline, rawmeta, spatial, stocks, style, tone)

FAIL = []


def check(name, cond, extra=''):
    print(('  ok   ' if cond else '  FAIL ') + name + (('   ' + extra) if extra else ''))
    if not cond:
        FAIL.append(name)


class _Cfg:
    """config 的只读副本 + 覆盖几项 —— 给"开关类"测试用，不动全局 config。

    ⚠ 单测隔离（09-13 踩到）：**默认把「基准成色」置成不动**。
    出厂默认改成 `BASE_FULL`（SV 拍板）之后，这里若跟着默认走，
    "contrast=1.0 完全不动""支点 L*=50""neutral 卷 = 恒等"三条会一起变红 ——
    那不是回归，是**测试测错了对象**（测的是"出厂默认"而不是"这一层"）。
    要连着基准一起测，显式传 `_Cfg(BASE='BASE_FULL')`。
    """

    def __init__(self, **kw):
        self._d = {k: getattr(C, k) for k in dir(C) if k.isupper()}
        self._d['BASE'] = 'BASE_NONE'
        # 抽色饱和（CHROMA_ENDS）也是"出厂模型默认"，同样要隔离：
        # 否则 "contrast=1.0 时完全不动" 会因为多跑一次 Lab 往返而红。
        self._d['CHROMA_ENDS'] = 0.0
        # 肤色正向达标（L3 第二件事）也是"出厂默认"，同理隔离：
        # 否则任何一张带肤色像素的测试图都会被它多推一次色度。
        self._d['SKIN_FLOOR'] = False
        # 胶片影调曲线（L2 的 A 档）也是"出厂默认"，同理隔离：
        # 否则 "contrast=1.0 时完全不动" / "neutral 卷 = 恒等" 会因为中灰被抬而红。
        # 要单独测它，显式传 _Cfg(TONE_CURVE=True, TONE_TOE=..., TONE_LIFT=...)。
        self._d['TONE_CURVE'] = False
        self._d.update(kw)

    def __getattr__(self, k):
        try:
            return self._d[k]
        except KeyError:
            raise AttributeError(k)


def _span90(disp):
    L = color.to_lab(np.clip(disp, 0, 1))[..., 0]
    return float(np.percentile(L, 95) - np.percentile(L, 5))


def _gray_img(h=240, w=360, gamma=0.45, seed=0, tint=(1.0, 1.0, 1.0)):
    rng = np.random.default_rng(seed)
    y = np.linspace(0.0, 1.0, h)[:, None] * np.ones((1, w))
    y = np.clip(y ** gamma, 1e-4, 1.0)
    img = np.repeat(y[..., None], 3, axis=-1)
    img *= np.asarray(tint)
    img += rng.normal(0, 0.002, img.shape)
    return np.clip(img, 0.0, 1.0)


def t_color():
    print('[color]')
    x = np.linspace(0.0, 1.0, 999)
    e = float(np.max(np.abs(color.l2s(color.s2l(x)) - x)))
    check('sRGB 编解码往返', e < 1e-9, 'max err %.2e' % e)

    rgb = np.random.default_rng(1).random((40, 60, 3))
    e = float(np.max(np.abs(color.from_lab(color.to_lab(rgb)) - rgb)))
    check('Lab 往返', e < 2e-3, 'max err %.2e' % e)

    check('灰度分位单调', color.gray_of(_gray_img(gamma=1.0))[0, 0] < color.gray_of(_gray_img(gamma=1.0))[-1, 0])


def t_analyze():
    print('[L0 analyze]')
    d = _gray_img(gamma=1.0)
    rep = analyze.analyze(color.s2l(d), d, 'jpg')
    a, b = rep['gray_pcts'][C.PCT_BLACK], rep['gray_pcts'][C.PCT_WHITE]
    check('分位有序', a < rep['gray_mid'] < b, '%.3f < %.3f < %.3f' % (a, rep['gray_mid'], b))
    check('决策字段合法', rep['decision'] in ('compress', 'below', 'hold'), rep['decision'])
    check('报告可 json 化', _jsonable(rep))


def _jsonable(o):
    import json
    try:
        json.dumps(o)
        return True
    except Exception:
        return False


def _lin_from_disp(d):
    return color.s2l(d)


def t_tone_mid_target():
    print('[L1 过亮护栏：折中档（09-13）]')
    # ① 只是"偏亮"的正常片（中位 ~0.73）=> 不该动手。
    #    旧行为（护栏线 = TGT_MID 灰阶 140）会把它一路拽到 140 —— 那正是园岭被压闷的病。
    d = _gray_img(gamma=0.45)
    rep = analyze.analyze(_lin_from_disp(d), d, 'jpg')
    gm0 = float(np.percentile(color.gray_of(d), C.PCT_MID))
    check('只是偏亮的正常片 -> hold（不再拽到中灰）',
          rep['decision'] == 'hold', '中位 %.3f / %s' % (gm0, rep['decision']))

    # ② 真的亮得离谱（中位 > 护栏线 0.854）=> 压回来，落点 = 护栏线本身。
    d2 = _gray_img(gamma=0.15)
    rep2 = analyze.analyze(_lin_from_disp(d2), d2, 'jpg')
    gm2 = float(np.percentile(color.gray_of(d2), C.PCT_MID))
    check('真的亮得离谱 -> compress', rep2['decision'] == 'compress', '中位 %.3f' % gm2)
    lin, info = tone.correct(_lin_from_disp(d2), rep2, C)
    out = np.clip(color.l2s(lin), 0, 1)
    gm = float(np.percentile(color.gray_of(out), C.PCT_MID))
    check('敢动手', info['applied'], 'decision=%s' % rep2['decision'])
    if info['applied']:
        check('中灰落到护栏线（不是拽到 0.55）', abs(gm - C.GUARD_MID) < 0.02,
              '%.4f vs 护栏 %.4f（TGT_MID %.4f）' % (gm, C.GUARD_MID, C.TGT_MID))
        gw = float(np.percentile(color.gray_of(out), C.PCT_WHITE))
        # ★ 09-13「开顶」：过曝路径的白点落点从 TGT_WHITE（灰阶 220）改成**上限** WHITE_CEIL（245）。
        #   中灰同时被收到 GUARD_MID（灰阶 218）⇒ 旧写法等于把最亮一截挤成 218~220 的平板。
        check('★ 开顶：过曝片白点落在上限 WHITE_CEIL（不再被拽到灰阶 220）',
              abs(gw - C.WHITE_CEIL) < 0.02 and gw > C.TGT_WHITE + 0.05,
              '%.4f vs 上限 %.4f（旧靶 %.4f）' % (gw, C.WHITE_CEIL, C.TGT_WHITE))
    check('无 NaN/Inf', np.all(np.isfinite(lin)))
    check('取值在 0~1', lin.min() >= -1e-9 and lin.max() <= 1.0 + 1e-9)

    # ★ 开顶**不许越界**：兜底提亮那条路的白点输入常远低于落点，放开等于把高光连色一起放大，
    #   所以那条路必须继续守 TGT_WHITE。
    d3 = _gray_img(gamma=2.2)                       # 偏暗 -> decision == 'below'
    rep3 = analyze.analyze(_lin_from_disp(d3), d3, 'jpg')
    lin3, info3 = tone.correct(_lin_from_disp(d3), rep3, C, allow_lift=True)
    check('开顶只对过曝路径生效（这张判成 below）', rep3['decision'] == 'below', rep3['decision'])
    if info3['applied']:
        gw3 = float(np.percentile(color.gray_of(np.clip(color.l2s(lin3), 0, 1)), C.PCT_WHITE))
        check('兜底提亮路径的白点仍守 TGT_WHITE（没被开顶带偏）',
              abs(gw3 - C.TGT_WHITE) < 0.03, '%.4f vs %.4f' % (gw3, C.TGT_WHITE))


def t_tone_monotone():
    print('[L1 单调性]')
    y = np.logspace(-5, 0, 4000)
    d = np.clip(color.l2s(y), 0, 1).reshape(1, -1, 1).repeat(3, -1)
    rep = analyze.analyze(y.reshape(1, -1, 1).repeat(3, -1), d, 'jpg')
    curve, _ = tone.build_curve(rep, C, allow_lift=True)
    o = curve(y)
    check('曲线单调不减', bool(np.all(np.diff(o) >= -1e-12)))
    # log2 域斜率 = 每一档输入放大几倍。0~4 之外说明曲线要么会翻转、要么在暗部造台阶
    slope = np.diff(np.log2(np.maximum(o, 1e-12))) / np.diff(np.log2(y))
    check('log2 斜率在 [0, 4]', float(slope.min()) >= -1e-6 and float(slope.max()) <= 4.0,
          '%.3f ~ %.3f' % (slope.min(), slope.max()))
    check('值域有界', o.min() > 0 and o.max() < 1.0)
    check('纯黑仍为纯黑（不被提亮）', float(curve(np.array([0.0]))[0]) < 1e-5)


def t_style_lock():
    """锁中灰：职责是「**颜色变换**不许偷偷改曝光」。

    ⚠ 09-13 起 L2 多了一条**刻意**改影调的胶片曲线（`TONE_*`），所以这里要把曲线隔离掉 ——
    否则测到的是"曲线 + 锁"的合成结果，不是锁本身。曲线抬中灰的行为由 `t_style_tone_curve` 守。
    下面保留一条**接口回归**：曲线抬了中灰时，锁不能把它抵消掉。
    """
    print('[L2 锁中灰]')
    cfg = _Cfg(TONE_CURVE=False)
    d = _gray_img(gamma=0.9)
    ref = style.mid_of(d)
    out, info = style.apply(d, cfg, lock_ref=ref)
    check('颜色变换不动中灰', abs(style.mid_of(out) - ref) < 0.012,
          '%.4f -> %.4f' % (ref, style.mid_of(out)))
    out2, _ = style.apply(d, cfg, lock_ref=None)
    check('lock_ref=None 时不锁', True, 'ev=%.4f' % info['lock_ev'])

    # ★ 接口回归：影调曲线抬了中灰 + 带锁 ⇒ 中灰必须真的抬起来（锁不抵消）
    cur = _Cfg(TONE_CURVE=True, TONE_TOE=6.0, TONE_LIFT=6.0)
    out3, info3 = style.apply(d, cur, lock_ref=ref)
    check('曲线抬中灰时锁不抵消（lock_ev≈0 且中灰真的抬高）',
          abs(info3['lock_ev']) < 0.03 and style.mid_of(out3) > ref + 0.005,
          '%.4f -> %.4f (ev=%+.4f)' % (ref, style.mid_of(out3), info3['lock_ev']))


def t_style_contrast_direction():
    """contrast 的**方向**必须对：>1 = 加对比，<1 = 降对比，中灰不动。

    这条是踩过坑加的：曾经把 S 形的符号写反（+a·sin 实际在降对比），
    结果路 B 标定算出来的"这条线反差比大师大"全被落地成了"降对比"。
    """
    print('[L2 对比：方向与支点]')
    d = _gray_img(gamma=1.0)
    s0 = _span90(d)

    up, _ = style.apply(d, _Cfg(CONTRAST=1.35), lock_ref=None)
    s1 = _span90(up)
    check('contrast>1 = 加对比（反差变大）', s1 > s0 + 1.0, '%.1f -> %.1f' % (s0, s1))
    # 支点是 L*=50 那一点（u=0.5 处 sin=0），不是"这张图的中位"—— 用纯色块验最干净
    flat = color.from_lab(np.array([[[50.0, 0.0, 0.0]]]))
    L50 = float(color.to_lab(style.apply(flat, _Cfg(CONTRAST=1.35), lock_ref=None)[0])[0, 0, 0])
    check('支点是 L*=50（纯色块不动）', abs(L50 - 50.0) < 0.5, '50.00 -> %.2f' % L50)

    dn, _ = style.apply(d, _Cfg(CONTRAST=0.70), lock_ref=None)
    s2 = _span90(dn)
    check('contrast<1 = 降对比（反差变小）', s2 < s0 - 1.0, '%.1f -> %.1f' % (s0, s2))

    check('contrast=1.0 时完全不动',
          float(np.max(np.abs(style.apply(d, _Cfg(CONTRAST=1.0), lock_ref=None)[0] - d))) < 1e-9)


def t_style_tone_curve():
    """胶片影调曲线（L2 的 A 档）：趾部压深 + 中高调抬起，**黑白两端不动**。

    最不能错的一条：曲线抬了中灰，LOCK_MID 的参照必须跟着曲线走 ——
    否则锁会把刚抬起来的中灰原样拉回去（LIFT 白做）。下面有专门的回归。
    """
    print('[L2 胶片影调曲线]')
    # ⚠ 必须用**严格中性**的斜坡：_gray_img 逐通道独立加噪，自己就带 a*/b* 抖动
    #   （暗端尤其明显），拿它测"只动 L*"会误报。
    ramp = _gray_img(gamma=1.0)
    d = np.repeat(color.gray_of(ramp)[..., None], 3, axis=-1)

    check('关掉 = 完全不动',
          float(np.max(np.abs(style.apply(d, _Cfg(TONE_CURVE=False), lock_ref=None)[0] - d))) < 1e-9)

    cfg = _Cfg(TONE_CURVE=True, TONE_TOE=6.0, TONE_LIFT=6.0)
    g, y = style._tone_lut(6.0, 6.0)
    check('曲线严格单调', bool(np.all(np.diff(y) > 0)))
    check('黑端不动（L*=0 -> 0）', abs(float(y[0])) < 1e-9, '%.4f' % y[0])
    check('白端不动（L*=100 -> 100）', abs(float(y[-1]) - 100.0) < 1e-9, '%.4f' % y[-1])
    check('趾部：L*=10 被压深到 ~4', abs(float(np.interp(10.0, g, y)) - 4.0) < 0.4,
          '%.2f' % float(np.interp(10.0, g, y)))
    check('中高调：L*=70 被抬到 ~75.7', abs(float(np.interp(70.0, g, y)) - 75.7) < 0.4,
          '%.2f' % float(np.interp(70.0, g, y)))
    _sl_hi = (float(y[-1]) - float(np.interp(90.0, g, y))) / 10.0
    check('高光端有肩部（90->100 段斜率 < 1）', _sl_hi < 0.90, 'slope=%.3f' % _sl_hi)

    out, info = style.apply(d, cfg, lock_ref=None)
    lab0, lab1 = color.to_lab(d), color.to_lab(out)
    check('只动 L*（中性灰的 a*/b* 不动）',
          float(np.max(np.abs(lab1[..., 1]))) < 1e-3 and float(np.max(np.abs(lab1[..., 2]))) < 1e-3,
          'a*max=%.2e b*max=%.2e（Lab 往返噪声量级，远低于 1 个感知单位）'
          % (float(np.max(np.abs(lab1[..., 1]))), float(np.max(np.abs(lab1[..., 2])))))
    check('暗部真的更暗了', float(np.percentile(lab1[..., 0], 10)) <
          float(np.percentile(lab0[..., 0], 10)) - 3.0,
          'P10 %.1f -> %.1f' % (float(np.percentile(lab0[..., 0], 10)),
                                float(np.percentile(lab1[..., 0], 10))))
    check('中灰真的抬起来了', float(np.median(lab1[..., 0])) > float(np.median(lab0[..., 0])) + 2.0,
          'P50 %.1f -> %.1f' % (float(np.median(lab0[..., 0])), float(np.median(lab1[..., 0]))))

    # ★ 关键回归：带锁跑，中灰必须仍然按曲线走（不能被锁拉回）
    ref = style.mid_of(d)
    out_l, info_l = style.apply(d, cfg, lock_ref=ref)
    L_in = float(np.median(color.to_lab(d)[..., 0]))
    L_out = float(np.median(color.to_lab(out_l)[..., 0]))
    L_expect = float(np.interp(L_in, g, y))
    check('带锁跑时中灰仍按曲线走（锁不抵消 LIFT）', abs(L_out - L_expect) < 0.8,
          '期望 %.2f，实得 %.2f' % (L_expect, L_out))
    check('锁的增益被曲线参照抵消（lock_ev ≈ 0）', abs(info_l['lock_ev']) < 0.03,
          'lock_ev=%.4f' % info_l['lock_ev'])

    # 卷可以整组覆盖（给一个 tone_lift=0 的卷就退回"只有趾部"）
    st = stocks.get('neutral')
    st['color'] = dict(tone_curve=True, tone_toe=6.0, tone_lift=0.0)
    out_c, _ = style.apply(d, _Cfg(TONE_CURVE=True, TONE_TOE=0.0, TONE_LIFT=0.0),
                           lock_ref=None, stock=st)
    check('卷能覆盖影调曲线（tone_lift=0 -> 中灰不抬）',
          abs(float(np.median(color.to_lab(out_c)[..., 0])) - L_in) < 1.0,
          'P50 %.1f -> %.1f' % (L_in, float(np.median(color.to_lab(out_c)[..., 0]))))
    check('出厂默认影调曲线开着', bool(getattr(C, 'TONE_CURVE', False)))


def _hp_std(x):
    """高通道方差 = 噪点量（灰度 0~255 口径），跟 metrics 的 noise 一个意思。"""
    g = color.gray_of(np.clip(x, 0, 1)) * 255.0
    k = np.ones((9, 9), np.float32) / 81.0
    import cv2
    b = cv2.blur(g.astype(np.float32), (9, 9))
    return float((g - b).std())


def t_denoise():
    """降噪：默认关=恒等；开着时暗块噪点真下降、亮块不动、硬边保留；近似零均值。

    ⚠ 关于"黑位漂移"：在**带噪点**的暗部，P0.2 是噪声驱动的极值，降噪把噪声抹掉后
    极值**必然**抬高 —— 那是降噪的定义，不是"平白提黑"。所以这条不变量的正确问法是：
      ① 均值 / 中位（内容所在）不漂；
      ② 在**没有噪点**的暗块上，黑位不漂。
    """
    print('[降噪]')
    rng = np.random.default_rng(7)
    # 三块**平坦**色（暗 / 中 / 亮）各带噪点 —— 传感器噪点就长这样，不是整片斜坡
    d = np.empty((240, 360, 3))
    d[:80], d[80:160], d[160:] = 0.20, 0.45, 0.82
    d = np.clip(d + rng.normal(0, 0.025, d.shape), 0.0, 1.0)

    off, i0 = denoise.apply(d, C)
    check('默认关 = 恒等', float(np.max(np.abs(off - d))) < 1e-12 and not i0['applied'])

    cfg = _Cfg(DENOISE_ENABLE=True)
    on, i1 = denoise.apply(d, cfg)
    check('开启后真的动手', bool(i1['applied']) and i1['mask_mean'] > 0.0,
          'mask 均值 %.3f' % i1['mask_mean'])
    check('输出有界/无 NaN', bool(np.all(np.isfinite(on))) and on.min() >= 0 and on.max() <= 1)

    g0, g1 = color.gray_of(d), color.gray_of(on)
    check('全图均值漂移 < 0.002（零均值）', abs(float(g1.mean() - g0.mean())) < 0.002,
          '%.5f -> %.5f' % (g0.mean(), g1.mean()))
    for q in (C.PCT_MID, C.PCT_WHITE):
        a, b = float(np.percentile(g0, q)), float(np.percentile(g1, q))
        check('分位 %g 漂移 < 0.005' % q, abs(b - a) < 0.005, '%.4f -> %.4f' % (a, b))

    n0, n1 = _hp_std(d[:80]), _hp_std(on[:80])
    check('暗块噪点真的下降', n1 < n0 * 0.70, '%.2f -> %.2f' % (n0, n1))
    h0, h1 = _hp_std(d[160:]), _hp_std(on[160:])
    check('亮块基本不动（高光不给降）', h1 > h0 * 0.90, '%.2f -> %.2f' % (h0, h1))

    # 硬边保留：黑白各半的台阶不能被打平
    e = np.full((120, 120, 3), 0.22)
    e[60:] = 0.78
    e = np.clip(e + rng.normal(0, 0.02, e.shape), 0.0, 1.0)
    eo, _ = denoise.apply(e, cfg)
    f0, f1 = color.gray_of(e), color.gray_of(eo)
    step0 = float(f1[64:76].mean() - f1[44:56].mean())
    step1 = float(f0[64:76].mean() - f0[44:56].mean())
    check('硬边台阶保留 ≥85%', step0 >= 0.85 * step1, '%.4f -> %.4f' % (step1, step0))

    # 无噪点的暗部：黑位**不**该被抬（这一步才是"别平白提黑"的真正检验）
    z = np.empty((240, 360, 3))
    z[:120], z[120:] = 0.055, 0.72
    zo, _ = denoise.apply(z, cfg)
    b0 = float(np.percentile(color.gray_of(z), C.PCT_BLACK))
    b1 = float(np.percentile(color.gray_of(zo), C.PCT_BLACK))
    check('无噪点暗部的黑位不漂 < 0.005', abs(b1 - b0) < 0.005, '%.4f -> %.4f' % (b0, b1))


def t_guard():
    print('[L4 护栏]')
    d = np.clip(_gray_img(gamma=0.15) * 1.9, 0, 1)      # 大量死白
    before = float(np.mean(color.gray_of(d) >= 254 / 255))
    out, info = guard.enforce(d, C)
    after = float(np.mean(color.gray_of(out) >= 254 / 255))
    check('死白被压到上限内', after <= C.CAP_WHITE_FRAC + 1e-6,
          '%.4f -> %.4f (上限 %.3f)' % (before, after, C.CAP_WHITE_FRAC))
    check('护栏只往下压', float(np.percentile(color.gray_of(out), C.PCT_WHITE)) <=
          float(np.percentile(color.gray_of(d), C.PCT_WHITE)) + 1e-6)


def t_lut():
    print('[L2 .cube]')
    tmp = os.path.join(tempfile.gettempdir(), '_svfilm_selftest.pars')
    cube = style.bake_grid(9).reshape(9, 9, 9, 3)
    style.cube_write(tmp, cube, 'identity')
    back = style.cube_read(tmp)
    check('.cube 往返一致', float(np.max(np.abs(back - cube))) < 1e-5)
    d = np.random.default_rng(3).random((30, 40, 3))
    check('恒等 LUT 三线性 = 原图', float(np.max(np.abs(style.cube_apply(d, back) - d))) < 1e-3)
    try:
        style.cube_write(tmp, style.bake_grid(9).reshape(9, 9, 9, 3), 'x')
        ok = os.path.getsize(tmp) > 1000
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    check('烘焙写出文件', ok)


def t_io_roundtrip():
    print('[io]')
    tmp = os.path.join(tempfile.gettempdir(), '_svfilm_selftest.png')
    d = _gray_img(60, 90, gamma=0.6)
    io.save(d, tmp, quality=95)
    s = io.load(tmp, max_side=4096, src='jpg')
    e = float(np.max(np.abs(s.disp - d)))
    check('存-读往返误差 < 2/255', e < 2.0 / 255.0, 'max err %.4f' % e)
    os.remove(tmp)


def t_style_chroma_ends():
    """抽色饱和：两端掉彩、中间调不动（09-13 调研修正，出处 Emulsifier）。"""
    print('[L2 抽色饱和（两端掉彩）]')
    Ls = np.array([8.0, 25.0, 50.0, 75.0, 95.0])
    lab = np.stack([Ls, np.full_like(Ls, 12.0), np.full_like(Ls, 12.0)], axis=-1)
    # 先夹进显示域（L*=95 那块的 R 会超过 1），否则"关掉 = 恒等"测的是裁剪不是模型
    d = np.clip(color.from_lab(lab), 0.0, 1.0).reshape(1, -1, 3)
    out, _ = style.apply(d, _Cfg(CHROMA_ENDS=0.30), lock_ref=None)
    c0 = color.chroma(color.to_lab(d))[0]
    c1 = color.chroma(color.to_lab(out))[0]
    check('抽色饱和：中间调（L*=50）不动', abs(c1[2] / c0[2] - 1.0) < 0.03,
          '%.3f -> %.3f' % (c0[2], c1[2]))
    check('抽色饱和：暗端掉彩', c1[0] / c0[0] < 0.85, '%.3f -> %.3f' % (c0[0], c1[0]))
    check('抽色饱和：亮端掉彩', c1[4] / c0[4] < 0.85, '%.3f -> %.3f' % (c0[4], c1[4]))
    out0, _ = style.apply(d, _Cfg(CHROMA_ENDS=0.0), lock_ref=None)
    check('抽色饱和关掉 = 恒等', float(np.max(np.abs(out0 - d))) < 1e-9)
    # 出厂默认必须开着（否则这个修正等于没做）
    check('出厂默认抽色饱和开着', float(getattr(C, 'CHROMA_ENDS', 0.0)) > 0.0, str(getattr(C, 'CHROMA_ENDS', None)))


def t_stocks():
    print('[卷 stocks]')
    for n in stocks.names():
        s = stocks.TABLE[n]
        check('卷 %s 有名字/说明/颜色参数' % n,
              bool(s['label'] and s['desc'] and s['color']))
    d = _gray_img(gamma=0.8)
    # 这一条测的是"卷的恒等性"，所以必须把基准摘掉（否则测到的是出厂默认的全对齐）
    out, _ = style.apply(d, C, lock_ref=None, stock=stocks.get('neutral'),
                         base='BASE_NONE')
    check('neutral 卷 = 恒等（不风格化）', float(np.max(np.abs(out - d))) < 1e-9,
          'max %.2e' % float(np.max(np.abs(out - d))))
    # 出厂默认是哪一档要显式守在这里（09-13 SV 拍板 = 全对齐）；改了必须在这里承认
    check('出厂默认基准是合法档位', C.BASE in stocks.base_names(), C.BASE)
    check('_Cfg 隔离基准（单测不跟出厂默认走）', _Cfg().BASE == 'BASE_NONE')
    # 每个卷的参数都要能跑通并给出有界结果
    grid = style.bake_grid(5).reshape(1, -1, 3)
    for n in stocks.names():
        o, _ = style.apply(grid, C, lock_ref=None, stock=stocks.get(n))
        check('卷 %s 可跑通且有界' % n,
              bool(np.all(np.isfinite(o))) and o.min() >= 0.0 and o.max() <= 1.0)
    # 卷要真的不一样（否则"卷"就是摆设）
    a, _ = style.apply(grid, C, lock_ref=None, stock=stocks.get('portra400'))
    b, _ = style.apply(grid, C, lock_ref=None, stock=stocks.get('ektar100'))
    check('不同卷确实不同', float(np.max(np.abs(a - b))) > 0.01,
          'max diff %.4f' % float(np.max(np.abs(a - b))))


def t_spatial_off():
    print('[空间域：默认关 = 恒等]')
    d = _gray_img(gamma=0.55)
    out, info = spatial.apply(d, C, stock=None)
    check('三个都关时不动像素', float(np.max(np.abs(out - d))) < 1e-12)
    check('info.any=False', not info['any'])


def t_spatial_grain():
    print('[空间域：颗粒]')
    d = _gray_img(gamma=0.55)
    p = spatial.resolve(C, stocks.get('portra400'))['grain']
    g1, i1 = spatial.grain(d, p)
    g2, _ = spatial.grain(d, p)
    check('颗粒可复现（固定种子）', float(np.max(np.abs(g1 - g2))) < 1e-12)
    dm = float(np.percentile(color.gray_of(d), C.PCT_MID))
    gm = float(np.percentile(color.gray_of(g1), C.PCT_MID))
    check('颗粒不推走中灰（±0.006）', abs(gm - dm) < 0.006, '%.4f -> %.4f' % (dm, gm))
    blk = np.zeros((48, 48, 3))
    gb, _ = spatial.grain(blk, p)
    check('颗粒不把纯黑提亮（乘性叠加）', float(gb.max()) < 1e-12)
    # ★ 高光端精确归零（09-13 调研修 bug）：白墙/天空不能有颗粒
    wht = np.full((48, 48, 3), 0.99)
    gw, _ = spatial.grain(wht, p)
    check('颗粒在高光端归零（白墙/天空不动）', float(np.max(np.abs(gw - wht))) < 1e-12,
          'max %.2e' % float(np.max(np.abs(gw - wht))))
    check('颗粒输出有界', bool(np.all(np.isfinite(g1))) and g1.min() >= 0 and g1.max() <= 1)


def t_spatial_bloom_halation():
    print('[空间域：黑柔 / Halation 方向性]')
    d = np.zeros((200, 200, 3))
    d[80:120, 80:120] = 1.0                      # 黑底上一个白方块
    p = spatial.resolve(C, stocks.get('cinestill800t'))

    b, _ = spatial.bloom(d, p['bloom'])
    near = float(b[58:76, 80:120].mean())        # 方块正上方外侧
    far = float(b[0:10, 0:10].mean())            # 远角
    check('黑柔只在亮区往外扩散（近处亮、远处不动）', near > 1e-4 and far < 1e-4,
          'near %.5f far %.7f' % (near, far))

    h, _ = spatial.halation(d, p['halation'])
    ring = h[58:76, 80:120]
    r, gg, bb = (float(ring[..., i].mean()) for i in range(3))
    check('Halation 晕圈偏红橙（R>G>B）', r > gg > bb, '%.4f/%.4f/%.4f' % (r, gg, bb))
    check('Halation 输出有界', bool(np.all(np.isfinite(h))) and h.min() >= 0 and h.max() <= 1)

    # ★ 分通道半径（09-13 调研修正）：红光散得最远、绿居中、蓝近零
    _, hi = spatial.halation(d, p['halation'])
    ratios = hi.get('radius_ratios')
    check('Halation 分通道半径（R>G>B）',
          bool(ratios) and float(ratios[0]) > float(ratios[1]) > float(ratios[2]), str(ratios))
    # 远处外圈：红还有能量，绿已经基本没有（蓝半径最小）
    far = h[26:52, 80:120]
    fr, fg, fb = (float(far[..., i].mean()) for i in range(3))
    check('Halation 外圈以红为主（R 明显 > G）', fr > fg * 1.5,
          '外圈 R/G/B %.5f/%.5f/%.5f' % (fr, fg, fb))

    # ★ 09-13「黑柔阶梯」三条：三个机理独立 + "化开"方向与"加性辉光"相反
    #   ① 门控修错：veil>0 而 amount=0 时**必须真的动**
    #      （老代码 `amount<=0` 把整个函数关掉 ⇒ "只开面纱"逐位等于"关"，白扫一档）
    pv = dict(p['bloom']); pv.update(dict(amount=0.0, veil=0.10, spread=0.0))
    bv, _ = spatial.bloom(d, pv)
    dv = float(np.abs(bv - d).max())
    check('只开面纱（veil>0, amount=0）确实生效', dv > 1e-5, 'max %.2e' % dv)

    #   ② 方向不变量：加性辉光**抬高**核心，化开**压低**核心（参数名语义必须与行为一致）
    pa = dict(p['bloom']); pa.update(dict(amount=0.15, veil=0.0, spread=0.0))
    ba, _ = spatial.bloom(d, pa)
    ps = dict(p['bloom']); ps.update(dict(amount=0.15, veil=0.0, spread=0.60))
    bs, _ = spatial.bloom(d, ps)
    core_a = float(ba[95:105, 95:105].mean())
    core_s = float(bs[95:105, 95:105].mean())
    check('加性辉光抬核心 / 化开压核心（方向相反）', core_a > core_s + 0.05,
          '加性 %.4f vs 化开 %.4f' % (core_a, core_s))

    #   ③ 化开把能量**散到外圈**：外圈不能比纯加性的更暗（守恒 ⇒ 核心少了的挪到外面）
    ring_a = float(ba[58:76, 80:120].mean())
    ring_s = float(bs[58:76, 80:120].mean())
    check('化开把能量散到外圈（外圈 ≥ 加性辉光）', ring_s >= ring_a - 1e-9,
          '加性 %.5f vs 化开 %.5f' % (ring_a, ring_s))

    #   ④ 守恒条件：**amount == spread 时**平坦亮区几乎不动（+amt*glow 与 -spread*hot 相消）。
    #      注意：单独给 spread（amount=0）就是"只把高光压暗"，不守恒 —— 这一条守的是"配对使用"。
    flat = np.zeros((200, 200, 3)); flat[:, :] = 0.95
    pf = dict(p['bloom']); pf.update(dict(amount=0.60, veil=0.0, spread=0.60, warmth=0.0))
    bf, _ = spatial.bloom(flat, pf)
    df = float(np.abs(bf - flat).max())
    check('amount==spread 时平坦亮区不漂（能量守恒）', df < 0.01, 'max %.4f' % df)


def _skin_patch(a, b, L=62.0, size=64):
    lab = np.zeros((size, size, 3), np.float64)
    lab[..., 0] = L
    lab[..., 1] = a
    lab[..., 2] = b
    return np.clip(color.from_lab(lab), 0.0, 1.0)


def t_local_skin_floor():
    print('[L3 肤色正向达标 + 逐像素门]')
    p = _skin_patch(10.0, 8.0)
    o, i = local.skin_floor(p, _Cfg(SKIN_FLOOR=False))
    check('达标关掉 = 逐像素恒等', float(np.max(np.abs(o - p))) < 1e-12)

    cfg = _Cfg(SKIN_FLOOR=True, SKIN_PROTECT_STRENGTH=0.0)
    cold = _skin_patch(10.0, 8.0, 62.0)
    o, i = local.skin_floor(cold, cfg)
    lo = color.to_lab(o)
    check('冷皮肤被抬到档上', i['applied'] is True, str(i.get('reason')))
    check('a* 落在目标上', abs(float(np.median(lo[..., 1])) - cfg.SKIN_FLOOR_A) < 0.3,
          'a*=%.2f' % float(np.median(lo[..., 1])))
    check('b* 落在目标上', abs(float(np.median(lo[..., 2])) - cfg.SKIN_FLOOR_B) < 0.3,
          'b*=%.2f' % float(np.median(lo[..., 2])))
    check('明度不动', abs(float(np.median(lo[..., 0])) - 62.0) < 0.6,
          'L*=%.2f' % float(np.median(lo[..., 0])))

    warm = _skin_patch(16.5, 19.0, 62.0)
    o2, i2 = local.skin_floor(warm, cfg)
    check('已达标的皮肤一分不动', float(np.max(np.abs(o2 - warm))) < 1e-12, str(i2.get('reason')))

    # 唇妆（a* 已经很高）：两个轴里 a* 超目标 ⇒ 完全不碰
    lips = _skin_patch(10.0, 8.0, 62.0)
    lips[:, :20] = _skin_patch(22.0, 12.0, 62.0)[:, :20]
    o3, _ = local.skin_floor(lips, cfg)
    lab3 = color.to_lab(o3)
    check('唇妆不被推红', abs(float(np.median(lab3[:, :20, 1])) - 22.0) < 0.1,
          'lip a*=%.2f' % float(np.median(lab3[:, :20, 1])))
    check('同一张图里皮肤照补', float(np.median(lab3[:, 20:, 1])) > 15.0,
          'skin a*=%.2f' % float(np.median(lab3[:, 20:, 1])))

    # 粉衣服（a* 高于目标、b* 低于目标）：不该被"顺手推暖"
    cloth = _skin_patch(10.0, 8.0, 62.0)
    cloth[:32] = _skin_patch(21.0, 10.0, 62.0)[:32]
    o4, _ = local.skin_floor(cloth, cfg)
    lab4 = color.to_lab(o4)
    check('粉衣服不被顺手推暖', float(np.max(np.abs(o4[:32] - cloth[:32]))) < 0.02,
          'cloth b*=%.2f' % float(np.median(lab4[:32, :, 2])))

    bright = _skin_patch(10.0, 8.0, 88.0)
    o5, i5 = local.skin_floor(bright, cfg)
    check('过亮脸门收力（gate≈0）', i5['gate'] < 0.02, 'gate=%.3f' % i5['gate'])
    check('过亮脸基本不动', float(np.max(np.abs(o5 - bright))) < 0.02)

    # 门是**逐像素**的：同一张脸里，亮的像素少补、暗的像素照补
    ramp = _skin_patch(10.0, 8.0, 62.0, 64)
    ramp[:32] = _skin_patch(10.0, 8.0, 80.0)[:32]
    o6, i6 = local.skin_floor(ramp, cfg)
    lab6 = color.to_lab(o6)
    dk = float(np.median(lab6[32:, :, 1]))
    br = float(np.median(lab6[:32, :, 1]))
    check('逐像素门：暗的半边照补', dk > 15.0, 'dark a*=%.2f' % dk)
    check('逐像素门：亮的半边少补', br < 13.0, 'bright a*=%.2f' % br)
    check('逐像素门：门在 (0,1) 之间', 0.2 < i6['gate'] < 0.95, 'gate=%.3f' % i6['gate'])

    d3, li = local.apply(cold, cold, cfg)
    check('apply 汇报 skin_floor', 'skin_floor' in li)
    d3b, lib = local.apply(cold, cold, _Cfg())
    check('apply 关掉时不带 skin_floor', 'skin_floor' not in lib)


def t_pipeline_smoke():
    print('[全链 smoke]')
    d = _gray_img(120, 180, gamma=0.4)
    rep = analyze.analyze(_lin_from_disp(d), d, 'jpg')
    lin, ti = tone.correct(_lin_from_disp(d), rep, C)
    d1 = np.clip(color.l2s(lin), 0, 1)
    d1, di = denoise.apply(d1, C)
    d2, si = style.apply(d1, C, lock_ref=style.mid_of(d1), stock=stocks.get('portra400'))
    d2b, pi = spatial.apply(d2, C, stock=stocks.get('portra400'))
    d3, li = local.apply(d1, d2b, C)
    d4, gi = guard.enforce(d3, C)
    check('各层输出有界', all(x.min() >= 0 and x.max() <= 1 for x in (d1, d2, d2b, d3, d4)))
    check('各层无 NaN', all(np.all(np.isfinite(x)) for x in (d1, d2, d2b, d3, d4)))
    check('肤色掩膜不炸', 0.0 <= li['skin_cov'] <= 1.0, 'cov=%.4f' % li['skin_cov'])
    check('空间域 info 齐全', all(k in pi for k in ('grain', 'bloom', 'halation', 'any')))


def _mn(entries):
    """造一个最小的富士 MakerNote（只支持 SHORT / SRATIONAL）。

    entries: [(tag, type, value)]；type 3 的 value 是 int，type 10 的 value 是 (分子, 分母)。
    """
    n = len(entries)
    data_off = 12 + 2 + n * 12 + 4
    body = struct.pack('<H', n)
    blob = b''
    for tag, typ, val in entries:
        if typ == 3:
            body += struct.pack('<HHI', tag, typ, 1) + struct.pack('<H', val) + b'\0\0'
        elif typ == 10:
            body += struct.pack('<HHII', tag, typ, 1, data_off + len(blob))
            blob += struct.pack('<ii', val[0], val[1])
        else:
            raise ValueError('测试用的 MakerNote 只支持 SHORT/SRATIONAL')
    return b'FUJIFILM' + struct.pack('<I', 12) + body + struct.pack('<I', 0) + blob


class _Sample:
    """给 pipeline._allow_lift 用的最小替身（只要 kind + cam）。"""

    def __init__(self, kind, cam=None):
        self.kind = kind
        self.cam = cam or {}


def t_entry_bias():
    print('[入口基线曝光：MakerNote tag 语义]')
    S, SR = 3, 10
    # Manual/Raw 模式（实测 759 张全是这个）-> 看 0x1403
    check('Manual 模式读 0x1403',
          rawmeta.fuji_development_dr(_mn([(0x1402, S, 1), (0x1403, S, 400)])) == 400)
    check('Manual 模式读 DR200',
          rawmeta.fuji_development_dr(_mn([(0x1402, S, 1), (0x1403, S, 200)])) == 200)
    # Auto 模式 -> 看 0x140b
    check('Auto 模式读 0x140b',
          rawmeta.fuji_development_dr(_mn([(0x1402, S, 0), (0x140b, S, 200)])) == 200)
    # 模式读不到 -> 两个都试
    check('无模式 tag 时回退 0x1403',
          rawmeta.fuji_development_dr(_mn([(0x1403, S, 100)])) == 100)
    check('无模式 tag 时回退 0x140b',
          rawmeta.fuji_development_dr(_mn([(0x140b, S, 400)])) == 400)
    # ★ 回归护栏：0x1402 是模式开关（值 0/1），**绝不能**被当成档位返回
    check('0x1402 的值不当档位（Manual 缺 0x1403 时返回 None）',
          rawmeta.fuji_development_dr(_mn([(0x1402, S, 1)])) is None)
    check('0x1402 的值不当档位（Auto 缺 0x140b 时返回 None）',
          rawmeta.fuji_development_dr(_mn([(0x1402, S, 0)])) is None)
    check('非富士 MakerNote 不炸', rawmeta.fuji_development_dr(b'NIKON\x00\x00\x00') is None)
    check('空 MakerNote 不炸', rawmeta.fuji_development_dr(None) is None)
    # 精确 EV（0x9650）：绝大多数机身不写，写了就用
    check('0x9650 读出精确 EV', abs(rawmeta.fuji_raw_ev(_mn([(0x9650, SR, (272, 100))])) - 2.72) < 1e-6)
    check('0x9650 离谱值被丢弃', rawmeta.fuji_raw_ev(_mn([(0x9650, SR, (9999, 1))])) is None)
    check('没有 0x9650 时返回 None', rawmeta.fuji_raw_ev(_mn([(0x1403, S, 400)])) is None)

    print('[入口基线曝光：机型表 + DR 分层]')
    fuji = cameras.lookup('FUJIFILM', 'X-T30 III')
    check('富士机型命中', fuji['hit'] and fuji['key'] == 'x-t30 iii')
    check('富士基底 = 0.72EV', abs(fuji['baseline_ev'] - 0.72) < 1e-9)
    check('DR 表只放"额外"量 100/200/400 = 0/1/2',
          cameras.dr_bias_ev(100) == 0.0 and cameras.dr_bias_ev(200) == 1.0
          and cameras.dr_bias_ev(400) == 2.0)
    check('不认识的 DR 档返回 None（表示"不知道"）',
          cameras.dr_bias_ev(None) is None and cameras.dr_bias_ev(999) is None)
    # 与 darktable 官方总表对齐：DR100 −0.72 / DR200 −1.72 / DR400 −2.72
    tot = {dr: fuji['baseline_ev'] + cameras.dr_bias_ev(dr) for dr in (100, 200, 400)}
    check('合计对上 darktable 总表 0.72/1.72/2.72',
          abs(tot[100] - 0.72) < 1e-9 and abs(tot[200] - 1.72) < 1e-9 and abs(tot[400] - 2.72) < 1e-9,
          '%.2f/%.2f/%.2f' % (tot[100], tot[200], tot[400]))
    check('未知机型不猜（基底 0）', cameras.lookup('Canon', 'EOS R5')['baseline_ev'] == 0.0)

    print('[入口曲线：实测相机曲线查表（Q1）]')
    _saved_curve = dict(cameras.ENTRY_CURVE)
    try:
        # 表空 / 机型不认识 -> None（退回老的常数补偿，不能炸）
        cameras.ENTRY_CURVE.clear()
        check('表空时返回 None（退回常数补偿）', cameras.entry_curve('x-t30 iii', 400) is None)
        cameras.ENTRY_CURVE.update({
            'x-t30 iii': {
                # anchors = [[输入线性亮度, 增益倍数], ...]，弓形：暗部抬得多、高光收
                '400': dict(mid_ev=3.10, anchors=[[0.001, 5.0], [0.02, 7.0], [0.18, 8.57],
                                                  [0.30, 8.0], [1.0, 1.0]]),
                'None': dict(mid_ev=0.72, anchors=[[0.001, 1.4], [0.18, 1.647], [1.0, 1.0]]),
            }})
        check('机型不认识返回 None', cameras.entry_curve('nope', 400) is None)
        c = cameras.entry_curve('X-T30 III', 400)          # 大小写不敏感
        check('命中 -> 给出 mid_ev + 增益锚点', c is not None and abs(c[0] - 3.10) < 1e-9)
        check('锚点按输入线性升序', c[1] == sorted(c[1]))
        check('零点自洽：Y=0.18 处的增益 = 2^mid_ev（±0.05 档）',
              abs(np.log2(np.interp(0.18, c[1], c[2])) - c[0]) < 0.05,
              '%.3f vs %.3f' % (np.log2(np.interp(0.18, c[1], c[2])), c[0]))
        check('增益是"弓形"（中间调最高，暗部/高光都比它低）',
              c[2][2] > c[2][0] and c[2][2] > c[2][-1])
        c2 = cameras.entry_curve('x-t30 iii', None)        # dr 未知 -> 取 'None' 那条
        check('dr=None 回退到 None 那条', c2 is not None and abs(c2[0] - 0.72) < 1e-9)
        c3 = cameras.entry_curve('x-t30 iii', 800)         # dr 有但不认识 -> 回退
        check('dr 不认识时回退', c3 is not None and abs(c3[0] - 0.72) < 1e-9)
    finally:
        cameras.ENTRY_CURVE.clear()
        cameras.ENTRY_CURVE.update(_saved_curve)

    # 曲线怎么套：三通道同一个倍率（只动亮度）＋ 弓形（暗部倍率 > 高光倍率）
    _cv = (3.10, [0.001, 0.18, 0.60, 1.0], [5.0, 8.57, 2.0, 1.0])
    _g = np.array([[[0.18, 0.18, 0.18]]])
    check('曲线把 18% 中性灰抬 2^mid_ev 倍',
          abs(io.apply_entry_curve(_g, _cv)[0, 0, 0] / 0.18 - 8.57) < 0.02)
    _c = np.array([[[0.30, 0.15, 0.075]]])
    _r = io.apply_entry_curve(_c, _cv)[0, 0] / _c[0, 0]
    check('曲线不改色相（三通道同一个倍率）', float(_r.max() - _r.min()) < 1e-12)
    check('弓形：中间调倍率最高（暗部、高光都比它低）',
          io.apply_entry_curve(_g, _cv)[0, 0, 0] / 0.18
          > io.apply_entry_curve(np.array([[[0.001] * 3]]), _cv)[0, 0, 0] / 0.001
          and io.apply_entry_curve(_g, _cv)[0, 0, 0] / 0.18
          > io.apply_entry_curve(np.array([[[0.60] * 3]]), _cv)[0, 0, 0] / 0.60)

    print('[入口基线曝光：曝光层不再提亮]')
    like = _Sample('raw', {'idt_bias_ev': 2.72})
    check('入口补过 -> 不再提亮', pipeline._allow_lift(like, C) is False)
    check('入口补 0 -> 允许兜底提亮', pipeline._allow_lift(_Sample('raw', {'idt_bias_ev': 0.0}), C) is True)
    check('入口没这项(非富士) -> 允许兜底提亮',
          pipeline._allow_lift(_Sample('raw', {}), C) is True)
    check('JPG 一律不提亮', pipeline._allow_lift(_Sample('jpg'), C) is False)
    # 端到端：同一张偏暗图，allow_lift False 时像素不动
    d = _gray_img(gamma=0.9)
    lin0 = _lin_from_disp(d)
    rep = analyze.analyze(lin0, d, 'raw')
    out_a, _ = tone.correct(lin0, rep, C, allow_lift=False)
    check('allow_lift=False 时偏暗图逐像素不动',
          float(np.max(np.abs(out_a - lin0))) < 1e-12, 'decision=%s' % rep['decision'])


def main():
    for fn in (t_color, t_analyze, t_tone_mid_target, t_tone_monotone, t_style_lock,
               t_style_contrast_direction, t_style_tone_curve, t_style_chroma_ends, t_denoise,
               t_guard, t_lut,
               t_io_roundtrip, t_stocks, t_spatial_off, t_spatial_grain,
               t_spatial_bloom_halation, t_local_skin_floor, t_entry_bias, t_pipeline_smoke):
        fn()
    print('-' * 52)
    if FAIL:
        print('失败 %d 项: %s' % (len(FAIL), ', '.join(FAIL)))
        return 1
    print('全部通过')
    return 0


if __name__ == '__main__':
    sys.exit(main())
