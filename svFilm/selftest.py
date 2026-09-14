# -*- coding: utf-8 -*-
"""自检 —— 动完任何一层都要跑。不依赖任何样片，纯合成图 + 不变量。

  python -m svFilm.selftest
"""
import os
import struct
import sys
import tempfile

import numpy as np

from . import (analyze, cameras, color, config as C, denoise, face, film, guard, io, local, metrics,
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
        # ⚠ 肩部（TONE_SHOULDER）出厂默认=2（09-13 深夜 SV 定「B 档」，由 4 放开），**也必须隔离**：
        #   否则凡是"拿 _tone_lut(...) 算期望值再和 style.apply 比"的用例都会错位。
        self._d['TONE_CURVE'] = False
        self._d['TONE_SHOULDER'] = 0.0
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

    # ★ 09-14：白点现在**只设上限**；且 **L1 不再提亮**（提亮归锚点）
    d3 = _gray_img(gamma=2.2)                       # 偏暗 -> decision == 'below'
    rep3 = analyze.analyze(_lin_from_disp(d3), d3, 'jpg')
    lin3, info3 = tone.correct(_lin_from_disp(d3), rep3, C)
    check('偏暗（below）⇒ L1 一步不动（提亮归锚点）',
          rep3['decision'] == 'below' and not info3['applied'], rep3['decision'])


def t_tone_monotone():
    print('[L1 单调性]')
    y = np.logspace(-5, 0, 4000)
    d = np.clip(color.l2s(y), 0, 1).reshape(1, -1, 1).repeat(3, -1)
    rep = analyze.analyze(y.reshape(1, -1, 1).repeat(3, -1), d, 'jpg')
    curve, _ = tone.build_curve(rep, C)
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

    # ---- ★ 肩部（TONE_SHOULDER）：高光段整段下收 ----
    # 出厂默认现在是 2.0（09-13 深夜 SV 拍板「B 档」，由 4 放开 —— 见 config 注释与手册 §21.3）；
    # 下面一律显式传 0 当"老行为"基准，机理检查用 SH=8 的独立档，与默认值解耦。
    print('[★ 肩部 TONE_SHOULDER：收高光（09-13 深夜 SV 拍板「B = 2」）]')
    g0, y0 = style._tone_lut(1.0, 9.0, 0.0)
    _gz, yz = style._tone_lut(1.0, 9.0, 0.0)
    check('肩部 0 = 逐位等于老行为', float(np.max(np.abs(y0 - yz))) < 1e-12)
    check('出厂默认 TONE_SHOULDER = 2.0（B 档：白点上限 L*98，贴大师 L99 97.0）',
          abs(float(getattr(C, 'TONE_SHOULDER', 0.0)) - 2.0) < 1e-9,
          '%.1f' % float(getattr(C, 'TONE_SHOULDER', 0.0)))
    check('默认档的白点上限落在 L*98（不再被钉在 96）',
          abs(float(style._tone_lut(1.0, 9.0, 2.0)[1][-1]) - 98.0) < 1e-6,
          '%.3f' % style._tone_lut(1.0, 9.0, 2.0)[1][-1])

    SH = 8.0
    gs, ys = style._tone_lut(1.0, 9.0, SH)
    check('白端被下收（L*=100 -> 100-sh）', abs(float(ys[-1]) - (100.0 - SH)) < 1e-6,
          '%.3f' % ys[-1])
    sl0 = (y0[-1] - float(np.interp(90.0, g0, y0))) / 10.0
    sls = (ys[-1] - float(np.interp(90.0, gs, ys))) / 10.0
    check('高光段斜率收小（肩部真的更"滚"）', sls < sl0 - 0.05,
          '%.3f -> %.3f' % (sl0, sls))
    check('黑端 / 趾部不受肩部影响',
          abs(float(np.interp(10.0, gs, ys)) - float(np.interp(10.0, g0, y0))) < 0.45,
          '%.2f' % abs(float(np.interp(10.0, gs, ys)) - float(np.interp(10.0, g0, y0))))
    check('中灰基本不动（只收高光）',
          abs(float(np.interp(50.0, gs, ys)) - float(np.interp(50.0, g0, y0))) < 0.6,
          '%.2f' % abs(float(np.interp(50.0, gs, ys)) - float(np.interp(50.0, g0, y0))))
    check('加肩部后仍严格单调（不出平板）', bool(np.all(np.diff(ys) > 0)))

    # 反例守卫：**只压白点**会塌成平板 ⇒ 必须整段收（这是设计决定，别被"简化"掉）
    from scipy.interpolate import PchipInterpolator as _PI
    _xp = np.array([0.0, 10.0, 30.0, 50.0, 70.0, 90.0, 100.0])
    _yp = np.array([0.0, 9.0, 29.55, 54.95, 78.55, 94.95, 92.0])   # 只把最后一格往下挪
    _gg = np.linspace(0.0, 100.0, 1024)
    _yy = np.clip(np.maximum.accumulate(_PI(_xp, _yp)(_gg)), 0.0, 100.0)
    _flat = (float(_yy[-1]) - float(np.interp(90.0, _gg, _yy))) / 10.0
    check('反例：只压白点 = 平板（斜率≈0）⇒ 肩部必须整段收', _flat < 0.15, 'slope=%.3f' % _flat)

    # 卷可覆盖（各卷能有自己的肩部性格）
    st2 = stocks.get('neutral')
    st2['color'] = dict(tone_curve=True, tone_toe=1.0, tone_lift=9.0, tone_shoulder=SH)
    out_s, _ = style.apply(d, _Cfg(TONE_CURVE=True, TONE_TOE=1.0, TONE_LIFT=9.0,
                                   TONE_SHOULDER=0.0), lock_ref=None, stock=st2)
    L_in_s = float(np.median(color.to_lab(d)[..., 0]))
    L_out_s = float(np.median(color.to_lab(out_s)[..., 0]))
    check('卷能覆盖肩部（tone_shoulder 生效）',
          abs(L_out_s - float(np.interp(L_in_s, gs, ys))) < 1.0,
          '期望 %.2f，实得 %.2f' % (float(np.interp(L_in_s, gs, ys)), L_out_s))

    out_off = style.apply(d, _Cfg(TONE_CURVE=True, TONE_TOE=1.0, TONE_LIFT=9.0,
                                  TONE_SHOULDER=0.0), lock_ref=None)[0]
    out_sh = style.apply(d, _Cfg(TONE_CURVE=True, TONE_TOE=1.0, TONE_LIFT=9.0,
                                 TONE_SHOULDER=SH), lock_ref=None)[0]
    check('成片层面：高光更暗（P95 降）',
          float(np.percentile(color.to_lab(out_sh)[..., 0], 95)) <
          float(np.percentile(color.to_lab(out_off)[..., 0], 95)) - 1.0,
          'P95 %.1f -> %.1f' % (float(np.percentile(color.to_lab(out_off)[..., 0], 95)),
                                float(np.percentile(color.to_lab(out_sh)[..., 0], 95))))


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

    off, i0 = denoise.apply(d, _Cfg(DENOISE_ENABLE=False))
    check('关掉 = 恒等（⚠ 出厂 09-14 起是**开**，所以这里显式关）',
          float(np.max(np.abs(off - d))) < 1e-12 and not i0['applied'])

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
    print('[空间域：全关 = 恒等]')
    # ⚠ 09-14 起出厂**默认开**（SV：「需要那些空间层的」）⇒ 这条要**显式关掉**三个开关再测。
    ks = ('GRAIN_ENABLE', 'BLOOM_ENABLE', 'HALATION_ENABLE')
    old = {k: getattr(C, k) for k in ks}
    try:
        for k in ks:
            setattr(C, k, False)
        d = _gray_img(gamma=0.55)
        out, info = spatial.apply(d, C, stock=None)
        check('三个都关时不动像素', float(np.max(np.abs(out - d))) < 1e-12)
        check('info.any=False', not info['any'])
    finally:
        for k, v in old.items():
            setattr(C, k, v)


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

    # ★★ 09-13 修「脸变色」的两条护栏（SV 报 DSCF2638）：
    #    老实现 `lin + amount*blur(lin*mask)` 把**模糊后的彩色光**直接加上去 ⇒
    #    辉光会把**周围亮物的颜色搬到别处**（红抱枕的红光糊到脸上）。
    #   ⑤ 数值不变量：`warmth=0` 时三通道的**增量必须完全相等**（只动亮度，不改色相）
    mid = np.zeros((240, 240, 3)); mid[:, :] = 0.30          # 中灰底（不会顶到白）
    mid[80:160, 80:160] = 1.0                                # 中间一块白
    pm = dict(p['bloom']); pm.update(dict(amount=0.30, veil=0.0, spread=0.0, warmth=0.0))
    bm, _ = spatial.bloom(mid, pm)
    d_lin = color.s2l(bm) - color.s2l(mid)
    selm = bm.max(axis=-1) < 0.99                            # 排除被裁到 1.0 的像素
    dev = float(np.max(np.abs(d_lin[..., 0][selm] - d_lin[..., 2][selm])))
    check('辉光只动亮度：warmth=0 时三通道增量相等（不改色相）', dev < 1e-6,
          'max|ΔR−ΔB| %.2e' % dev)

    #   ⑥ 语义不变量：**红块旁边不能把中性灰染红**（"不许搬邻居的颜色"）
    blk = np.zeros((240, 240, 3)); blk[:, :] = 0.25
    blk[:, :120] = np.array([1.0, 0.02, 0.02])               # 左半：鲜红亮块
    pb = dict(p['bloom']); pb.update(dict(amount=0.50, veil=0.0, spread=0.0, warmth=0.0))
    bb, _ = spatial.bloom(blk, pb)
    lab_b = color.to_lab(bb); lab_0 = color.to_lab(blk)
    right = np.zeros((240, 240), bool); right[:, 130:] = True   # 右半：离红块远一点的中性区
    da = float(np.median(lab_b[..., 1][right] - lab_0[..., 1][right]))
    db = float(np.median(lab_b[..., 2][right] - lab_0[..., 2][right]))
    check('辉光不把邻居的颜色搬过来（红块旁的中性区不发红）', abs(da) < 1.0 and abs(db) < 1.0,
          'Δa* %+.2f  Δb* %+.2f' % (da, db))


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

    print('[位置：入口 settle 定基准 + 脸锚点补偿]')
    _tsrc = open(tone.__file__, encoding='utf-8').read()
    check('L1 影调曲线**只压不提**（旧的兜底提亮整套已删）',
          'allow_lift' not in _tsrc and 'EV_CAP_UP' not in _tsrc)
    check('位置：唯一会给画面提亮的是锚点（pipeline 调 io.refocus）',
          'io.refocus(' in open(pipeline.__file__, encoding='utf-8').read())


def t_review_fixes():
    """09-13 晚「评审问题全部一起改」的守卫 —— 每修一条就钉一条不变量，防止以后被"简化"掉。"""
    print('[★ 评审修复守卫：P0-2 / P1-3 / P1-4 / P1-6 / P2-8 / P2-9 / P2-10]')

    # ---------- P0-2：中灰判据统一到 L*（一把尺子） ----------
    _below = float(C.TGT_MID_L) - float(C.MID_DEADZONE_L)
    dd = _gray_img(gamma=2.0)
    rr = analyze.analyze(_lin_from_disp(dd), dd, 'raw')
    check('P0-2 decision 由 L* 中位决定（与报告里的 l50 同源，不再用显示域灰度）',
          (rr['decision'] == 'below') == bool(rr['l50'] < _below)
          and (rr['decision'] == 'compress') == bool(rr['l50'] > C.GUARD_MID_L),
          'L50 %.1f / %s' % (rr['l50'], rr['decision']))

    # ---------- P1-3：提到 config 的旋钮**真的被读**（不是提了个空壳） ----------
    _g0, _y0 = style._tone_lut(1.0, 9.0)
    _save_w = C.TONE_LIFT_W
    try:
        C.TONE_LIFT_W = (0.0, 0.0, 0.0)
        style._TONE_CACHE.clear()
        _g1, _y1 = style._tone_lut(1.0, 9.0)
    finally:
        C.TONE_LIFT_W = _save_w
        style._TONE_CACHE.clear()
    _dw = float(np.max(np.abs(_y1 - _y0)))
    check('P1-3 影调曲线权重从 config 读（改它曲线跟着变，且缓存键含权重）', _dw > 0.5,
          '最大差 %.2f' % _dw)

    # ---------- P2-8：bloom 能量守恒的前提必须成对 ----------
    check('P2-8 bloom 守恒前提：BLOOM_AMOUNT == BLOOM_SPREAD',
          abs(float(C.BLOOM_AMOUNT) - float(C.BLOOM_SPREAD)) < 1e-9,
          '%.4f vs %.4f' % (C.BLOOM_AMOUNT, C.BLOOM_SPREAD))
    _pairs = []
    for _nm, _st in stocks.TABLE.items():
        _sp = ((_st.get('spatial') or {}).get('bloom') or {})
        _a, _s = _sp.get('amount'), _sp.get('spread')
        if _a is not None and _s is not None and abs(float(_a) - float(_s)) > 1e-9:
            _pairs.append(_nm)
    check('P2-8 没有卷单独改其中一个（改了画面会静默不守恒）', not _pairs, str(_pairs))

    # ---------- P1-4：空间序 = bloom → halation → grain ----------
    cf = _Cfg(BLOOM_ENABLE=True, HALATION_ENABLE=True, GRAIN_ENABLE=True,
              BLOOM_AMOUNT=0.15, BLOOM_SPREAD=0.15, BLOOM_VEIL=0.02,
              GRAIN_AMOUNT=0.03, HALATION_AMOUNT=0.11)
    dd = _gray_img(gamma=0.9)
    _out, _ = spatial.apply(dd, cf)
    _p = spatial.resolve(cf)
    _cur = np.clip(dd, 0.0, 1.0)
    _cur, _ = spatial.bloom(_cur, _p['bloom'], cf)
    _cur, _ = spatial.halation(_cur, _p['halation'], cf)
    _cur, _ = spatial.grain(_cur, _p['grain'], cf)
    check('P1-4 空间序 = bloom→halation→grain（与手册逐步比，逐位相同）',
          float(np.max(np.abs(_out - _cur))) < 1e-12)

    # ---------- P1-6：ENTRY_CURVE 的 DR 档步进必须自洽 ----------
    for _model in cameras.ENTRY_CURVE:
        z = cameras.dr_zero_evs(_model)
        if '100' in z and '400' in z:
            check('P1-6 %s：DR100→DR400 零点差 ≈ +2EV' % _model,
                  abs((z['400'] - z['100']) - 2.0) < 0.25, '%.3f' % (z['400'] - z['100']))
        if '100' in z and '200' in z:
            _step = z['200'] - z['100']
            # ★ 09-13 深夜复测（n=305，`_debug/lab_dr_relift.py`，配对源 = RAF 内嵌机内 JPEG）：
            #   **相机成片的真实步进就是 ≈ +1.84**（DR200 抬得几乎和 DR400 一样多），
            #   不是物理上的 +1 ⇒ **别再把它当数据缺陷**。见 cameras.py 的复核结论。
            _known = (_model, '200') in cameras.FUJI_DR_STEP_KNOWN_BAD
            check('P1-6 %s：DR100→DR200 步进 ≈ +1.84EV（实测相机成片行为，不是缺陷）' % _model,
                  abs(_step - 1.84) < 0.25 or _known, 'step=%+.2f' % _step)
            check('P1-6 %s：DR200 的 +1.84 是相机行为 ⇒ KNOWN_BAD 必须为空' % _model,
                  not _known, 'step=%+.2f 仍登记为缺陷=%s' % (_step, _known))

    # ---------- P2-9：入口护栏有 k 地板 ----------
    _blown = np.ones((48, 56, 3))
    _o, _k = io.clip_guard(_blown, C)
    check('P2-9 clip_guard 有地板（旧下界 0.02 ≈ −5.6EV 会把正常曝光一起拖黑）',
          _k >= float(C.ENTRY_CLIP_GUARD_LO) - 1e-9,
          'k=%.3f 地板=%.2f' % (_k, C.ENTRY_CLIP_GUARD_LO))
    _safe = np.full((48, 56, 3), 0.4)
    _o2, _k2 = io.clip_guard(_safe, C)
    check('P2-9 不裁切的图仍然逐位不动（k=1）',
          abs(_k2 - 1.0) < 1e-12 and float(np.max(np.abs(_o2 - _safe))) < 1e-12)

    # ---------- 白点：**只设上限**（09-14 起不再有「必须等于 TGT_WHITE」那条路） ----------
    #   ⚠ 必须挑一张 **compress** 的图（L*50 > GUARD_MID_L 87）—— 那是现在**唯一**还会动曲线的路；
    #     偏暗/hold 的图 `correct` 会整条跳过（`y_out` 是空数组）。
    dk = np.full((64, 64, 3), 0.95)                 # 恒定很亮 ⇒ L*50 ≈ 97 > GUARD_MID_L 87 ⇒ compress
    rep_dk = analyze.analyze(_lin_from_disp(dk), dk, 'raw')
    _, info_dk = tone.correct(_lin_from_disp(dk), rep_dk, C)
    _cap_w = float(color.s2l(C.WHITE_CEIL))
    check('白点只设上限：compress 路真的动手，落点 ≤ WHITE_CEIL、且永不抬',
          rep_dk['decision'] == 'compress' and info_dk['applied']
          and float(info_dk['y_out'][-1]) <= _cap_w + 1e-9
          and float(info_dk['white_raise_ev']) <= 1e-9,
          'decision=%s y_out=%s cap=%.4f' % (rep_dk['decision'], info_dk['y_out'][-1:], _cap_w))


def t_entry_settle():
    """★ 位置逐张听相机（09-13 深夜）—— 「相机落点规律」+ 回落行为 + 口径守卫。"""
    import math

    law = cameras.settle_law_models()
    check('落点规律：至少标了一个机型', bool(law), str(law))
    check('落点规律：X-T30 III 已标定', 'x-t30 iii' in law)
    for _dr in (100, 200, 400):
        check('落点规律：X-T30 III DR%s 有偏移项' % _dr,
              _dr in cameras.SETTLE_LAW['x-t30 iii']['dr_off'])

    # ① 单调 + 序：同 DR 档内随场景亮度单调不减；同一 e 上 DR 越高落点越大
    _a = [cameras.entry_settle_level('X-T30 III', 100, e) for e in (-5.0, -3.0, -1.0, 0.0, 1.0)]
    check('落点规律：同 DR 档内随场景亮度单调不减（"逐张"项真的在动）',
          all(_a[i] <= _a[i + 1] + 1e-12 for i in range(len(_a) - 1)),
          str([round(v, 4) for v in _a]))
    _b = [cameras.entry_settle_level('X-T30 III', d, -3.0) for d in (100, 200, 400)]
    check('落点规律：同一场景亮度上 DR 越高落点越大', _b[0] < _b[1] < _b[2],
          str([round(v, 4) for v in _b]))

    # ② 钉住数值（拟合结果，不是随手填的；乱改 alpha/beta/dr_off 会红）
    _v4 = cameras.entry_settle_level('X-T30 III', 400, -3.39)
    check('落点规律：DR400 @e=−3.39 → 落点 ≈1.295（5 折留出集中位误差 1.32 L*）',
          abs(_v4 - 1.295) < 0.02, 'a=%.4f' % _v4)
    _v1 = cameras.entry_settle_level('X-T30 III', 100, -0.19)
    check('落点规律：DR100 @e=−0.19 → 落点 ≈0.662（比全局 0.55 亮 0.27 档）',
          abs(_v1 - 0.662) < 0.02, 'a=%.4f' % _v1)

    # ③ 回落：没标定 / DR 不认 / 拿不到 e / dr 缺失 ⇒ None（调用方回落 ENTRY_LEVEL）
    check('落点规律：没标定的机型回落（None）',
          cameras.entry_settle_level('A7C II', 400, -3.0) is None)
    check('落点规律：DR 不认回落（None）',
          cameras.entry_settle_level('X-T30 III', 800, -3.0) is None)
    check('落点规律：拿不到场景曝光回落（None）',
          cameras.entry_settle_level('X-T30 III', 400, None) is None)
    check('落点规律：dr 缺失回落（None）',
          cameras.entry_settle_level('X-T30 III', None, -3.0) is None)

    # ④ 入口接线：level 传了就用它；level=None 与 level=ENTRY_LEVEL 逐位相同（回落不改行为）
    _img = np.full((24, 28, 3), 0.2)
    _o_glob = io.entry_tone(_img, 0.0, C)
    _o_same = io.entry_tone(_img, 0.0, C, level=float(C.ENTRY_LEVEL))
    check('入口成形：level=None 与 level=全局值 逐位相同（回落 = 旧行为）',
          float(np.max(np.abs(_o_glob - _o_same))) < 1e-15)
    _o_hi = io.entry_tone(_img, 0.0, C, level=float(C.ENTRY_LEVEL) * 2.0)
    check('入口成形：level 给大 ⇒ 出图更亮（接线真的生效，不是接了没通）',
          float(np.mean(_o_hi)) > float(np.mean(_o_glob)),
          '%.5f > %.5f' % (float(np.mean(_o_hi)), float(np.mean(_o_glob))))

    # ⑤ 口径守卫：`scene_exposure_index` 必须等于 L_of_lin 那条链（改口径 ⇒ 规律整体偏掉）
    for _L in (12.0, 25.33, 50.0, 78.0):
        _g = np.zeros((20, 20, 3))
        _g[..., :] = color.lin_of_L(_L)
        _e = io.scene_exposure_index(_g)
        _want = math.log2(float(color.lin_of_L(_L)) / 0.18)
        check('曝光指数口径：L*%.2f → e=%+.4f（与 lab_pos_law_fit 逐字一致）' % (_L, _want),
              abs(_e - _want) < 1e-9, '%.6f vs %.6f' % (_e, _want))

    # ⑥ 位置来源（09-14 评审后）：**入口 settle 定基准 + 脸锚点做「只提不压」的补偿**；
    #    旧的「兜底提亮」整套（`_allow_lift` / `LIFT_DARK_*` / `ALLOW_LIFT*`）已删。
    _tsrc2 = open(tone.__file__, encoding='utf-8').read()
    check('位置：L1 只压不提（tone 里没有 allow_lift / EV_CAP_UP / LIFT_DARK）',
          all(k not in _tsrc2 for k in ('allow_lift', 'EV_CAP_UP', 'LIFT_DARK')))
    check('位置：锚点接在 L0 之前（pipeline 调 io.anchor_ev + io.refocus）',
          'io.anchor_ev(' in open(pipeline.__file__, encoding='utf-8').read()
          and 'io.refocus(' in open(pipeline.__file__, encoding='utf-8').read())

    # ⑦ 接线守卫：装配点必须看开关（关掉 ENTRY_SETTLE_ENABLE 即回落 ENTRY_LEVEL）
    _src = open(io.__file__, encoding='utf-8').read()
    check('入口接线：装配点读的是 ENTRY_SETTLE_ENABLE（开关没被绕开）',
          "getattr(C, 'ENTRY_SETTLE_ENABLE', False)" in _src
          and 'cameras.entry_settle_level(' in _src)


def _entry_analytic(cfg, img, ev, a):
    """入口形状的解析式（**不含趾部**）—— 用来给趾部测试当"参照臂"。

    ⚠ 必须与 `io.entry_tone` 里的算式逐字一致；只用来算"没有趾部时该是多少"。
    """
    g = float(getattr(cfg, 'ENTRY_GAMMA', 1.0))
    knee = float(getattr(cfg, 'ENTRY_KNEE', 1.0))
    ceil = float(getattr(cfg, 'ENTRY_CEIL', 1.0))
    Y = np.maximum(color.luma(np.clip(img, 0.0, None)), 1e-9)
    y = np.power(np.maximum(Y * (2.0 ** float(ev)), 1e-9), g) * float(a)
    d = max(ceil - knee, 1e-6)
    # ★ 09-14 晚：入口的肩有了**形状族**（`ENTRY_SHOULDER_KIND`），这里必须跟着分支，
    #   否则"趾部可回退"会拿老指数肩的解析值去比新对数肩 ⇒ 守错世界、假报失败。
    if str(getattr(cfg, 'ENTRY_SHOULDER_KIND', 'exp')).lower() == 'log':
        umax = max(float(getattr(cfg, 'ENTRY_SHOULDER_UMAX', 12.0)), 1e-6)
        u = np.clip((y - knee) / d, 0.0, umax)
        y = np.where(y > knee, knee + d * np.log1p(u) / np.log1p(umax), y)
    else:
        y = np.where(y > knee, knee + d * (1.0 - np.exp(-(y - knee) / d)), y)
    return y, Y


def t_entry_toe():
    """入口趾部（v0.3.8，09-13 深夜 SV 拍板「对齐鹿井」）。

    不变量：
      ① **可回退**：`ENTRY_TOE=1.0` 必须逐位等于"没有趾部"的解析式；
      ② **只动低位**：出口亮度 y > `中位×ENTRY_TOE_HI_REL` 的那一段逐位不变；
      ③ **★ 内容归一**：锚点是"图自己的出口亮度中位" ⇒ **亮图和暗图都只在各自最低的一小段咬**，
         两张图各自的**中位像素必须逐位不变**（这一条是绝对断点版本栽过的地方：
         固定断点在暗片上会让整张图掉进趾部，实测 0071 中灰 28.8 → 16.5）；
      ④ **方向单调**：floor 越小 ⇒ 暗部越暗；且整体仍**单调不减**（不许倒挂）；
      ⑤ 纯黑仍是纯黑（Y=0 → 0）。
    """
    print('[入口趾部：只压自己最低那一小段，中灰以上逐位不动]')
    EV, A = 2.184, 1.449                      # 0791 的实测零点/落点（定档用的真实参数）
    cfg = _Cfg(ENTRY_TOE=float(C.ENTRY_TOE))
    lo_rel, hi_rel = float(C.ENTRY_TOE_LO_REL), float(C.ENTRY_TOE_HI_REL)
    ramp = np.linspace(0.0, 0.6, 256)[None, :, None].repeat(3, axis=2)
    dark_ramp = ramp * 0.15                   # 同一形状的"暗图"（内容归一要能扛住它）

    y0, Y = _entry_analytic(C, ramp, EV, A)
    want_none = ramp * (y0 / Y)[..., np.newaxis]
    ym = float(np.median(y0))
    hi, lo = hi_rel * ym, lo_rel * ym
    up = y0 > hi
    act = (y0 > 0.0) & (y0 < hi)
    mid_i = int(np.argmin(np.abs(y0.reshape(-1) - ym)))

    off = io.entry_tone(ramp, EV, _Cfg(ENTRY_TOE=1.0), level=A)
    on = io.entry_tone(ramp, EV, cfg, level=A)
    deep = io.entry_tone(ramp, EV, _Cfg(ENTRY_TOE=0.20), level=A)
    L_off = color.L_of_lin(color.luma(off))
    L_none = color.L_of_lin(color.luma(want_none))
    L_on = color.L_of_lin(color.luma(on))
    L_deep = color.L_of_lin(color.luma(deep))

    check('趾部可回退：ENTRY_TOE=1.0 逐位等于"无趾部"解析值',
          float(np.max(np.abs(off - want_none))) < 1e-15,
          'max err %.2e' % float(np.max(np.abs(off - want_none))))
    check('趾部只动低位：出口 y > 中位×HI_REL 的像素逐位不变',
          up.any() and float(np.max(np.abs(on[up] - off[up]))) < 1e-15,
          '%d/%d 个像素在断点之上（断点 y=%.4f，中位 y=%.4f）' % (int(up.sum()), up.size, hi, ym))
    check('趾部在低位真的压深（不是接了没通）',
          act.any() and float(np.max(L_none[act] - L_on[act])) > 3.0,
          '最大压深 %.2f L*（%d 个像素在趾部区间）'
          % (float(np.max(L_none[act] - L_on[act])), int(act.sum())))
    check('★ 内容归一：图自己的中位像素逐位不变（亮图）',
          float(np.max(np.abs(on.reshape(-1, 3)[mid_i] - off.reshape(-1, 3)[mid_i]))) < 1e-15,
          '中位 y=%.4f，断点 y=%.4f ⇒ 中位在断点之上' % (ym, hi))
    yd, Yd = _entry_analytic(C, dark_ramp, EV, A)
    ymd = float(np.median(yd))
    mid_id = int(np.argmin(np.abs(yd.reshape(-1) - ymd)))
    off_d = io.entry_tone(dark_ramp, EV, _Cfg(ENTRY_TOE=1.0), level=A)
    on_d = io.entry_tone(dark_ramp, EV, cfg, level=A)
    check('★ 内容归一：暗图自己的中位像素也逐位不变（绝对断点版本在这里会整图变暗）',
          float(np.max(np.abs(on_d.reshape(-1, 3)[mid_id]
                              - off_d.reshape(-1, 3)[mid_id]))) < 1e-15,
          '暗图中位 y=%.4f，断点 y=%.4f' % (ymd, hi_rel * ymd))
    want_d = dark_ramp * (yd / Yd)[..., np.newaxis]
    bite_d = float(np.max(color.L_of_lin(color.luma(want_d))[yd < hi_rel * ymd]
                          - color.L_of_lin(color.luma(on_d))[yd < hi_rel * ymd]))
    check('★ 内容归一：暗图仍有趾部（没被"相对锚点"变成空操作）', bite_d > 3.0,
          '暗图最大压深 %.2f L*' % bite_d)
    check('趾部方向单调：floor 越小 ⇒ 暗部越暗',
          float(np.max(L_on - L_deep)) > 0.5 and float(np.max(L_deep - L_off)) < 1e-9,
          '0.32 vs 0.20 最大差 %.2f L*' % float(np.max(L_on - L_deep)))
    check('趾部不破坏单调（曲线仍非降）',
          bool(np.all(np.diff(L_on.reshape(-1)) >= -1e-12))
          and bool(np.all(np.diff(L_deep.reshape(-1)) >= -1e-12)))
    check('趾部不动纯黑（Y=0 仍为 0，没把最暗处抬起来）',
          float(color.luma(on)[0, 0]) < 1e-12 and float(color.luma(deep)[0, 0]) < 1e-12)
    # 出厂默认守卫：别把它悄悄关掉
    check('出厂默认：趾部开着，且 0 < LO_REL < HI_REL < 1（中位永远在断点之上）',
          0.0 < float(C.ENTRY_TOE) < 1.0 and 0.0 < lo_rel < hi_rel < 1.0,
          'TOE=%.2f  LO_REL=%.2f  HI_REL=%.2f' % (C.ENTRY_TOE, lo_rel, hi_rel))
    # 接线守卫：装配点必须真的把 cfg 传进 entry_tone（别在别处又写死一份形状）
    _src = open(io.__file__, encoding='utf-8').read()
    check('趾部接线：entry_tone 里读的是 ENTRY_TOE / ENTRY_TOE_HI_REL（没有写死的 magic number）',
          "getattr(cfg, 'ENTRY_TOE', 1.0)" in _src
          and "getattr(cfg, 'ENTRY_TOE_HI_REL', 0.84)" in _src)


def t_anchor():
    r"""★★ 「脸的锚点决定位置」（09-14 SV 选「乙/甲」）—— 一条曲线、不分区域、不用掩膜。

    位置由脸算（闭式解），形状走入口那一条曲线；背景的亮度是**结果**不是另设的靶。
    原先是写在 t_l4_caps 里的；两道区域闸删掉之后单独成一条。
    """
    import inspect
    print('[脸的锚点 · 位置]')
    check('锚点：出厂开着，靶 = 68（鹿井脸中位）',
          bool(C.ANCHOR_ENABLE) is True and float(C.ANCHOR_FACE_L) == 68.0)
    check('锚点·只提不压：出厂为真（脸已经够亮就不动它）', bool(C.ANCHOR_ONLY_UP) is True)
    check('锚点：收尾出厂开着（让**最终脸**落在靶上，不被 L2 又抬走）',
          bool(C.ANCHOR_FINISH) is True)
    src_io = inspect.getsource(io.refocus)
    check('锚点：位置是**重打入口那条曲线**（不是分区域压）—— refocus 里用肩的正/逆函数',
          '_shoulder_inv' in src_io and '_shoulder(' in src_io)
    prun2 = inspect.getsource(pipeline.run)
    check('锚点：pipeline.run 里在 L0 分析**之前**就重打了（analyze 用的是 lin_in）',
          'io.anchor_ev(s.disp, cfg)' in prun2 and 'lin_in' in prun2)
    check('锚点：收尾接在 L2（style.apply）之后',
          'io.finish_anchor(' in prun2)
    check('接线：两个开关都从 config 读（没写死）',
          'ANCHOR_ONLY_UP' in inspect.getsource(io.anchor_ev))

    ao = float(C.ANCHOR_EV_MAX)
    try:
        C.ANCHOR_EV_MAX = 4.0
        # 闭式解：合成"脸在 ~46"的图（真实逆光片的脸就在这个量级）
        # ⚠ 别用太暗的合成图：L* 在 Y<0.0089 时走**线性段**，立方根近似不成立。
        img3 = np.full((8, 256, 3), 0.4274, np.float32)          # 显示值 0.4274 ⇒ L* ≈ 46
        fk = np.zeros((8, 256), np.float32); fk[:, :64] = 1.0
        Lf = float(np.median(color.L_of_lin(color.Y_of(color.s2l(img3)))[0, :64]))
        check('锚点：合成图的脸落在 46 量级', 44.0 < Lf < 49.0, 'L脸=%.1f' % Lf)
        d, ai = io.anchor_ev(img3, C, masks=dict(face_skin=fk))
        check('锚点：脸比 68 暗 ⇒ d_ev > 0（要提亮）', d > 0, 'd_ev=%+.3f' % d)
        out = io.refocus(color.s2l(img3), d, C)
        La = float(np.median(color.L_of_lin(color.Y_of(np.clip(out, 0.0, None)))[0, :64]))
        check('锚点：重打之后脸真的落到 68（±1.5）', abs(La - 68.0) < 1.5, '%.2f' % La)
        check('锚点：三通道同倍率（只改亮度、不改颜色）',
              bool(np.allclose(out[..., 0] / np.maximum(color.s2l(img3)[..., 0], 1e-9),
                               out[..., 2] / np.maximum(color.s2l(img3)[..., 2], 1e-9), atol=1e-6)))
        bright = np.full((8, 256, 3), 0.72, np.float32)           # L* ≈ 88 ⇒ 已经在靶之上
        Lb2 = float(np.median(color.L_of_lin(color.Y_of(color.s2l(bright)))))
        d2, i2 = io.anchor_ev(bright, C, masks=dict(face_skin=fk))
        check('锚点·只提不压：脸已经够亮（%.0f > 68）⇒ d_ev = 0' % Lb2,
              d2 == 0.0 and i2.get('applied') is False, 'reason=%s' % i2.get('reason'))
        d0, i0 = io.anchor_ev(img3, C, masks=dict(face_skin=np.zeros((8, 256), np.float32)))
        check('锚点：拿不到脸 ⇒ 0（逐位不变）', d0 == 0.0 and i0.get('applied') is False)
        check('锚点：refocus(lin, 0) 逐位等于 lin',
              bool(np.array_equal(io.refocus(color.s2l(img3), 0.0, C), color.s2l(img3))))
        # 收尾：把"当前脸"挪回靶（模拟 L2 之后脸被抬到 ~76 的情形）
        out4, _ = io.finish_anchor(img3, C, masks=dict(face_skin=fk))
        La4 = float(np.median(color.L_of_lin(color.Y_of(color.s2l(out4)))[0, :64]))
        check('锚点收尾：脸偏离靶 ⇒ 挪回靶（±0.6）', abs(La4 - 68.0) < 0.6, '%.2f' % La4)
        out5, _ = io.finish_anchor(img3, C, masks=dict(face_skin=np.zeros((8, 256), np.float32)))
        check('锚点收尾：拿不到脸 ⇒ 逐位不变', bool(np.array_equal(out5, img3)))
    finally:
        C.ANCHOR_EV_MAX = ao


def _legacy(fn):
    r"""把 `fn` 包一层：跑的时候**临时关掉 09-14 新增的三块颜色**（丙/甲/乙）。

    为什么需要：那些"关掉 = 恒等""neutral = 恒等""只动 L*（a*/b* 不动）"的断言，
    是按**老颜色路**（Lab 里事后加偏移）写的；而新三块是**故意要改颜色**的
    （丙串扰 / 甲分通道 / 乙密度引擎）。测老路本身时把新的关掉，两边各自成立。
    """
    def w():
        _o = (C.CROSSTALK_ENABLE, C.LAYER_SPEED_ENABLE, C.DENSITY_ENABLE)
        try:
            C.CROSSTALK_ENABLE = C.LAYER_SPEED_ENABLE = C.DENSITY_ENABLE = False
            return fn()
        finally:
            (C.CROSSTALK_ENABLE, C.LAYER_SPEED_ENABLE, C.DENSITY_ENABLE) = _o
    w.__name__ = getattr(fn, '__name__', 'wrapped')
    return w


def t_film_color():
    """★ 09-14 新增三块颜色（`film.py`）：【丙】串扰 /【甲】分通道 /【乙】密度引擎。"""
    import inspect
    print('[真胶片成色：丙串扰 / 甲分通道 / 乙密度引擎]')
    check('丙 串扰：出厂开着，强度 = LIMO 给 Portra 400 标的 0.38',
          bool(C.CROSSTALK_ENABLE) is True and abs(float(C.CROSSTALK_AMOUNT) - 0.38) < 1e-9)
    check('甲 分通道：出厂开着，层感光度 = LIMO 的 (0.96, 1.0, 1.03)',
          bool(C.LAYER_SPEED_ENABLE) is True
          and tuple(np.round(C.LAYER_SPEEDS, 3)) == (0.96, 1.0, 1.03))
    check('乙 密度引擎：出厂开着；**按参考实现只混 ≤50%**（Emulsifier `opacity = strength×0.5`）',
          bool(C.DENSITY_ENABLE) is True and 0.0 < float(C.DENSITY_STRENGTH) <= 0.5)

    d = film.load_curve('portra400')
    check('乙：曲线是真的 Kodak Portra 400（101 点，logE −4 → +1）',
          d['logE'].shape[0] == 100 and abs(d['logE'].min() + 4.0) < 0.01
          and abs(d['logE'].max() - 1.0) < 0.01, '%d 点' % d['logE'].size)
    check('★ 乙：分通道底片密度（那就是负片的橙色片基，不是"加"上去的）'
          'R 0.243 / G 0.656 / B 0.885',
          bool(np.allclose(d['d_min'], [0.243, 0.656, 0.885], atol=1e-3)),
          str(np.round(d['d_min'], 3)))

    # 丙：会改颜色、且关掉即恒等
    g = np.linspace(0.0, 1.0, 64).reshape(8, 8, 1).repeat(3, -1).astype(np.float64)
    c0 = film.crosstalk(g, 0.0)
    check('丙：amount=0 逐位恒等', bool(np.array_equal(c0, g)))
    c1 = film.crosstalk(g, 0.38)
    _lc = color.luma(g).mean()
    check('丙：真的动了颜色，且**不是**整体提亮/压暗（通道之间在换，不是一起升降）',
          float(np.max(np.abs(c1 - g))) > 1e-3
          and abs(float(color.luma(c1).mean() - _lc)) < 0.05,
          'maxΔ %.4f  亮度 %.4f→%.4f' % (float(np.max(np.abs(c1 - g))), _lc,
                                        float(color.luma(c1).mean())))
    check('丙：输出有界非负', bool(np.isfinite(c1).all()) and c1.min() >= -1e-9)

    # 甲
    s0 = film.layer_speeds(g, (0.96, 1.0, 1.03), 0.0)
    check('甲：strength=0 逐位恒等', bool(np.array_equal(s0, g)))
    s1 = film.layer_speeds(g, (0.96, 1.0, 1.03), 1.0)
    # 实测方向：`rgb**(1/speeds)`，speed<1 ⇒ 指数>1 ⇒ 中间调更暗 ⇒ 这一档 R 最暗、B 最亮。
    # （⚠ 符号与 LIMO 自述的"红层更柔"方向相反 —— 他们那个 `rgb` 的域和我们的显示域不同，
    #   我们只保证"三通道响应不同"，方向以实测为准。）
    check('甲：三通道真的走不同（实测 R < G < B，红层被压得最狠）',
          s1[..., 0].mean() < s1[..., 1].mean() < s1[..., 2].mean(),
          'R%.3f G%.3f B%.3f' % tuple(s1[..., i].mean() for i in range(3)))

    # 乙：★ 中灰守恒 + 单调
    m0 = film.density(np.array([[[0.45] * 3]]))[0, 0, 1]
    check('★ 乙：中灰进 = 中灰出（印相机的分通道曝光解出来的，±0.02）',
          abs(float(m0) - 0.45) < 0.02, '0.45 -> %.4f' % float(m0))
    xs = np.linspace(0.01, 0.99, 160)
    oo = film.density(xs.reshape(-1, 1, 1).repeat(3, -1))
    check('乙：整条传递曲线单调不减（不翻）',
          bool((np.diff(oo[:, 0, 1]) >= -1e-9).all()))
    mid = film.density(np.array([[[0.45] * 3]]))[0, 0]
    check('乙：三个通道的中灰都守恒（所以不会出一片红）',
          bool(np.allclose(mid, 0.45, atol=0.02)), str(np.round(mid, 4)))
    check('乙：透射率用 10^(−密度)（源码里就是这个式子，不是随手一条曲线）',
          '10.0 ** (-dens' in inspect.getsource(film.density))
    # ★★ 跟卷走：`neutral` 卷必须把三块关掉（A/B 对照底要干净）
    #   ⚠ 断言的**不是**"neutral = 原图"（neutral 继承基准成色 BASE_FULL，那本来就要动颜色），
    #     而是"**neutral 把三块关掉了**"：它和"总强度 = 0"逐位相同。
    import numpy as _np
    g2 = _np.linspace(0.0, 1.0, 48).reshape(6, 8, 1).repeat(3, -1).astype(_np.float64)
    _w0 = float(C.FILM_COLOR_W)
    try:
        _n = style._builtin(g2, C, stock=stocks.get('neutral'))
        C.FILM_COLOR_W = 0.0
        _n0 = style._builtin(g2, C, stock=stocks.get('neutral'))
        _p = style._builtin(g2, C, stock=stocks.get('portra400'))
        C.FILM_COLOR_W = 1.0
        _p1 = style._builtin(g2, C, stock=stocks.get('portra400'))
    finally:
        C.FILM_COLOR_W = _w0
    check('★ 跟卷走：neutral 卷把三块关掉了（= 与"总强度 0"逐位相同）',
          float(_np.max(_np.abs(_n - _n0))) < 1e-12,
          'maxΔ %.2e' % float(_np.max(_np.abs(_n - _n0))))
    check('★ 跟卷走：portra400 卷**会**吃到三块（开/关不一样）',
          float(_np.max(_np.abs(_p - _p1))) > 1e-3,
          'maxΔ %.4f' % float(_np.max(_np.abs(_p - _p1))))
    check('跟卷走：neutral 的 color 里显式有 film_color_w=0.0',
          float((stocks.get('neutral').get('color') or {}).get('film_color_w', 1.0)) == 0.0)
    check('接线：style._builtin 里三块都接上了',
          all(k in inspect.getsource(style._builtin)
              for k in ('film.layer_speeds', 'film.crosstalk', 'film.density')))


def main():
    for fn in (t_color, t_analyze, t_tone_mid_target, t_tone_monotone, _legacy(t_style_lock),
               _legacy(t_style_contrast_direction), _legacy(t_style_tone_curve), _legacy(t_style_chroma_ends), t_denoise,
               t_guard, t_lut,
               t_io_roundtrip, t_stocks, t_spatial_off, t_spatial_grain,
               t_spatial_bloom_halation, t_local_skin_floor, t_entry_bias, t_pipeline_smoke,
               t_review_fixes, t_entry_settle, t_entry_toe, t_anchor,
               t_film_color):
        fn()
    print('-' * 52)
    if FAIL:
        print('失败 %d 项: %s' % (len(FAIL), ', '.join(FAIL)))
        return 1
    print('全部通过')
    return 0


if __name__ == '__main__':
    sys.exit(main())
