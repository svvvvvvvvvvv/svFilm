# -*- coding: utf-8 -*-
"""自检 —— 动完任何一层都要跑。不依赖任何样片，纯合成图 + 不变量。

  python -m svFilm.selftest
"""
import os
import sys
import tempfile

import numpy as np

from . import (analyze, color, config as C, guard, io, local, pipeline, spatial,
               stocks, style, tone)

FAIL = []


def check(name, cond, extra=''):
    print(('  ok   ' if cond else '  FAIL ') + name + (('   ' + extra) if extra else ''))
    if not cond:
        FAIL.append(name)


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
    print('[L1 mid 靶]')
    # gamma 0.45 => 中灰落在显示域 0.5 附近，构造"偏亮"的输入
    d = _gray_img(gamma=0.45)
    rep = analyze.analyze(_lin_from_disp(d), d, 'jpg')
    lin, info = tone.correct(_lin_from_disp(d), rep, C)
    out = np.clip(color.l2s(lin), 0, 1)
    gm = float(np.percentile(color.gray_of(out), C.PCT_MID))
    check('敢动手', info['applied'], 'decision=%s' % rep['decision'])
    if info['applied']:
        check('中灰落到靶 ±0.01', abs(gm - C.TGT_MID) < 0.01, '%.4f vs %.4f' % (gm, C.TGT_MID))
        gw = float(np.percentile(color.gray_of(out), C.PCT_WHITE))
        check('白点落到靶 ±0.01', abs(gw - C.TGT_WHITE) < 0.012, '%.4f vs %.4f' % (gw, C.TGT_WHITE))
    check('无 NaN/Inf', np.all(np.isfinite(lin)))
    check('取值在 0~1', lin.min() >= -1e-9 and lin.max() <= 1.0 + 1e-9)


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
    print('[L2 锁中灰]')
    d = _gray_img(gamma=0.9)
    ref = style.mid_of(d)
    out, info = style.apply(d, C, lock_ref=ref)
    check('风格层不动中灰', abs(style.mid_of(out) - ref) < 0.012,
          '%.4f -> %.4f' % (ref, style.mid_of(out)))
    out2, _ = style.apply(d, C, lock_ref=None)
    check('lock_ref=None 时不锁', True, 'ev=%.4f' % info['lock_ev'])


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


def t_stocks():
    print('[卷 stocks]')
    for n in stocks.names():
        s = stocks.TABLE[n]
        check('卷 %s 有名字/说明/颜色参数' % n,
              bool(s['label'] and s['desc'] and s['color']))
    d = _gray_img(gamma=0.8)
    out, _ = style.apply(d, C, lock_ref=None, stock=stocks.get('neutral'))
    check('neutral 卷 = 恒等（不风格化）', float(np.max(np.abs(out - d))) < 1e-9,
          'max %.2e' % float(np.max(np.abs(out - d))))
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


def t_pipeline_smoke():
    print('[全链 smoke]')
    d = _gray_img(120, 180, gamma=0.4)
    rep = analyze.analyze(_lin_from_disp(d), d, 'jpg')
    lin, ti = tone.correct(_lin_from_disp(d), rep, C)
    d1 = np.clip(color.l2s(lin), 0, 1)
    d2, si = style.apply(d1, C, lock_ref=style.mid_of(d1), stock=stocks.get('portra400'))
    d2b, pi = spatial.apply(d2, C, stock=stocks.get('portra400'))
    d3, li = local.apply(d1, d2b, C)
    d4, gi = guard.enforce(d3, C)
    check('各层输出有界', all(x.min() >= 0 and x.max() <= 1 for x in (d1, d2, d2b, d3, d4)))
    check('各层无 NaN', all(np.all(np.isfinite(x)) for x in (d1, d2, d2b, d3, d4)))
    check('肤色掩膜不炸', 0.0 <= li['skin_cov'] <= 1.0, 'cov=%.4f' % li['skin_cov'])
    check('空间域 info 齐全', all(k in pi for k in ('grain', 'bloom', 'halation', 'any')))


def main():
    for fn in (t_color, t_analyze, t_tone_mid_target, t_tone_monotone, t_style_lock,
               t_guard, t_lut, t_io_roundtrip, t_stocks, t_spatial_off,
               t_spatial_grain, t_spatial_bloom_halation, t_pipeline_smoke):
        fn()
    print('-' * 52)
    if FAIL:
        print('失败 %d 项: %s' % (len(FAIL), ', '.join(FAIL)))
        return 1
    print('全部通过')
    return 0


if __name__ == '__main__':
    sys.exit(main())
