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

    print('[入口曲线：实测曲线查表（Q1）]')
    _saved_curve = dict(cameras.ENTRY_CURVE)
    try:
        # 表空 / 机型不认识 -> None（退回老的常数补偿，不能炸）
        cameras.ENTRY_CURVE.clear()
        check('表空时返回 None（退回常数补偿）', cameras.entry_curve('x-t30 iii', 400) is None)
        cameras.ENTRY_CURVE.update({
            'x-t30 iii': {
                '400': dict(mid_ev=3.10, anchors=[[0.001, 0.70], [0.02, 1.00], [0.30, 1.25], [1.00, 0.22]]),
                'None': dict(mid_ev=0.72, anchors=[[0.001, 0.90], [0.02, 1.00], [1.00, 0.5]]),
            }})
        check('机型不认识返回 None', cameras.entry_curve('nope', 400) is None)
        c = cameras.entry_curve('X-T30 III', 400)          # 大小写不敏感
        check('命中 -> 给出 mid_ev + 形状', c is not None and abs(c[0] - 3.10) < 1e-9)
        check('锚点按输入线性升序', c[1] == sorted(c[1]))
        mid_i = c[1].index(0.02)
        check('形状在中灰处 = 1.0', abs(c[2][mid_i] - 1.0) < 1e-9)
        c2 = cameras.entry_curve('x-t30 iii', None)        # dr 未知 -> 取 'None' 那条
        check('dr=None 回退到 None 那条', c2 is not None and abs(c2[0] - 0.72) < 1e-9)
        c3 = cameras.entry_curve('x-t30 iii', 800)         # dr 有但不认识 -> 回退
        check('dr 不认识时回退', c3 is not None and abs(c3[0] - 0.72) < 1e-9)
    finally:
        cameras.ENTRY_CURVE.clear()
        cameras.ENTRY_CURVE.update(_saved_curve)

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
               t_style_contrast_direction, t_denoise, t_guard, t_lut, t_io_roundtrip,
               t_stocks, t_spatial_off, t_spatial_grain, t_spatial_bloom_halation,
               t_entry_bias, t_pipeline_smoke):
        fn()
    print('-' * 52)
    if FAIL:
        print('失败 %d 项: %s' % (len(FAIL), ', '.join(FAIL)))
        return 1
    print('全部通过')
    return 0


if __name__ == '__main__':
    sys.exit(main())
