# -*- coding: utf-8 -*-
"""L2 —— 风格层。胶片性格。disp 域进、disp 域出。

原则：风格是"围绕 L1 定下的中灰"做事，不许把它改回去。
所以最后一步是 LOCK_MID —— 风格层把中灰推歪多少，就拉回多少。

内含三件东西：
  a) 内置参数化胶片色（色交叉 + 分裂色调 + 彩度）
  b) 明度对比（只动 L*，a/b 原样保留，所以对比不脏色）
  c) 外部 .cube（读 + 三线性插值），这就是"现成胶片 LUT 当骨架"的接口
外加 bake_cube()：把本层离线烘成自己的 .cube。
"""
from __future__ import annotations

import os

import numpy as np

from . import color
from . import config as C
from . import film
from . import stocks
from . import tone


# ---------------- .cube 读写 ----------------
def cube_read(path):
    n = None
    data = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            if s.upper().startswith('LUT_3D_SIZE'):
                n = int(s.split()[-1])
                continue
            if s.upper().startswith('TITLE') or s.upper().startswith('DOMAIN'):
                continue
            parts = s.split()
            if len(parts) >= 3:
                try:
                    data.append([float(parts[0]), float(parts[1]), float(parts[2])])
                except ValueError:
                    continue
    if n is None or len(data) != n ** 3:
        raise ValueError('不是合法的 3D .cube: %s (size=%s, 行数=%d)' % (path, n, len(data)))
    # 顺序：R 变化最慢，B 最快
    arr = np.asarray(data, np.float64).reshape(n, n, n, 3)
    return arr


def cube_write(path, cube, title='svFilm'):
    n = cube.shape[0]
    cube = np.asarray(cube, np.float64)
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write('TITLE "%s"\n' % title)
        f.write('LUT_3D_SIZE %d\n' % n)
        f.write('DOMAIN_MIN 0.0 0.0 0.0\n')
        f.write('DOMAIN_MAX 1.0 1.0 1.0\n')
        for r in range(n):
            for g in range(n):
                for b in range(n):
                    v = cube[r, g, b]
                    f.write('%.6f %.6f %.6f\n' % (v[0], v[1], v[2]))
    return path


def cube_apply(disp, cube):
    """三线性插值。disp 越界部分先夹到 0~1。"""
    n = cube.shape[0]
    x = np.clip(np.asarray(disp, np.float64), 0.0, 1.0) * (n - 1)
    i0 = np.clip(np.floor(x).astype(np.int32), 0, n - 2)
    d = x - i0
    r0, g0, b0 = i0[..., 0], i0[..., 1], i0[..., 2]
    r1, g1, b1 = r0 + 1, g0 + 1, b0 + 1
    dr, dg, db = d[..., 0:1], d[..., 1:2], d[..., 2:3]
    c = cube
    out = (
        c[r0, g0, b0] * (1 - dr) * (1 - dg) * (1 - db) + c[r1, g0, b0] * dr * (1 - dg) * (1 - db) +
        c[r0, g1, b0] * (1 - dr) * dg * (1 - db) + c[r0, g0, b1] * (1 - dr) * (1 - dg) * db +
        c[r1, g1, b0] * dr * dg * (1 - db) + c[r1, g0, b1] * dr * (1 - dg) * db +
        c[r0, g1, b1] * (1 - dr) * dg * db + c[r1, g1, b1] * dr * dg * db
    )
    return out


# ---------------- 内置颜色模型 ----------------
# ★ 胶片影调曲线（09-13 SV 拍板「先做 A 影调」）：
#   控制点里**黑(0) 与白(100) 都不动**，只做两件事 —— 趾部压深 + 中高调抬起；
#   中灰抬起而白锚在 100 ⇒ 高光端自然形成肩部（"胶片高光滚降"而不是"刺白"）。
#   定义域/值域都是 L*（0~100）。用 PCHIP 保证**严格单调**（折点会在天空那种平滑渐变里出色带）。
#   形状依据与参数含义见 config.py 的 TONE_TOE / TONE_LIFT 注释。
_TONE_CACHE = {}


def _tone_pts(toe, lift, shoulder=0.0):
    """曲线控制点。`shoulder` = 高光段**整段**下收量（L*，见 config.TONE_SHOULDER）。

    ⚠ 必须整段收（70/90/100 一起），**不能只压白点** —— 只压 100 会让 90→100 塌成平板
      （实测斜率掉到 0 = 顶上一截全是死白，没有层次）。
    """
    xs = np.array([0.0, 10.0, 30.0, 50.0, 70.0, 90.0, 100.0])
    sh = float(shoulder)
    w_toe, w_lift, w_sh = _tone_w()
    ys = np.array([0.0,
                   10.0 - toe,                          # 趾部：暗部相对中灰压深
                   30.0 - toe * w_toe,
                   50.0 + lift * w_lift[0],             # 中灰抬起
                   70.0 + lift * w_lift[1] - sh * w_sh[0],   # ↓ 肩部：越靠顶收得越多
                   90.0 + lift * w_lift[2] - sh * w_sh[1],
                   100.0 - sh * w_sh[2]])               # 白端（这是"白锚"第一次可动）
    return xs, ys


def _tone_w():
    """影调曲线的形状权重（P1-3 提到 config）。★ 必须进缓存键：权重一变曲线就得重算。"""
    return (float(getattr(C, 'TONE_TOE_W', 0.45)),
            tuple(getattr(C, 'TONE_LIFT_W', (0.55, 0.95, 0.55))),
            tuple(getattr(C, 'TONE_SHOULDER_W', (0.30, 0.70, 1.00))))


def _tone_lut(toe, lift, shoulder=0.0):
    """(tone_toe, tone_lift, tone_shoulder) -> (x_grid, y_grid)。1024 点，带缓存。"""
    key = (round(float(toe), 4), round(float(lift), 4), round(float(shoulder), 4)) + _tone_w()
    hit = _TONE_CACHE.get(key)
    if hit is not None:
        return hit
    xs, ys = _tone_pts(float(toe), float(lift), float(shoulder))
    g = np.linspace(0.0, 100.0, 1024)
    try:
        from scipy.interpolate import PchipInterpolator
        y = np.asarray(PchipInterpolator(xs, ys)(g), np.float64)
    except Exception:                      # scipy 不在就退回折线（会有轻微折点，但不会崩）
        y = np.interp(g, xs, ys)
    y = np.clip(np.maximum.accumulate(y), 0.0, 100.0)      # 兜底：单调 + 有界
    _TONE_CACHE[key] = (g, y)
    return g, y


def _tone_on(p):
    if not p.get('tone_curve', False):
        return False
    return abs(float(p.get('tone_toe', 0.0) or 0.0)) > 1e-9 or \
        abs(float(p.get('tone_lift', 0.0) or 0.0)) > 1e-9 or \
        abs(float(p.get('tone_shoulder', 0.0) or 0.0)) > 1e-9


def tone_ref(disp_value, cfg=C, stock=None, base=None):
    """把"锁中灰的参照"也过一遍影调曲线。

    为什么必须这样：曲线抬了中灰，而 LOCK_MID 的职责是"把中灰拉回修正层交出来的那个值"。
    若参照不过曲线，锁就会把曲线刚抬起来的中灰**原样拉回去**（LIFT 白做）。
    单调变换下分位数可交换（median(T(L)) = T(median(L))），所以参照过完曲线之后，
    锁算出来的增益恰好 ≈ 1 —— 锁退化成"只清颜色块造成的残余漂移"，正是它该干的事。
    """
    p = stocks.color_params(cfg, stock, base)
    if not _tone_on(p):
        return float(disp_value)
    g, y = _tone_lut(p['tone_toe'], p['tone_lift'], p.get('tone_shoulder', 0.0))
    v = float(np.clip(disp_value, 0.0, 1.0))
    lab = color.to_lab(np.full((1, 1, 3), v, np.float64))
    lab[..., 0] = np.interp(lab[..., 0], g, y)
    return float(np.clip(color.from_lab(lab)[0, 0, 0], 0.0, 1.0))


def _builtin(disp, cfg, stock=None, base=None):
    """按"量出来的数"给颜色。一次 Lab 往返做完，顺序：

      ⓪ 线性域「雾」：黑位抬升（胶片黑不是死黑）—— 基准成色专属，卷不设
      ① 线性域 3x3（负片交调，默认恒等）
      ② ★ 胶片影调曲线（只动 L*）—— 先定影调，后面按新的 L* 上色
      ③ 色偏：整体 a*/b* + 按亮度分裂的 b*（暗部/亮部分开，对齐大师那把尺子）
      ④ 彩度：C' = s·Cref·(C/Cref)^p —— 两个自由度正好对上"彩度中位"和"彩度P90"两个靶
      ⑤ 明度对比（只动 L*，a/b 不动 → 对比不脏色）

    参数 = cfg 默认 → 基准成色（中性路径对齐大师平均）→ 卷（相对大师平均的性格偏移）。
    色偏在纯黑纯白两端淡出：纯黑不该有颜色，纯白也不该被染色（否则高光被染脏、还容易削顶）。
    """
    p = stocks.color_params(cfg, stock, base)

    lin = color.s2l(disp)
    fog = float(p.get('fog', 0.0) or 0.0)
    if fog > 0.0:
        # 线性光域抬黑：纯黑抬到 fog，白端不动（fog 很小，实测 0.0063 ≈ 黑位 L* 6.6）
        lin = lin + fog * np.clip(1.0 - lin, 0.0, None)
    M = np.asarray(p['matrix'], np.float64)
    if not np.allclose(M, np.eye(3), atol=1e-9):
        lin = lin @ M.T
    out = np.clip(color.l2s(np.clip(lin, 0.0, None)), 0.0, 1.0)

    # ★★ 09-14 新增三块（`film.py`）—— 【丙】串扰 / 【甲】分通道响应 / 【乙】密度引擎
    #   丙 甲 在**线性光**上做（物理上是乳剂与染料的耦合）；乙 直接产出完整成色。
    #   ★ **整体强度跟卷走**（`p['film_color_w']`）：`neutral` 给 0.0 ⇒ 回到逐位恒等。
    _fw = float(np.clip(p.get('film_color_w', 1.0), 0.0, 1.0))
    _sp = bool(getattr(cfg, 'LAYER_SPEED_ENABLE', False)) and _fw > 0.0
    _ct = bool(getattr(cfg, 'CROSSTALK_ENABLE', False)) and _fw > 0.0
    if _sp or _ct:
        _l = color.s2l(out)
        if _sp:
            _l = film.layer_speeds(_l, getattr(cfg, 'LAYER_SPEEDS', (1.0, 1.0, 1.0)),
                                   float(getattr(cfg, 'LAYER_SPEED_STRENGTH', 1.0)) * _fw, cfg)
        if _ct:
            _l = film.crosstalk(_l, float(getattr(cfg, 'CROSSTALK_AMOUNT', 0.0)) * _fw,
                                getattr(cfg, 'CROSSTALK_CROSSOVERS', (0.25, 0.55, 0.88)), cfg)
        out = np.clip(color.l2s(np.clip(_l, 0.0, None)), 0.0, 1.0)

    _dens_w = 0.0
    if bool(getattr(cfg, 'DENSITY_ENABLE', False)) and _fw > 0.0:
        _dens_w = float(np.clip(getattr(cfg, 'DENSITY_STRENGTH', 1.0), 0.0, 1.0)) * _fw
        if _dens_w > 0.0:
            _d = film.density(out, p.get('density_stock') or getattr(cfg, 'DENSITY_STOCK', 'portra400'),
                              cfg=cfg)
            # ★★ 09-14 修（逐层追踪抓到的）：**乙 在高光端必须淡出**。
            #   不加这一条，它会把**入口交出来的近白全部砍平**（实测 0304 12.63%→0、0774 15.76%→0），
            #   顺带把白区的层次压掉（0350 2.80→1.84）—— 表现就是"白的东西不白、发肉"。
            #   物理上也讲得通：密度曲线是**整段**的，但我们只是拿它当"中间调/暗部的成色"，
            #   高光该留给入口+影调曲线自己那条肩部。
            _lo = float(getattr(cfg, 'DENSITY_FADE_LO', 0.72))
            _hi = float(getattr(cfg, 'DENSITY_FADE_HI', 0.94))
            _fade = 1.0 - color.smoothstep(color.luma(out), _lo, _hi)
            out = film.blend_map(out, _d, _dens_w * _fade)

    has_tone = _tone_on(p)
    has_tint = (abs(p['a']) + abs(p['b']) + abs(p['b_sh']) + abs(p['b_hi'])) > 1e-6
    has_chroma = abs(p['chroma_p'] - 1.0) > 1e-6 or abs(p['chroma_s'] - 1.0) > 1e-6
    _ce = float(p.get('chroma_ends', 0.0) or 0.0)      # 抽色饱和（两端掉彩）
    has_chroma = has_chroma or _ce > 0.0
    has_contrast = abs(p['contrast'] - 1.0) > 1e-6
    if not (has_tone or has_tint or has_chroma or has_contrast):
        return np.clip(out, 0.0, 1.0)

    lab = color.to_lab(out)
    L = lab[..., 0]

    if has_tone:
        _g, _y = _tone_lut(p['tone_toe'], p['tone_lift'], p.get('tone_shoulder', 0.0))
        L = np.interp(L, _g, _y)
        lab[..., 0] = L

    if has_tint:
        w = color.smoothstep(L, p['tint_lo'], p['tint_hi'])            # 0 = 暗部, 1 = 亮部
        _flo = tuple(getattr(cfg, 'TINT_FADE_LO', (2.0, 8.0)))
        _fhi = tuple(getattr(cfg, 'TINT_FADE_HI', (97.0, 100.0)))
        fade = color.smoothstep(L, _flo[0], _flo[1]) * (1.0 - color.smoothstep(L, _fhi[0], _fhi[1]))
        lab[..., 1] += p['a'] * fade
        lab[..., 2] += (p['b'] + (1.0 - w) * p['b_sh'] + w * p['b_hi']) * fade

    if has_chroma:
        a2, b2 = lab[..., 1], lab[..., 2]
        C = np.sqrt(a2 * a2 + b2 * b2)
        Cr = float(p['chroma_ref'])
        Cn = p['chroma_s'] * Cr * np.power(np.maximum(C, 1e-6) / Cr, p['chroma_p'])
        k = np.where(C > 1e-6, Cn / np.maximum(C, 1e-6), 1.0)
        # ★ 抽色饱和（09-13 调研修正）：两端掉彩，中间调（L*=50）不动。
        #   公式出处 Emulsifier（`_debug/_rs/emul/engine.py`）：sat_mult = 1 - subsat*(2*luma-1)^2。
        #   和 L1 的 HILIGHT_DESAT 分工：那边管高光端，这里补上**暗部端**
        #   （09-13 园岭暗部泛紫红斑就是这一块）。
        if _ce > 0.0:
            Ln = np.clip(L, 0.0, 100.0) / 100.0
            k = k * (1.0 - _ce * (2.0 * Ln - 1.0) ** 2)
        lab[..., 1] = a2 * k
        lab[..., 2] = b2 * k

    if has_contrast:
        # S 形：以中灰为支点（u=0.5 处 sin=0 不动），暗部压下去、亮部提上来，两端点不动。
        # ⚠ 符号：+a·sin(2πu) 在暗部（u<0.5）是**加**值 ⇒ 那是"降对比"。
        #   要"加对比"必须**减**。这里曾经写成加号，导致所有卷的 contrast 方向反了
        #   （标定算出的 contrast>1 = 这条线反差比大师大，落地却在降对比）。
        _sc = float(getattr(cfg, 'CONTRAST_S_SCALE', 0.15))
        _cl = tuple(getattr(cfg, 'CONTRAST_S_CLAMP', (-0.10, 0.12)))
        a = float(np.clip((p['contrast'] - 1.0) * _sc, _cl[0], _cl[1]))
        u = np.clip(lab[..., 0] / 100.0, 0.0, 1.0)
        lab[..., 0] = np.clip(u - a * np.sin(2.0 * np.pi * u), 0.0, 1.0) * 100.0

    return np.clip(color.from_lab(lab), 0.0, 1.0)


def mid_of(disp):
    return float(np.percentile(color.gray_of(disp), C.PCT_MID))


def lock_mid(disp, ref_mid, max_gain=None):
    """把中灰分位拉回**修正层交出来的那个中灰**（不是全局靶）。

    这一步是"风格不许改回修正"的机械保证。注意参照物必须是上一层的实际输出，
    不是配置里的靶 —— 否则修正层因为限幅没到靶时，风格层会继续往上拽，
    等于风格层在偷偷做曝光补偿（白点被顶到 246 就是这么来的）。
    """
    if max_gain is None:
        max_gain = float(getattr(C, 'LOCK_MID_MAX_GAIN', 1.25))
    gm = mid_of(disp)
    if gm <= C.NOISE_FLOOR or ref_mid <= C.NOISE_FLOOR:
        return disp, 0.0
    k = float(color.s2l(ref_mid) / max(color.s2l(gm), C.NOISE_FLOOR))
    k = float(np.clip(k, 1.0 / max_gain, max_gain))
    if abs(np.log2(k)) < 0.004:
        return disp, 0.0
    out = color.l2s(np.clip(color.s2l(np.clip(disp, 0.0, 1.0)) * k, 0.0, None))
    return np.clip(out, 0.0, 1.0), float(np.log2(k))


def apply(disp, cfg=C, lut=None, lock_ref=None, stock=None, base=None):
    out = _builtin(disp, cfg, stock, base)

    if lut is not None:
        c = cube_apply(np.clip(out, 0.0, 1.0), lut)
        s = cfg.LUT_STRENGTH
        out = np.clip(out * (1.0 - s) + c * s, 0.0, 1.0)

    info = dict(lut=bool(lut is not None), lock_ev=0.0,
                stock=(stock or {}).get('name'),
                base=stocks.resolve_base(cfg, base)['name'])
    if cfg.LOCK_MID and lock_ref is not None:
        # ★ 参照也要过影调曲线 —— 否则锁会把曲线刚抬起来的中灰原样拉回去（LIFT 白做）。
        #   单调变换下 median(T(L)) = T(median(L))，所以过完曲线之后锁的增益恰好 ≈1，
        #   只清掉颜色块造成的残余漂移。
        ref = tone_ref(lock_ref, cfg, stock, base)
        out, ev = lock_mid(out, ref)
        info['lock_ev'] = ev
    return np.clip(out, 0.0, 1.0), info


# ---------------- 离线烘焙 ----------------
def bake_grid(size=33):
    """返回 (n^3, 3) 显示域格点（R 变化最慢，B 最快）与 reshape 用尺寸。"""
    a = np.linspace(0.0, 1.0, size)
    R, G, B = np.meshgrid(a, a, a, indexing='ij')
    return np.stack([R, G, B], axis=-1).reshape(-1, 3)


def bake_cube(path, size=33, cfg=C, chunk=4096, stock=None, base=None):
    """把 L2 层离线烘成 .cube（不含 LOCK_MID，那是逐图的）。

    给了 stock 就烘那一卷的颜色性格 —— 这样"卷"可以落成一个 .cube 文件，
    别的软件（达芬奇/PS）也能直接用同一套颜色。base 同理（基准成色也一起烘进去）。
    """
    grid = bake_grid(size)
    out = np.empty_like(grid)
    for i in range(0, grid.shape[0], chunk):
        blk = grid[i:i + chunk].reshape(1, -1, 3)
        o, _ = apply(blk, cfg, lock_ref=None, stock=stock, base=base)   # 烘焙时不做锁中灰（那是逐图的）
        out[i:i + chunk] = o.reshape(-1, 3)
    cube = out.reshape(size, size, size, 3)
    bn = stocks.base_label(cfg, base)
    title = ('svFilm L2 style / display domain'
             + (' / ' + stock['name'] if stock else '')
             + ((' / ' + bn) if bn != '无' else ''))
    return cube_write(path, cube, title=title)
