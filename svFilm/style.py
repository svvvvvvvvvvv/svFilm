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
def _builtin(disp, cfg, stock=None, base=None):
    """按"量出来的数"给颜色。一次 Lab 往返做完，顺序：

      ⓪ 线性域「雾」：黑位抬升（胶片黑不是死黑）—— 基准成色专属，卷不设
      ① 线性域 3x3（负片交调，默认恒等）
      ② 色偏：整体 a*/b* + 按亮度分裂的 b*（暗部/亮部分开，对齐大师那把尺子）
      ③ 彩度：C' = s·Cref·(C/Cref)^p —— 两个自由度正好对上"彩度中位"和"彩度P90"两个靶
      ④ 明度对比（只动 L*，a/b 不动 → 对比不脏色）

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

    has_tint = (abs(p['a']) + abs(p['b']) + abs(p['b_sh']) + abs(p['b_hi'])) > 1e-6
    has_chroma = abs(p['chroma_p'] - 1.0) > 1e-6 or abs(p['chroma_s'] - 1.0) > 1e-6
    _ce = float(p.get('chroma_ends', 0.0) or 0.0)      # 抽色饱和（两端掉彩）
    has_chroma = has_chroma or _ce > 0.0
    has_contrast = abs(p['contrast'] - 1.0) > 1e-6
    if not (has_tint or has_chroma or has_contrast):
        return np.clip(out, 0.0, 1.0)

    lab = color.to_lab(out)
    L = lab[..., 0]

    if has_tint:
        w = color.smoothstep(L, p['tint_lo'], p['tint_hi'])            # 0 = 暗部, 1 = 亮部
        fade = color.smoothstep(L, 2.0, 8.0) * (1.0 - color.smoothstep(L, 97.0, 100.0))
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
        a = float(np.clip((p['contrast'] - 1.0) * 0.15, -0.10, 0.12))
        u = np.clip(lab[..., 0] / 100.0, 0.0, 1.0)
        lab[..., 0] = np.clip(u - a * np.sin(2.0 * np.pi * u), 0.0, 1.0) * 100.0

    return np.clip(color.from_lab(lab), 0.0, 1.0)


def mid_of(disp):
    return float(np.percentile(color.gray_of(disp), C.PCT_MID))


def lock_mid(disp, ref_mid, max_gain=1.25):
    """把中灰分位拉回**修正层交出来的那个中灰**（不是全局靶）。

    这一步是"风格不许改回修正"的机械保证。注意参照物必须是上一层的实际输出，
    不是配置里的靶 —— 否则修正层因为限幅没到靶时，风格层会继续往上拽，
    等于风格层在偷偷做曝光补偿（白点被顶到 246 就是这么来的）。
    """
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
        out, ev = lock_mid(out, lock_ref)
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
