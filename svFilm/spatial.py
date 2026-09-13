# -*- coding: utf-8 -*-
"""空间域层 —— 颗粒 / 黑柔(Bloom) / Halation。塞不进 LUT 的三件。

为什么必须独立一层：L2 的 .cube 是"逐像素的颜色查找表"，只能装
"颜色对比 + 高光压缩"这类点运算。而下面三件事都是**邻域运算**：

  * 颗粒 Grain    —— 银盐颗粒的随机性，随亮度变化、随区域抑制
  * 黑柔 Bloom    —— 镜头/滤镜像差把亮部往外扩散（Black Pro Mist 的招牌）
  * Halation      —— 光穿过片基被反射回乳剂，在亮部周围散出红橙晕圈

位置：L2 之后、L3 之前。输入输出都是 disp 域。
强度由"卷"给（`stocks.py`），这里只负责算法；`resolve()` 负责把卷的强度
叠到 config 默认上。

三条纪律：
  1) 三个效果都**不改中灰**（加的量都在局部，且乘性/零均值）；
  2) 颗粒用**固定种子**，同一张图每次结果一致（可复现，别用随机种子）；
  3) 掩膜全部用"平滑单调量"，不用硬阈值 —— 硬阈值会在画面里留下边缘。
"""
from __future__ import annotations

import numpy as np

from . import color
from . import config as C


def resolve(cfg=C, stock=None):
    """把卷的空间参数叠到 config 默认上（卷优先），返回三个 dict。"""
    p = dict(
        grain=dict(enable=cfg.GRAIN_ENABLE, amount=cfg.GRAIN_AMOUNT, size=cfg.GRAIN_SIZE,
                   chroma=cfg.GRAIN_CHROMA, skin_suppress=cfg.GRAIN_SKIN_SUPPRESS,
                   detail_suppress=cfg.GRAIN_DETAIL_SUPPRESS, dark_floor=cfg.GRAIN_DARK_FLOOR),
        bloom=dict(enable=cfg.BLOOM_ENABLE, amount=cfg.BLOOM_AMOUNT, radius=cfg.BLOOM_RADIUS,
                   thr_lo=cfg.BLOOM_THR_LO, thr_hi=cfg.BLOOM_THR_HI,
                   warmth=cfg.BLOOM_WARMTH, veil=cfg.BLOOM_VEIL,
                   spread=cfg.BLOOM_SPREAD),
        halation=dict(enable=cfg.HALATION_ENABLE, amount=cfg.HALATION_AMOUNT,
                      radius=cfg.HALATION_RADIUS, thr_lo=cfg.HALATION_THR_LO,
                      thr_hi=cfg.HALATION_THR_HI, color=list(cfg.HALATION_COLOR),
                      radius_ratios=list(cfg.HALATION_RADIUS_RATIOS)),
    )
    if stock:
        sp = stock.get('spatial') or {}
        for k, v in sp.items():
            if k in p:
                p[k].update(v)
    return p


# ---------------- 基础件 ----------------
def _blur(a, sigma):
    """高斯模糊（float64 进、float64 出）。

    大半径（>4px）走"先缩图 -> 小核模糊 -> 再放大"：sigma 20 直接算核要 ~170 阶，
    又慢又没必要 —— 辉光本来就是低频，缩一半再糊肉眼看不出差别，快好几倍。
    """
    if sigma is None or sigma <= 0:
        return np.asarray(a, np.float64)
    import cv2
    x = np.ascontiguousarray(a, dtype=np.float32)
    f = 1.0
    if sigma > 4.0:
        f = sigma / 2.0
        h, w = x.shape[:2]
        nw, nh = max(1, int(round(w / f))), max(1, int(round(h / f)))
        x = cv2.resize(x, (nw, nh), interpolation=cv2.INTER_AREA)
        sigma = sigma / f
    y = cv2.GaussianBlur(x, (0, 0), float(sigma))
    if f != 1.0:
        h, w = a.shape[:2]
        y = cv2.resize(y, (w, h), interpolation=cv2.INTER_LINEAR)
    return y.astype(np.float64)


def _local_detail(disp):
    """高频能量（0~1，按 P95 归一）。用来知道"这块本来就纹理多"。"""
    g = color.gray_of(disp)
    hi = g - _blur(g, 2.0)
    n = float(np.percentile(np.abs(hi), 95)) + 1e-9
    return np.clip(np.abs(hi) / n, 0.0, 1.0)


def _skin_mask_fast(disp):
    """肤色软掩膜（0~1）。故意不用 Lab（全图 Lab 要 0.8s），
    用显示域的"红>绿>蓝 + 有彩度"近似 —— 只用来决定"这里少撒点颗粒"，不需要精确。"""
    r = disp[..., 0]
    g = disp[..., 1]
    b = disp[..., 2]
    m = color.smoothstep(r - b, 0.035, 0.150)          # 红明显高于蓝
    m *= color.smoothstep(r - g, 0.010, 0.080)         # 红高于绿
    m *= 1.0 - color.smoothstep(color.sat_hsv(disp), 0.62, 0.92)   # 太艳的（衣服/花）不算
    m *= color.smoothstep(color.gray_of(disp), 0.10, 0.22)         # 太暗不算
    return m


# ---------------- 乙：颗粒 ----------------
def grain(disp, p, cfg=C):
    if not p.get('enable') or p.get('amount', 0.0) <= 0.0:
        return np.clip(disp, 0.0, 1.0), dict(applied=False)

    cur = np.clip(disp, 0.0, 1.0)
    g = color.gray_of(cur)
    h, w = g.shape

    # 亮度包络：中间调最明显，两端收（胶片就是这样，不是均匀撒盐）
    env = np.clip(4.0 * g * (1.0 - g), 0.0, 1.0) ** 0.55
    env *= color.smoothstep(g, float(p.get('dark_floor', 0.03)), float(p.get('dark_floor', 0.03)) + 0.07)
    # ★ 高光端**精确归零**（09-13 调研修 bug）：原来只压 0.70，白墙/天空还留 30% 颗粒在动。
    #   物理：密度饱和区没有可显影的银盐。外面（LIMO `applyGrainAsExposure`、
    #   Emulsifier `grain_mask=(luma^0.5)(1-luma)^1.5`）两端都精确为 0。
    env *= 1.0 - color.smoothstep(g, float(cfg.GRAIN_HI_LO), float(cfg.GRAIN_HI_HI))

    # 区域抑制：脸和高细节处少撒
    if p.get('skin_suppress', 0.0) > 0.0:
        env *= 1.0 - float(p['skin_suppress']) * _skin_mask_fast(cur)
    if p.get('detail_suppress', 0.0) > 0.0:
        env *= 1.0 - float(p['detail_suppress']) * _local_detail(cur)

    # 噪声：单色为主 + 一点彩噪；高斯模糊出"颗粒尺度"，再归一化回 std≈1
    rng = np.random.default_rng(int(cfg.GRAIN_SEED))
    sigma = float(p.get('size', 1.2))
    mono = _blur(rng.normal(0.0, 1.0, (h, w)), sigma)
    mono /= float(mono.std()) + 1e-9
    ck = float(p.get('chroma', 0.0))
    if ck > 0.0:
        col = np.stack([_blur(rng.normal(0.0, 1.0, (h, w)), sigma) for _ in range(3)], axis=-1)
        col /= float(col.std()) + 1e-9
        col -= col.mean(axis=-1, keepdims=True)          # 去掉亮度分量：只留"彩"
        n = mono[..., None] * (1.0 - ck) + col * ck
    else:
        n = mono[..., None]

    # 乘性叠加：黑还是黑（0 乘任何数还是 0），中灰也不会被整体推走
    out = cur * (1.0 + float(p['amount']) * env[..., None] * n)
    out = np.clip(out, 0.0, 1.0)
    return out, dict(applied=True, amount=float(p['amount']),
                     env_mean=float(env.mean()), seed=int(cfg.GRAIN_SEED))


# ---------------- 丙：黑柔 / Bloom ----------------
def bloom(disp, p, cfg=C):
    """黑柔 = 三个**独立**机理，全部在**亮度域**算，只动亮度不改色相。

    ★★ 为什么必须走亮度域（09-13 修，SV 报「DSCF2638 加完后脸变色」）：
    老实现是 `out = lin + amount * blur(lin * mask)` —— 把**模糊后的彩色光**直接加到画面上。
    后果：辉光把**周围亮物的颜色搬到别处**。实测那张片子里一个红色抱枕的红光被糊到整张脸和
    画面上（脸区 a* 掉、b* 涨，视觉上是粉红雾）。
    参考实现都不是这么干的：
      * LIMO `blendBloom`：提取高光时按**自身颜色**提饱和（`mix(luma,rgb,1.05)`）再 **screen 混合**
        （`1-(1-a)(1-b)`），注释写明"preserving highlight energy"；
      * Emulsifier `engine.py`：**模糊的是"掩膜"（0~1 标量）而不是"光"**，再把模糊后的掩膜
        当**中性量**加回去 ⇒ 基本不改色相。
    ⇒ 这里统一成：**先把整件事在亮度 Y 上算完，再按同一个增量/增益作用回三通道**
      （与 `tone.py`、`io.apply_entry_curve` 同一条契约：「曲线/增益只作用在亮度上，三通道同步」）。
    顺带：模糊从"逐通道 3 次"降到"亮度 1 次"，spatial 更快。

    三个机理：
      * `amount` 加性辉光 —— 亮部往外**加**光（只加光）。
      * `spread` 化开     —— 把高光掩膜区**自己的**能量扣掉、由模糊版补上；`amount==spread` 时守恒。
      * `veil`   面纱     —— 整幅往模糊版靠（零均值：抬暗部、压高光）。
    `warmth` 是**唯一**会改色的旋钮（只给"加进来的那部分光"染色），默认建议留着 0。
    """
    amt = float(p.get('amount', 0.0))
    veil = float(p.get('veil', 0.0))
    spread = float(p.get('spread', 0.0))
    # ★ 门控修错（09-13）：以前是 `amount <= 0` 一刀切关掉**整个函数**，
    #   于是"只开面纱（veil>0 而 amount=0）"根本进不来 —— 实测黑柔阶梯里
    #   「只面纱」那一档与「关」逐位相同，白扫了一档。
    #   三个机理是独立的，门控改成"三个都为 0 才跳过"。
    if not p.get('enable') or (amt <= 0.0 and veil <= 0.0 and spread <= 0.0):
        return np.clip(disp, 0.0, 1.0), dict(applied=False)

    cur = np.clip(disp, 0.0, 1.0)
    lin = color.s2l(cur)
    g = color.gray_of(cur)
    radius = float(p.get('radius', 22.0))

    Y = color.luma(lin)                                   # 线性亮度（标量场）
    m = color.smoothstep(g, float(p.get('thr_lo', 0.74)), float(p.get('thr_hi', 0.93)))
    hotY = Y * m                                          # 亮部自己的能量（标量）

    Yg = Y
    if amt > 0.0:
        Yg = Yg + amt * _blur(hotY, radius)               # ① 加性辉光
    if spread > 0.0:
        Yg = Yg - spread * hotY                           # ② 化开（核心峰值下降 / 外圈散出去）
    if veil > 0.0:
        Yg = Yg + veil * (_blur(Yg, radius * 0.55) - Yg)  # ③ 面纱（零均值：抬暗部、压高光）

    dY = Yg - Y
    dY = np.maximum(dY, -Y)                               # 不许把亮度扣成负的
    if dY.size and float(np.max(np.abs(dY))) < 1e-12:
        return cur, dict(applied=False)

    # `warmth`：**唯一**改色的地方 —— 只给"加进来的那部分光"染色（0 = 完全中性）
    wa = float(p.get('warmth', 0.0))
    tint = np.array([1.0 + 0.10 * wa, 1.0 + 0.02 * wa, 1.0 - 0.10 * wa])
    out_lin = lin + dY[..., None] * tint

    out = np.clip(color.l2s(out_lin), 0.0, 1.0)
    return out, dict(applied=True, amount=amt, radius=radius, veil=veil, spread=spread,
                     warmth=wa, dY_mean=float(np.mean(dY)),
                     dY_p99=float(np.percentile(dY, 99)),
                     dY_p1=float(np.percentile(dY, 1)))


# ---------------- 丁：Halation ----------------
def halation(disp, p, cfg=C):
    if not p.get('enable') or p.get('amount', 0.0) <= 0.0:
        return np.clip(disp, 0.0, 1.0), dict(applied=False)

    cur = np.clip(disp, 0.0, 1.0)
    lin = color.s2l(cur)
    g = color.gray_of(cur)

    # 亮部才有能量；晕圈出现在亮区的**外侧**（片基把光散回去）
    bright = color.smoothstep(g, float(p.get('thr_lo', 0.78)), float(p.get('thr_hi', 0.99)))
    src = bright[..., None] * lin

    # ★ 分通道扩散半径（09-13 调研修正）：红光穿透片基散射得最远，蓝光几乎不散。
    #   出处：LIMO `FilmShaderCommon.h` RED/GREEN/BLUE_PENETRATION = 0.88/0.10/0.02；
    #         spektrafilm 也是三通道各自独立的散射 sigma。
    #   ⇒ 红边变成"外圈红、里层偏白"，而不是把亮部整块叠一层橙。
    r0 = float(p.get('radius', 18.0))
    ratios = p.get('radius_ratios') or [1.0, 0.45, 0.15]
    spread = np.stack([_blur(src[..., i], r0 * float(ratios[i])) for i in range(3)], axis=-1)
    ring = spread * (1.0 - np.clip(bright * 1.25, 0.0, 1.0))[..., None]

    col = np.asarray(p.get('color', [1.0, 0.30, 0.12]), np.float64)
    out_lin = lin + float(p['amount']) * ring * col
    out = np.clip(color.l2s(out_lin), 0.0, 1.0)
    return out, dict(applied=True, amount=float(p['amount']), radius=r0,
                     radius_ratios=[float(x) for x in ratios])


# ---------------- 编排 ----------------
def apply(disp, cfg=C, stock=None):
    """按 光学 → 乳剂的物理顺序：Halation（片基）→ Bloom（镜头）→ Grain（银盐）。"""
    p = resolve(cfg, stock)
    cur = np.clip(disp, 0.0, 1.0)
    cur, h_info = halation(cur, p['halation'], cfg)
    cur, b_info = bloom(cur, p['bloom'], cfg)
    cur, g_info = grain(cur, p['grain'], cfg)
    info = dict(halation=h_info, bloom=b_info, grain=g_info,
                any=bool(h_info['applied'] or b_info['applied'] or g_info['applied']),
                stock=(stock or {}).get('name'))
    return cur, info
