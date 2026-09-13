# -*- coding: utf-8 -*-
"""svFilm 色彩基元。全部自实现，不引用任何既有管线代码。

三代域定义（全项目只允许这三个名字）：
  * lin  线性域  : 场景线性 sRGB 基色，白点 = 1.0，中灰 ≈ 0.18
  * disp 显示域  : 0~1 编码 sRGB（.cube LUT 的常用输入域）
  * L*   Lab 明度: 0~100（只用于"看"，不用于"算"）

图像统一 float64 ndarray，形状 (H, W, 3)。
"""
import numpy as np

# ---------- 矩阵 ----------
M_RGB2XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])
M_XYZ2RGB = np.linalg.inv(M_RGB2XYZ)
WP = np.array([0.95047, 1.00000, 1.08883])

# Rec.709 亮度权重
W_LUMA = np.array([0.2126, 0.7152, 0.0722])

_EPS = 1e-12
_DELTA = 6.0 / 29.0


# ---------- 传递函数 ----------
def s2l(x):
    """disp -> lin"""
    x = np.asarray(x, np.float64)
    return np.where(x <= 0.04045, x / 12.92, ((np.maximum(x, 0.0) + 0.055) / 1.055) ** 2.4)


def l2s(y):
    """lin -> disp（不做上限裁剪，便于检出错值）"""
    y = np.clip(np.asarray(y, np.float64), 0.0, None)
    return np.where(y <= 0.0031308, y * 12.92, 1.055 * y ** (1.0 / 2.4) - 0.055)


# ---------- 读写 uint8 ----------
def u8_to_display(u8):
    return np.asarray(u8, np.float64) / 255.0


def display_to_u8(d):
    return np.clip(np.rint(np.asarray(d, np.float64) * 255.0), 0, 255).astype(np.uint8)


# ---------- 亮度 / 灰度 ----------
def luma(rgb):
    """线性权重亮度。lin 域调用得到线性亮度 Y；disp 域调用得到"灰度"。"""
    r = np.asarray(rgb, np.float64)
    return r[..., 0] * W_LUMA[0] + r[..., 1] * W_LUMA[1] + r[..., 2] * W_LUMA[2]


def Y_of(lin):
    """lin -> 线性亮度 Y"""
    return luma(lin)


def gray_of(disp):
    """disp -> 显示域灰度（所有阈值判据统一用这把尺子，单位 = 0~1）"""
    return luma(disp)


def gray255(disp):
    """disp -> 0~255 灰度（给人看）"""
    return luma(disp) * 255.0


# ---------- Lab ----------
def f_lab(t):
    t = np.asarray(t, np.float64)
    return np.where(t > _DELTA ** 3, np.cbrt(np.maximum(t, 0.0)), t / (3.0 * _DELTA ** 2) + 4.0 / 29.0)


def f_lab_inv(t):
    t = np.asarray(t, np.float64)
    return np.where(t > _DELTA, t ** 3, 3.0 * _DELTA ** 2 * (t - 4.0 / 29.0))


def to_lab(disp):
    """disp -> Lab，返回 (...,3) = [L, a, b]"""
    lin = s2l(np.clip(np.asarray(disp, np.float64), 0.0, 1.0))
    xyz = (lin @ M_RGB2XYZ.T) / WP
    f = f_lab(xyz)
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def from_lab(lab):
    """Lab -> disp（可能越界，调用方负责裁剪）"""
    lab = np.asarray(lab, np.float64)
    fy = (lab[..., 0] + 16.0) / 116.0
    fx = fy + lab[..., 1] / 500.0
    fz = fy - lab[..., 2] / 200.0
    xyz = np.stack([f_lab_inv(fx), f_lab_inv(fy), f_lab_inv(fz)], axis=-1) * WP
    lin = xyz @ M_XYZ2RGB.T
    return l2s(np.clip(lin, 0.0, None))


def L_of_lin(lin):
    """线性亮度 Y -> L*"""
    return 116.0 * f_lab(np.clip(lin, 0.0, None)) - 16.0


def lin_of_L(L):
    """L* -> 线性亮度 Y"""
    return f_lab_inv((np.asarray(L, np.float64) + 16.0) / 116.0)


def retone_L(lin, L_new):
    """★ **只改明度**：把线性图按"新的 L* 映射"重打一遍。

    做法 = 每个像素按 `Y_new / Y_old` **同步缩放 RGB** ⇒ 色相完全不动、
    彩度关系也不动（这正是"只动亮暗、不动颜色"该有的形态；换 Lab 改 L* 会顺带改彩度）。
    `L_new` 与 `lin` 同形（或标量）。返回 disp 域。
    """
    lin = np.asarray(lin, np.float64)
    Y = np.maximum(Y_of(lin), _EPS)
    k = lin_of_L(np.clip(L_new, 0.0, 100.0)) / Y
    return np.clip(l2s(lin * k[..., None]), 0.0, 1.0)


def chroma(lab):
    return np.sqrt(lab[..., 1] ** 2 + lab[..., 2] ** 2)


def chroma_c90(disp):
    return float(np.percentile(chroma(to_lab(disp)), 90))


# ---------- 分位 ----------
def pct_of(arr, pcts):
    """返回 {pct: value}，判据统一入口"""
    return {float(p): float(np.percentile(arr, p)) for p in pcts}


# ---------- 其它 ----------
def sat_hsv(disp):
    """HSV 的 S（找近中性像素用）"""
    r = np.asarray(disp, np.float64)
    mx = r.max(axis=-1)
    mn = r.min(axis=-1)
    return np.where(mx > _EPS, (mx - mn) / np.maximum(mx, _EPS), 0.0)


def smoothstep(x, lo, hi):
    """0~1 平滑阶跃，lo<hi"""
    t = (np.asarray(x, np.float64) - lo) / max(hi - lo, _EPS)
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def hue_deg(lab):
    """Lab 色相角，0~360"""
    a = lab[..., 1]
    b = lab[..., 2]
    return (np.degrees(np.arctan2(b, a)) + 360.0) % 360.0
