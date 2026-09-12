# -*- coding: utf-8 -*-
"""入口关口 + 出口。**整个工程里唯一允许出现"机型"三个字的地方。**

入口要做完三件事，做完之后下游不许再关心这张图是谁拍的：
  1) 解码：RAW 走 rawpy（线性、相机白平衡、关自动亮度）；JPG 走 PIL。
  2) IDT（输入设备变换）：白平衡 + 色彩矩阵 + 黑白电平 —— RAW 由 rawpy 一次做完；
     JPG 已经在相机里做完，拿不回来，所以是"显式降级"。
  3) 归一：对外只吐 lin（白点 1.0）、disp（0~1）两个域。
"""
from __future__ import annotations

import io as _stdlib_io
import os

import numpy as np
from PIL import Image, ImageOps

from . import color
from . import config as C

try:
    import cv2
except Exception:      # pragma: no cover
    cv2 = None

RAW_EXT = {'.raf', '.arw', '.cr2', '.cr3', '.nef', '.dng', '.orf', '.rw2', '.pef', '.srw', '.raw'}
STD_EXT = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.webp'}


class Sample:
    __slots__ = ('lin', 'disp', 'kind', 'path', 'name', 'exif', 'cam')

    def __init__(self, lin, disp, kind, path, exif=None, cam=None):
        self.lin = lin
        self.disp = disp
        self.kind = kind
        self.path = path
        self.name = os.path.splitext(os.path.basename(path))[0]
        self.exif = exif
        self.cam = cam if cam is not None else {}


def kind_of(path):
    e = os.path.splitext(path)[1].lower()
    if e in RAW_EXT:
        return 'raw'
    if e in STD_EXT:
        return 'std'
    raise ValueError('不认识的扩展名: %s' % path)


def _resize(arr, max_side):
    """缩到长边 max_side。大图先缩后转浮点 —— 40MP 转 float32 再缩要多花好几秒。"""
    h, w = arr.shape[:2]
    long_side = max(h, w)
    if max_side is None or long_side <= max_side:
        return arr
    s = float(max_side) / float(long_side)
    nw, nh = int(round(w * s)), int(round(h * s))
    if nw < 1 or nh < 1:
        raise ValueError('缩得太小')
    if cv2 is not None and arr.dtype == np.uint16:
        return cv2.resize(arr, (nw, nh), interpolation=cv2.INTER_AREA)
    if cv2 is not None:
        return cv2.resize(arr.astype(np.float32), (nw, nh),
                          interpolation=cv2.INTER_AREA).astype(np.float64)
    ch = [np.asarray(Image.fromarray(arr[..., i].astype(np.float32), mode='F')
                     .resize((nw, nh), Image.BILINEAR), np.float64) for i in range(arr.shape[-1])]
    return np.stack(ch, axis=-1)


def _exif_orientation_fixed(im):
    """exif_transpose 之后把 orientation 标签抹成 1，否则存回去会被转两次。"""
    ex = im.getexif()
    if not ex:
        return None
    if 274 in ex:
        ex[274] = 1
    try:
        return ex.tobytes()
    except Exception:
        return None


def idt_wb(lin, cfg=C):
    """IDT 的最后一步：白平衡。默认关。

    为什么默认关：RAW 走相机白平衡、JPG 是相机已经定死的，绝大多数情况本来就是对的。
    乱做自动白平衡会把黄金时刻的暖调当成"偏色"抹平。
    只在输入明显拍歪（白炽灯下没改色温）时手动打开。

    做法（灰世界，但只在"近中性像素"上算）：取低饱和度像素的线性均值，
    求一组限幅增益。不做则返回原图。
    """
    if not cfg.WB_ENABLE:
        return lin, dict(applied=False)
    disp = np.clip(color.l2s(np.clip(lin, 0.0, None)), 0.0, 1.0)
    s = color.sat_hsv(disp)
    g = color.gray_of(disp)
    sel = (s < cfg.WB_NEUTRAL_SAT) & (g > 0.05) & (g < 0.95)
    cov = float(np.mean(sel))
    if cov < cfg.WB_MIN_COVER:
        return lin, dict(applied=False, cover=cov)
    m = lin[sel].mean(axis=0)
    m = np.maximum(m, cfg.NOISE_FLOOR)
    gain = float(m.mean()) / m
    gain = 1.0 + (gain - 1.0) * float(cfg.WB_STRENGTH)
    gain = np.clip(gain, 1.0 - cfg.WB_MAX_GAIN, 1.0 + cfg.WB_MAX_GAIN)
    return lin * gain, dict(applied=True, cover=cov, gain=[float(v) for v in gain])


def load_std(path, max_side=C.MAX_SIDE):
    im = Image.open(path)
    im = ImageOps.exif_transpose(im).convert('RGB')
    arr = np.asarray(im, np.float64) / 255.0
    arr = _resize(arr, max_side)
    disp = np.clip(arr, 0.0, 1.0)
    lin, wb_info = idt_wb(color.s2l(disp))
    disp = np.clip(color.l2s(lin), 0.0, 1.0)
    return Sample(lin, disp, 'jpg', path, _exif_orientation_fixed(im),
                  dict(kind='jpg', wb=wb_info, note='无 IDT 余量（相机曲线已压过）'))


def load_raw(path, max_side=C.MAX_SIDE):
    """RAW 解码 + IDT（白平衡 / 色彩矩阵由 rawpy 完成；基线增益查机型表，加机型不改代码）。"""
    import rawpy
    from . import cameras

    kw = dict(C.RAW_DECODE)
    exif = None
    make = model = None
    with rawpy.imread(path) as raw:      # 一次打开：解码 + 缩略图（拿 EXIF 和机型）
        try:
            th = raw.extract_thumb()
            if th.format == rawpy.ThumbFormat.JPEG:
                tmp = ImageOps.exif_transpose(Image.open(_stdlib_io.BytesIO(th.data)))
                ex = tmp.getexif()
                make = str(ex.get(271) or '') or None
                model = str(ex.get(272) or '') or None
                exif = _exif_orientation_fixed(tmp)
        except Exception:
            pass
        rgb = raw.postprocess(use_camera_wb=kw['use_camera_wb'],
                              no_auto_bright=kw['no_auto_bright'],
                              output_bps=kw['output_bps'],
                              gamma=kw['gamma'],
                              half_size=kw['half_size'],
                              output_color=rawpy.ColorSpace.sRGB)

    rgb = _resize(rgb, max_side)           # 先在 uint16 上缩，省一次 40MP 的浮点转换
    lin = rgb.astype(np.float64) / 65535.0
    lin = np.clip(lin, 0.0, None)          # 只挡负值，不夹上限（高光余量要留着给 L1）

    cam = dict(cameras.lookup(make, model), make=make, model=model)
    gain = float(2.0 ** float(cam['baseline_ev']))
    if gain != 1.0:
        lin = lin * gain

    lin, wb_info = idt_wb(lin)
    disp = np.clip(color.l2s(lin), 0.0, 1.0)
    cam = dict(cam, wb=wb_info)
    return Sample(lin, disp, 'raw', path, exif, cam)


def load(path, max_side=C.MAX_SIDE, src=None):
    """src: None/'auto' | 'raw' | 'jpg'（jpg = 走解码器出来的显示域，不是机内 JPEG）"""
    k = kind_of(path)
    if src in (None, 'auto'):
        src = 'raw' if k == 'raw' else 'jpg'
    if src == 'raw':
        if k != 'raw':
            raise ValueError('要求 RAW，但 %s 不是 RAW' % path)
        return load_raw(path, max_side)
    return load_std(path, max_side)


def find_pair(path, prefer='raw'):
    """给一个 stem 或任一路径，找同名的 RAW/JPG 对。返回实际存在的路径。"""
    stem, ext = os.path.splitext(path)
    ext = ext.lower()
    if prefer == 'raw' and ext in RAW_EXT:
        return path
    if prefer == 'jpg' and ext in STD_EXT:
        return path
    order = (RAW_EXT, STD_EXT) if prefer == 'raw' else (STD_EXT, RAW_EXT)
    for exts in order:
        for e in sorted(exts):
            cand = stem + e
            if os.path.exists(cand):
                return cand
    return path


def save(disp, path, exif=None, quality=C.JPEG_QUALITY):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
    im = Image.fromarray(color.display_to_u8(disp))
    kw = dict(quality=int(quality), subsampling=0, optimize=True)
    if exif:
        kw['exif'] = exif
    im.save(path, **kw)
    return path
