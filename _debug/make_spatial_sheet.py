# -*- coding: utf-8 -*-
"""空间域对照图：一张/几张照片，把一个一个空间效果单独打开，看它到底做了什么。

栏位：0 RAW 未修 → 1 出片·空间全关 → 2 +颗粒 → 3 +黑柔 → 4 +Halation → 5 三件全开
（颜色统一用 neutral，这样看到的变化只来自空间效果，不掺卷的颜色）

默认出 **1:1 原始像素切图** —— 颗粒这种东西缩图看不出来，必须看真实像素。
用 --no-crop 看整图（看黑柔/Halation 的大范围光晕，整图更直观）。

用法（在工程根目录执行）：
  python _debug/make_spatial_sheet.py --purpose 空间三件对照 --root D:/照片库/svFilm效果 <stem...>
  python _debug/make_spatial_sheet.py --purpose 空间_整图 --no-crop --cx 700 --cy 400 <stem...>
"""
import argparse
import concurrent.futures as cf
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from svFilm import color, config as C, io, paths, pipeline, spatial, stocks  # noqa: E402

FONT = 'C:/Windows/Fonts/msyh.ttc'
FONT_B = 'C:/Windows/Fonts/msyhbd.ttc'
CW, CH = 520, 347
HDR, FTR = 32, 26

_G = dict(enable=True, amount=0.030, size=1.2, chroma=0.22,
          skin_suppress=0.60, detail_suppress=0.30, dark_floor=0.03)
_B = dict(enable=True, amount=0.085, radius=22.0, thr_lo=0.74, thr_hi=0.93,
          warmth=0.30, veil=0.045)
_H = dict(enable=True, amount=0.130, radius=18.0, thr_lo=0.78, thr_hi=0.99,
          color=[1.000, 0.300, 0.120])

COLS = [
    ('0 RAW 未修', 'base'),
    ('1 出片·空间全关', 'none'),
    ('2 +颗粒', 'grain'),
    ('3 +黑柔', 'bloom'),
    ('4 +Halation', 'halation'),
    ('5 三件全开', 'all'),
]
TITLES = dict(COLS)
ALL_KEYS = [k for _, k in COLS]


def _font(size, bold=False):
    for p in ((FONT_B, FONT) if bold else (FONT, FONT_B)):
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default()


def _fake(spatial, boost=1.0):
    """neutral 卷（颜色恒等）+ 指定的空间效果；boost 只用于"夸张版"演示（把强度乘 k）。"""
    s = stocks.get('neutral')
    s['label'] = '空间域'
    sp = {}
    for k, v in spatial.items():
        v = dict(v)
        if 'amount' in v:
            v['amount'] = float(v['amount']) * float(boost)
        sp[k] = v
    s['spatial'] = sp
    return s


def _cell(u8, crop=True, cx=None, cy=None):
    im = Image.fromarray(u8)
    if crop:
        w, h = min(CW, im.width), min(CH, im.height)
        x = im.width // 2 - w // 2 if cx is None else max(0, min(cx - w // 2, im.width - w))
        y = im.height // 2 - h // 2 if cy is None else max(0, min(cy - h // 2, im.height - h))
        im = im.crop((x, y, x + w, y + h))
    if im.size != (CW, CH):
        r = max(CW / im.width, CH / im.height)
        im = im.resize((int(im.width * r) + 1, int(im.height * r) + 1), Image.LANCZOS)
        l, t = (im.width - CW) // 2, (im.height - CH) // 2
        im = im.crop((l, t, l + CW, t + CH))
    return im


def _tag(im, title, sub, hi=False):
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, CW, HDR], fill=(0, 0, 0))
    d.text((8, 5), title, font=_font(18, True), fill=(140, 220, 255) if hi else (255, 255, 255))
    d.rectangle([0, CH - FTR, CW, CH], fill=(0, 0, 0))
    d.text((8, CH - FTR + 3), sub, font=_font(14), fill=(255, 235, 150))
    return im


def _grain_e(u8_a, u8_b):
    """两栏之间的颗粒能量（高频差），用来证明"颗粒真的加上了"。"""
    a = u8_a.astype(np.float64)
    b = u8_b.astype(np.float64)
    return float(np.std(b - a))


def _auto_center(disp, cx=None, cy=None):
    """没给切图中心时自动找"皮肤最多的地方"—— 颗粒最容易在脸上看出脏不脏。
    带垂直先验：优先上半部分（人脸很少在最下面）。"""
    if cx is not None:
        return cx, cy
    import cv2
    m = spatial._skin_mask_fast(np.clip(disp, 0, 1))
    if float(m.max()) < 0.15:
        return None, None
    h = m.shape[0]
    ramp = 1.0 - color.smoothstep(np.arange(h, dtype=np.float64) / max(h - 1, 1), 0.55, 0.95)
    m = m * ramp[:, None]
    s = cv2.blur(np.ascontiguousarray(m, np.float32), (CW // 2, CH // 2))
    y, x = np.unravel_index(int(np.argmax(s)), s.shape)
    return int(x), int(y)


def _job(t):
    path, max_side, key, cx, cy, boost = t
    variant = {
        'base': None,
        'none': _fake({}, boost),
        'grain': _fake(dict(grain=dict(_G)), boost),
        'bloom': _fake(dict(bloom=dict(_B)), boost),
        'halation': _fake(dict(halation=dict(_H)), boost),
        'all': _fake(dict(grain=dict(_G), bloom=dict(_B), halation=dict(_H)), boost),
    }[key]
    res = pipeline.run(path, src='raw', max_side=max_side, stock=variant, keep_stages=True)
    d = res.sample.disp if key == 'base' else res.disp
    ax, ay = _auto_center(d, cx, cy)
    return (path, key, color.display_to_u8(d), _stats(d), None, (ax, ay))


def _stats(disp):
    g = color.gray_of(disp)
    lab = color.to_lab(np.clip(disp, 0, 1))
    return '中灰%d 黑%d 白%d 彩度%.0f' % (
        np.percentile(g, 50) * 255, np.percentile(g, C.PCT_BLACK) * 255,
        np.percentile(g, C.PCT_WHITE) * 255, color.chroma_c90(np.clip(disp, 0, 1)))


def build(stems, purpose, root=None, tag=None, date=None, max_side=2048,
          crop=True, cx=None, cy=None, jobs=4, keys=None, cell=None, boost=1.0):
    global CW, CH
    if cell:
        CW, CH = int(cell[0]), int(cell[1])
    keys = list(keys or ALL_KEYS)
    root = root or (os.path.dirname(os.path.abspath(os.path.splitext(stems[0])[0])) or '.')
    raws = []
    for st in stems:
        rp = io.find_pair(os.path.splitext(st)[0], prefer='raw')
        if io.kind_of(rp) != 'raw':
            print('跳过（没有 RAW）:', st)
            continue
        raws.append(rp)
    if not raws:
        print('没有可用的 RAW')
        return None

    tasks = [(rp, max_side, key, cx, cy, boost) for rp in raws for key in keys]
    print('共 %d 张渲染（%d 图 x %d 栏），并发 %d，强度 x%.1f' % (len(tasks), len(raws), len(keys), jobs, boost))
    sys.stdout.flush()

    got = {}
    with cf.ProcessPoolExecutor(max_workers=max(1, jobs)) as ex:
        for p, key, u8, su, _full, xy in ex.map(_job, tasks):
            got[(p, key)] = (u8, su, xy)
            print('  ok  %-14s %-9s %s' % (os.path.basename(p), key, su))
            sys.stdout.flush()

    ncol = len(keys)
    W = CW * ncol + 5 * (ncol - 1)
    lh = 34
    H = lh + len(raws) * (CH + 8)
    sheet = Image.new('RGB', (W, H), (18, 18, 18))
    d0 = date or paths.today()
    ImageDraw.Draw(sheet).text(
        (10, 7), 'svFilm 空间域对照 · %s   ——   颜色固定中性，只逐个开空间效果      [%s]'
        % (d0, purpose), font=_font(20, True), fill=(255, 255, 255))

    y = lh
    for rp in raws:
        base_u8 = got[(rp, 'base')][0] if (rp, 'base') in got else got[(rp, keys[0])][0]
        ax, ay = got[(rp, 'none' if (rp, 'none') in got else keys[0])][2]
        for i, key in enumerate(keys):
            u8, su, _xy = got[(rp, key)]
            extra = ''
            if key == 'grain':
                extra = '  颗粒能量%.1f' % _grain_e(base_u8, u8)
            sheet.paste(_tag(_cell(u8, crop, ax, ay), TITLES.get(key, key), su + extra,
                             key in ('all', 'grain')),
                        (i * (CW + 5), y))
        ImageDraw.Draw(sheet).text((8, y + CH - FTR + 2), os.path.basename(rp),
                                   font=_font(14), fill=(120, 255, 160))
        y += CH + 8

    out = paths.debug_path(purpose, (tag or purpose) + '.jpg', root, d0)
    sheet.save(out, quality=92)
    print('-> %s  (%d x %d)' % (out, W, H))
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stems', nargs='+')
    ap.add_argument('--purpose', required=True)
    ap.add_argument('--root', default=None)
    ap.add_argument('--tag', default=None)
    ap.add_argument('--date', default=None)
    ap.add_argument('--max-side', type=int, default=2048, dest='max_side')
    ap.add_argument('--cx', type=int, default=None)
    ap.add_argument('--cy', type=int, default=None)
    ap.add_argument('--no-crop', action='store_true', dest='no_crop')
    ap.add_argument('--cols', default=None, help='只渲染这些栏，逗号分隔：%s' % ','.join(ALL_KEYS))
    ap.add_argument('--cell', default=None, help='单元格尺寸 WxH，默认 520x347')
    ap.add_argument('--boost', type=float, default=1.0, help='把空间效果强度乘 k（做"夸张版"演示用）')
    ap.add_argument('--jobs', type=int, default=4)
    a = ap.parse_args()
    cell = tuple(int(v) for v in a.cell.lower().split('x')) if a.cell else None
    keys = [s.strip() for s in a.cols.split(',')] if a.cols else None
    build(a.stems, a.purpose, a.root, a.tag, a.date, a.max_side,
          not a.no_crop, a.cx, a.cy, a.jobs, keys, cell, a.boost)
