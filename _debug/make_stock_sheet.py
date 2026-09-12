# -*- coding: utf-8 -*-
"""卷对照图：同一批照片，一行一张、一列一个卷（外加第一列"RAW 未修"当底）。

看的就是"甲"：不同胶片卷给人的感觉差在哪 —— 肤色、绿、蓝、彩度、颗粒、光晕。

产出落到 <root>/效果debug/<日期>/<目的>/<标签>.jpg

用法（在工程根目录执行）：
  python _debug/make_stock_sheet.py --purpose 卷对照 --root D:/照片库/svFilm效果 D:/照片库/svFilm效果/DSCF2328.RAF
  python _debug/make_stock_sheet.py --purpose 卷对照 --stocks portra400 ektar100 <stem...>
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


def _font(size, bold=False):
    for p in ((FONT_B, FONT) if bold else (FONT, FONT_B)):
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default()


def _cell(u8, crop=None):
    im = Image.fromarray(u8)
    if crop is not None:
        # 100% 细节：按归一化中心切一个 CW x CH 的原尺寸窗口（各列同一切法）
        cx, cy = crop
        W, H = im.size
        if W >= CW and H >= CH:
            x = int(W * cx - CW / 2.0)
            y = int(H * cy - CH / 2.0)
            x = max(0, min(W - CW, x))
            y = max(0, min(H - CH, y))
            return im.crop((x, y, x + CW, y + CH))
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


def _stats(disp):
    g = color.gray_of(disp)
    lab = color.to_lab(np.clip(disp, 0, 1))
    a = np.percentile(lab[..., 1], 50)
    b = np.percentile(lab[..., 2], 50)
    return '中灰%d L*%.0f 白%d 彩度%.0f 色度a%.0f/b%.0f' % (
        np.percentile(g, 50) * 255, np.percentile(lab[..., 0], 50),
        np.percentile(g, C.PCT_WHITE) * 255, color.chroma_c90(np.clip(disp, 0, 1)), a, b)


def _auto_center_norm(disp):
    """自动找"皮肤最多的地方"（归一化坐标），颗粒/光晕最容易在脸上看出差别。"""
    import cv2
    m = spatial._skin_mask_fast(np.clip(disp, 0, 1))
    if float(m.max()) < 0.15:
        return None
    h = m.shape[0]
    ramp = 1.0 - color.smoothstep(np.arange(h, dtype=np.float64) / max(h - 1, 1), 0.55, 0.95)
    m = m * ramp[:, None]
    s = cv2.blur(np.ascontiguousarray(m, np.float32), (CW // 2, CH // 2))
    y, x = np.unravel_index(int(np.argmax(s)), s.shape)
    return (float(x) / m.shape[1], float(y) / m.shape[0])


def _job(t):
    path, max_side, stock, is_base = t
    res = pipeline.run(path, src='raw', max_side=max_side,
                       stock=(None if is_base else stock), keep_stages=True)
    d = res.sample.disp if is_base else res.disp
    ctr = _auto_center_norm(d) if is_base else None
    return (path, ('base' if is_base else stock), color.display_to_u8(d), _stats(d), ctr)


def build(stems, purpose, stock_list, root=None, tag=None, date=None,
          max_side=1200, jobs=4, crop=None):
    root = root or (os.path.dirname(os.path.abspath(os.path.splitext(stems[0])[0])) or '.')
    raw_paths = []
    for st in stems:
        rp = io.find_pair(os.path.splitext(st)[0], prefer='raw')
        if io.kind_of(rp) != 'raw':
            print('跳过（没有 RAW）:', st)
            continue
        raw_paths.append(rp)
    if not raw_paths:
        print('没有可用的 RAW')
        return None

    tasks = []
    for rp in raw_paths:
        tasks.append((rp, max_side, None, True))              # 第一列：RAW 未修
        for s in stock_list:
            tasks.append((rp, max_side, s, False))
    print('共 %d 张渲染（%d 图 x %d 列），并发 %d' % (len(tasks), len(raw_paths),
                                                    len(stock_list) + 1, jobs))
    sys.stdout.flush()

    got = {}
    with cf.ProcessPoolExecutor(max_workers=max(1, jobs)) as ex:
        for p, key, u8, su, ctr in ex.map(_job, tasks):
            got[(p, key)] = (u8, su, ctr)
            print('  ok  %-14s %-14s' % (os.path.basename(p), key))
            sys.stdout.flush()

    ncol = len(stock_list) + 1
    W = CW * ncol + 5 * (ncol - 1)
    lh = 34
    H = lh + len(raw_paths) * (CH + 8)
    sheet = Image.new('RGB', (W, H), (18, 18, 18))
    d0 = date or paths.today()
    title = 'svFilm 卷对照 · %s   ——   左一是 RAW 未修，往右是各卷成片      [%s]' % (d0, purpose)
    if crop:
        title = 'svFilm 卷对照(100%%细节) · %s   ——   左一是 RAW 未修，往右是各卷成片      [%s]' % (d0, purpose)
    ImageDraw.Draw(sheet).text((10, 7), title, font=_font(20, True), fill=(255, 255, 255))

    y = lh
    for rp in raw_paths:
        x = 0
        ctr = crop
        if ctr == 'auto':
            ctr = got[(rp, 'base')][2] or (0.5, 0.4)
        u8, su, _ = got[(rp, 'base')]
        sheet.paste(_tag(_cell(u8, ctr), '0 RAW 未修', su), (x, y))
        x += CW + 5
        for s in stock_list:
            u8, su, _ = got[(rp, s)]
            sheet.paste(_tag(_cell(u8, ctr), stocks.label_of(s), su, True), (x, y))
            x += CW + 5
        ImageDraw.Draw(sheet).text((8, y + CH - FTR + 2), os.path.basename(rp),
                                   font=_font(14), fill=(120, 255, 160))
        y += CH + 8

    out = paths.debug_path(purpose, (tag or purpose) + '.jpg', root, d0)
    sheet.save(out, quality=90)
    print('-> %s  (%d x %d)' % (out, W, H))
    return out


DEFAULT_STOCKS = ['portra400', 'pro400h', 'fuji_c200', 'ektar100', 'cinestill800t', 'air']

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stems', nargs='+')
    ap.add_argument('--purpose', required=True)
    ap.add_argument('--stocks', nargs='*', default=DEFAULT_STOCKS)
    ap.add_argument('--root', default=None)
    ap.add_argument('--tag', default=None)
    ap.add_argument('--date', default=None)
    ap.add_argument('--max-side', type=int, default=1200, dest='max_side')
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--crop', nargs='+', default=None,
                    help="100%% 细节：--crop auto 自动找脸，或给两个归一化中心 --crop CX CY（配 --max-side 2048 看真实像素）")
    a = ap.parse_args()
    crop = None
    if a.crop:
        if a.crop[0] == 'auto':
            crop = 'auto'
        elif len(a.crop) >= 2:
            crop = (float(a.crop[0]), float(a.crop[1]))
    build(a.stems, a.purpose, a.stocks, a.root, a.tag, a.date, a.max_side, a.jobs, crop)
