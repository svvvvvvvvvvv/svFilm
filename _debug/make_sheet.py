# -*- coding: utf-8 -*-
"""对照图：同一张照片，把管线各段摊开并排。

栏位按素材自动长：
  没有同名 JPG  ->  [1] RAW 线性直出（未修） [2] 只做影调修正 [3] 出片
  有同名 JPG    ->  再加 [0] 机内 JPG（相机直出）

产出落到 <root>/效果debug/<日期>/<目的>/<标签>.jpg

用法（在工程根目录执行）：
  python _debug/make_sheet.py --purpose 三张RAW_分段对照 <stem 或任意同名路径> ...
  python _debug/make_sheet.py --purpose XXX --root <样片目录> --tag 名字 --date 2026-09-12 <stem...>
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from svFilm import color, config as C, io, paths, pipeline  # noqa: E402

FONT = 'C:/Windows/Fonts/msyh.ttc'
FONT_B = 'C:/Windows/Fonts/msyhbd.ttc'
CW, CH = 520, 347          # 单元格尺寸（3:2）
HDR, FTR = 32, 28


def _font(size, bold=False):
    for p in ((FONT_B, FONT) if bold else (FONT, FONT_B)):
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default()


def _cell(disp):
    im = Image.fromarray(color.display_to_u8(np.clip(disp, 0, 1)))
    if im.size != (CW, CH):
        r = max(CW / im.width, CH / im.height)
        im = im.resize((int(im.width * r) + 1, int(im.height * r) + 1), Image.LANCZOS)
        l = (im.width - CW) // 2
        t = (im.height - CH) // 2
        im = im.crop((l, t, l + CW, t + CH))
    return im


def _tag(im, title, sub, hi=False):
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, CW, HDR], fill=(0, 0, 0))
    d.text((8, 6), title, font=_font(19, True), fill=(140, 220, 255) if hi else (255, 255, 255))
    d.rectangle([0, CH - FTR, CW, CH], fill=(0, 0, 0))
    d.text((8, CH - FTR + 3), sub, font=_font(16), fill=(255, 235, 150))
    return im


def _stats(disp):
    g = color.gray_of(disp)
    lab = color.to_lab(np.clip(disp, 0, 1))
    return '中灰%3d · L*%4.1f · 黑%3d 白%3d · 彩度%4.1f · 死白%.2f%%' % (
        np.percentile(g, 50) * 255, np.percentile(lab[..., 0], 50),
        np.percentile(g, C.PCT_BLACK) * 255, np.percentile(g, C.PCT_WHITE) * 255,
        color.chroma_c90(np.clip(disp, 0, 1)), np.mean(g >= 254 / 255) * 100)


def _root_of(stems, root):
    if root:
        return root
    return os.path.dirname(os.path.abspath(os.path.splitext(stems[0])[0])) or '.'


def build(stems, purpose, root=None, tag=None, date=None, max_side=1400):
    root = _root_of(stems, root)
    rows = []
    for st in stems:
        stem = os.path.splitext(st)[0]
        rp = io.find_pair(stem, prefer='raw')
        if io.kind_of(rp) != 'raw':
            print('跳过（没有 RAW）:', st)
            continue
        jp = None
        for e in ('.JPG', '.jpg', '.JPEG', '.jpeg'):
            if os.path.exists(stem + e):
                jp = stem + e
                break

        res = pipeline.run(rp, src='raw', max_side=max_side, keep_stages=True)
        stg = res.report['stages']
        t = res.report['tone']

        cols = []
        if jp:
            rj = pipeline.run(jp, src='jpg', max_side=max_side)
            cols.append((_cell(rj.sample.disp), '0 机内 JPG（相机直出）', _stats(rj.sample.disp), False))
        cols.append((_cell(stg['base']), '1 RAW 线性直出（未修）', _stats(stg['base']), False))
        cols.append((_cell(stg['after_tone']), '2 只做影调修正',
                     _stats(stg['after_tone']) + ('  [%+.2fEV]' % t['ev_mid'] if t['applied'] else '  [未触发]'),
                     False))
        cols.append((_cell(res.disp), '3 svFilm 出片（完整）',
                     _stats(res.disp) + '  [' + ('/'.join(res.report['guard']['actions']) or '护栏未动') + ']', True))

        strip = Image.new('RGB', (CW * len(cols) + 5 * (len(cols) - 1), CH), (30, 30, 30))
        for i, (im, ti, su, hi) in enumerate(cols):
            strip.paste(_tag(im, ti, su, hi), (i * (CW + 5), 0))
        rows.append((os.path.basename(stem), strip))

    if not rows:
        print('没有可用的行')
        return None

    d0 = date or paths.today()
    lh = 36
    W = rows[0][1].width
    H = lh + len(rows) * (CH + 8)
    sheet = Image.new('RGB', (W, H), (18, 18, 18))
    d = ImageDraw.Draw(sheet)
    d.text((10, 8), 'svFilm 分段对照 · %s   ——   底 / 只修正 / 出片      [%s]' % (d0, purpose),
           font=_font(21, True), fill=(255, 255, 255))
    y = lh
    for name, strip in rows:
        sheet.paste(strip, (0, y))
        ImageDraw.Draw(sheet).text((8, y + CH - FTR + 3), name, font=_font(15), fill=(120, 255, 160))
        y += CH + 8

    out = paths.debug_path(purpose, (tag or purpose) + '.jpg', root, d0)
    sheet.save(out, quality=90)
    print('-> %s  (%d x %d, %d 张)' % (out, W, H, len(rows)))
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stems', nargs='+')
    ap.add_argument('--purpose', required=True, help='三级目录名：这批图是为了看什么')
    ap.add_argument('--root', default=None, help='效果debug 的上级目录，默认=样片所在目录')
    ap.add_argument('--tag', default=None, help='文件名，默认同 purpose')
    ap.add_argument('--date', default=None, help='二级目录名，默认今天')
    ap.add_argument('--max-side', type=int, default=1400, dest='max_side')
    a = ap.parse_args()
    build(a.stems, a.purpose, a.root, a.tag, a.date, a.max_side)
