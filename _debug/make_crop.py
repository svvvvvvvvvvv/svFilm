# -*- coding: utf-8 -*-
"""100% 原始像素细节切图：底 / 出片 并排，看噪点、色斑、光晕、肤色。

产出落到 <root>/效果debug/<日期>/<目的>/<标签>.jpg

用法（在工程根目录执行）：
  python _debug/make_crop.py --purpose 三张RAW_细节100 <stem...> [--cx 0.5 --cy 0.42 --w 780]
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from svFilm import color, io, paths, pipeline  # noqa: E402

FONT_B = 'C:/Windows/Fonts/msyhbd.ttc'
FONT = 'C:/Windows/Fonts/msyh.ttc'
HDR = 30


def _font(s):
    for p in (FONT_B, FONT):
        try:
            return ImageFont.truetype(p, s)
        except Exception:
            pass
    return ImageFont.load_default()


def crop(disp, cx, cy, w):
    im = Image.fromarray(color.display_to_u8(np.clip(disp, 0, 1)))
    h = int(w * im.height / im.width)
    x = int(np.clip(cx * im.width - w / 2, 0, im.width - w))
    y = int(np.clip(cy * im.height - h / 2, 0, im.height - h))
    return im.crop((x, y, x + w, y + h))


def _root_of(stems, root):
    if root:
        return root
    return os.path.dirname(os.path.abspath(os.path.splitext(stems[0])[0])) or '.'


def build(stems, purpose, root=None, tag=None, date=None, cx=0.50, cy=0.42, w=760, max_side=2048):
    root = _root_of(stems, root)
    rows = []
    for st in stems:
        rp = io.find_pair(st, prefer='raw')
        if io.kind_of(rp) != 'raw':
            print('跳过:', st)
            continue
        res = pipeline.run(rp, src='raw', max_side=max_side, keep_stages=True)
        b = crop(res.report['stages']['base'], cx, cy, w)
        a = crop(res.disp, cx, cy, w)
        h = b.height
        strip = Image.new('RGB', (w * 2 + 5, h + HDR), (25, 25, 25))
        d = ImageDraw.Draw(strip)
        d.text((8, 6), '底（RAW 线性直出）  100%%', font=_font(18), fill=(255, 255, 255))
        d.text((w + 13, 6), 'svFilm 出片  100%%   %s' % (
            ('%+.2fEV' % res.report['tone']['ev_mid']) if res.report['tone']['applied'] else '未触发'),
            font=_font(18), fill=(140, 220, 255))
        strip.paste(b, (0, HDR))
        strip.paste(a, (w + 5, HDR))
        rows.append((os.path.basename(os.path.splitext(st)[0]), strip))

    if not rows:
        print('没有可用的行')
        return None

    d0 = date or paths.today()
    W = rows[0][1].width
    H = sum(r[1].height + 26 for r in rows) + 34
    sheet = Image.new('RGB', (W, H), (16, 16, 16))
    d = ImageDraw.Draw(sheet)
    d.text((10, 8), 'svFilm 细节 100%% 对照 · 底 vs 出片      [%s · %s]' % (d0, purpose),
           font=_font(20), fill=(255, 255, 255))
    y = 34
    for name, strip in rows:
        sheet.paste(strip, (0, y))
        d.text((6, y + 8), '  ' + name, font=_font(15), fill=(120, 255, 160))
        y += strip.height + 26

    out = paths.debug_path(purpose, (tag or purpose) + '.jpg', root, d0)
    sheet.save(out, quality=92)
    print('-> %s (%dx%d)' % (out, W, H))
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stems', nargs='+')
    ap.add_argument('--purpose', required=True, help='三级目录名：这批图是为了看什么')
    ap.add_argument('--root', default=None, help='效果debug 的上级目录，默认=样片所在目录')
    ap.add_argument('--tag', default=None, help='文件名，默认同 purpose')
    ap.add_argument('--date', default=None, help='二级目录名，默认今天')
    ap.add_argument('--cx', type=float, default=0.50)
    ap.add_argument('--cy', type=float, default=0.42)
    ap.add_argument('--w', type=int, default=760)
    ap.add_argument('--max-side', type=int, default=2048, dest='max_side')
    a = ap.parse_args()
    build(a.stems, a.purpose, a.root, a.tag, a.date, a.cx, a.cy, a.w, a.max_side)
