# -*- coding: utf-8 -*-
"""命令行入口。

  python -m svFilm.cli probe  <img...>                 只看数，不写文件
  python -m svFilm.cli one    <img> -o out.jpg
  python -m svFilm.cli dir    <in_dir> -o <out_dir> [--jobs 4]
  python -m svFilm.cli bake   <out.cube> [--size 33]
"""
from __future__ import annotations

import argparse
import glob as _glob
import json
import os
import sys

from . import analyze, config as C, io, stocks, style


def _fmt_probe(res):
    a = res.report['analyze']
    g = a['gray255']
    bn = res.report.get('base_label') or '无'
    return ('%-16s %-4s %-12s 基准%-10s 中灰%6.1f(L*%5.1f) 黑%5.1f 白%6.1f '
            '死白%6.2f%% 彩度P90 %5.1f  ev%+5.2f  → %s' % (
                res.sample.name, res.sample.kind,
                (res.report.get('stock') or 'config默认'), bn,
                g[C.PCT_MID], a['L_pcts'][C.PCT_MID],
                g[C.PCT_BLACK], g[C.PCT_WHITE],
                a['dead_white_frac'] * 100.0, a['chroma_c90'],
                a['ev_est'], a['decision']))


def cmd_probe(args):
    from . import pipeline
    for p in _expand(args.inputs, args.recursive):
        res = pipeline.run(p, src=args.src, max_side=args.max_side,
                           stock=args.stock, base=args.base)
        line = _fmt_probe(res)
        if args.after:
            r = res.report
            g = r['guard']
            t = r['tone']
            sp = r.get('spatial') or {}
            line += '  ||  出片 中灰%6.1f 死白%6.2f%% 彩度%5.1f  修正%s  空间%s' % (
                _gray_mid(res.disp) * 255.0, g['dead_white_frac'] * 100.0, g['chroma_c90'],
                ('%+.2fEV' % t['ev_mid']) if t['applied'] else '未触发',
                ('有' if sp.get('any') else '无'))
        print(line)
    return 0


def cmd_calib(args):
    """量机型表用的基线增益：拿 RAW 与"同名 JPG"对齐中灰。"""
    import glob as _g
    import os as _os

    import numpy as np
    from . import color, io

    pairs = []
    for p in _expand(args.inputs, True):
        if io.kind_of(p) != 'raw':
            continue
        jpg = None
        for e in ('.JPG', '.jpg', '.JPEG', '.jpeg'):
            if _os.path.exists(_os.path.splitext(p)[0] + e):
                jpg = _os.path.splitext(p)[0] + e
                break
        if jpg:
            pairs.append((p, jpg))
    if not pairs:
        print('没找到 RAW+JPG 同名对')
        return 2

    print('%-16s %-14s %8s %8s %8s' % ('file', 'model', 'rawY50', 'jpgY50', 'gain'))
    gains = []
    for rp, jp in pairs:
        a = io.load(rp, max_side=1024, src='raw')
        b = io.load(jp, max_side=1024, src='jpg')
        ym = float(np.percentile(color.Y_of(a.lin), 50))
        jm = float(np.percentile(color.s2l(b.disp), 50))
        g = jm / max(ym, 1e-9)
        gains.append(g)
        md = a.cam.get('model') or '?'
        print('%-16s %-14s %8.4f %8.4f %8.2f' % (_os.path.basename(rp), md, ym, jm, g))
    med = float(np.median(gains))
    print('-' * 60)
    print('中位增益 %.3f  →  建议 baseline_ev = %+.2f' % (med, float(np.log2(med))))
    print('（挑曝光标准的样张；同一批里有人为欠曝的图会把数字拉偏）')
    return 0


def _gray_mid(disp):
    import numpy as np
    from . import color
    return float(np.percentile(color.gray_of(disp), C.PCT_MID))


def cmd_one(args):
    from . import pipeline
    res = pipeline.run(args.input, src=args.src, max_side=args.max_side,
                       stock=args.stock, base=args.base)
    out = args.out or (os.path.splitext(args.input)[0] + '_svFilm.jpg')
    res.save(out)
    print(res.summary())
    print('-> ' + out)
    if args.json:
        with open(args.json, 'w', encoding='utf-8') as f:
            json.dump(res.report, f, ensure_ascii=False, indent=1)
    return 0


def _worker(t):
    from . import pipeline
    src, max_side, outdir, p, stock, base = t
    name = os.path.splitext(os.path.basename(p))[0]
    # 输出名带 _svFilm 后缀：不能叫 <名字>.jpg —— 那会和"相机直出同名 JPG"撞名字，
    # 下游一旦按"同名 JPG = 机内直出"去解读，就会把自己的产出当成相机底来看
    suffix = '_svFilm' + (('_' + stock) if stock else '') + (('_' + base) if base else '')
    out = os.path.join(outdir, name + suffix + '.jpg')
    if os.path.abspath(out) == os.path.abspath(p):
        return (p, None, '', '输出会覆盖输入，已跳过')
    try:
        res = pipeline.run(p, src=src, max_side=max_side, stock=stock, base=base)
        res.save(out)
        return (p, out, res.summary(), None)
    except Exception as e:                                  # noqa: BLE001
        return (p, None, '', '%s: %s' % (type(e).__name__, e))


def _pair_resolve(paths, pair_dir):
    """给一批 JPG，去 pair_dir 里找**同名 RAW**（找不到就保留原文件）。

    用途：选片目录（只有 JPG）里的文件名 → 上一层拍摄目录里的同名 RAW。
    这是"RAW 优先、JPG 替补"的落地方式：能拿到 RAW 就一定用 RAW 走管线。
    """
    out = []
    for p in paths:
        try:
            k = io.kind_of(p)
        except ValueError:
            k = 'std'
        if k == 'raw':
            out.append(p)
            continue
        stem = os.path.splitext(os.path.basename(p))[0]
        hit = None
        for e in sorted(io.RAW_EXT):
            for ee in (e, e.upper()):
                cand = os.path.join(pair_dir, stem + ee)
                if os.path.exists(cand):
                    hit = cand
                    break
            if hit:
                break
        out.append(hit or p)
    return out


def cmd_dir(args):
    from . import io as _io
    ps = _expand(args.inputs, True, pattern=args.glob)
    if args.pair:
        ps = _pair_resolve(ps, args.pair)
    if args.raw_only:
        keep = []
        for p in ps:
            try:
                if _io.kind_of(p) == 'raw':
                    keep.append(p)
            except ValueError:
                pass
        ps = keep
    # 配对/过滤后再去一次重（两张不同 JPG 可能配到同一张 RAW）
    seen, uniq = set(), []
    for p in ps:
        k = os.path.abspath(p).lower()
        if k not in seen:
            seen.add(k)
            uniq.append(p)
    ps = uniq
    if args.limit:
        ps = ps[:args.limit]
    if not ps:
        print('没找到输入文件')
        return 2
    os.makedirs(args.out, exist_ok=True)
    jobs = max(1, int(args.jobs))

    # 先拿第一张热身，把可能的解码错误暴露在串行阶段
    print('共 %d 个文件，并发 %d' % (len(ps), jobs))
    sys.stdout.flush()

    from . import pipeline
    tasks = [(args.src, args.max_side, args.out, p, args.stock, args.base) for p in ps]
    done = fail = 0
    if jobs == 1:
        for t in tasks:
            p, out, s, err = _worker(t)
            done += err is None
            fail += err is not None
            print(('OK  ' + s) if err is None else ('FAIL ' + p + ' ' + err))
            sys.stdout.flush()
    else:
        import concurrent.futures as cf
        with cf.ProcessPoolExecutor(max_workers=jobs) as ex:
            for p, out, s, err in ex.map(_worker, tasks):
                done += err is None
                fail += err is not None
                print(('OK  ' + s) if err is None else ('FAIL ' + p + ' ' + err))
                sys.stdout.flush()
    print('完成 %d / 失败 %d  -> %s' % (done, fail, os.path.abspath(args.out)))
    return 0 if fail == 0 else 1


def cmd_bake(args):
    from . import paths
    if args.purpose:
        root = args.root or '.'
        out = paths.debug_path(args.purpose, args.out, root, args.date)
    else:
        out = args.out
    os.makedirs(os.path.dirname(os.path.abspath(out)) or '.', exist_ok=True)
    st = stocks.get(args.stock)
    style.bake_cube(out, size=args.size, stock=st, base=args.base)
    bn = stocks.base_label(C, args.base)
    print('已烘焙 -> %s (size %d%s%s)' % (os.path.abspath(out), args.size,
                                     (', 卷 ' + st['label']) if st else '',
                                     (', 基准 ' + bn) if bn != '无' else ''))
    return 0


def cmd_stocks(args):
    """列出所有卷（人话说明）。"""
    print('%-16s %-14s %s' % ('卷名', '中文名', '一句话'))
    print('-' * 92)
    for n in stocks.names():
        s = stocks.TABLE[n]
        flag = ' ←' if (args.stock or C.STOCK) == n else ''
        print('%-16s %-14s %s%s' % (n, s['label'], s['desc'], flag))
    print('-' * 92)
    print('用法：--stock <卷名>；不指定 = config.STOCK（当前 %s）' % (C.STOCK or 'None'))
    return 0


def cmd_bases(args):
    """列出基准成色的四条候选（不属于任何卷，是"我方中性路径对齐大师平均"的三档）。"""
    print('%-16s %-14s %s' % ('预设名', '中文名', '画面上看得出来什么'))
    print('-' * 108)
    for n in stocks.base_names():
        d = C.BASE_TABLE.get(n, {})
        flag = ' ←' if (args.base or C.BASE) == n else ''
        print('%-16s %-14s %s%s' % (n, d.get('label', n), d.get('desc', ''), flag))
    print('-' * 108)
    print('用法：--base <预设名>；不指定 = config.BASE（当前 %s）' % C.BASE)
    print('出处：卷标定报告 §四（我方中性路径 vs 大师平均的系统性差）')
    return 0


def _expand(inputs, recursive, pattern='*.jpg'):
    out = []
    for p in inputs:
        if os.path.isdir(p):
            pats = ['*.RAF', '*.raf', '*.ARW', '*.arw', '*.CR2', '*.CR3', '*.NEF', '*.DNG',
                    '*.JPG', '*.jpg', '*.JPEG', '*.jpeg', '*.PNG', '*.png']
            for pat in pats:
                out.extend(sorted(_glob.glob(os.path.join(p, '**', pat), recursive=True)
                                  if recursive else _glob.glob(os.path.join(p, pat))))
        elif any(ch in p for ch in '*?['):
            out.extend(sorted(_glob.glob(p, recursive=recursive)))
        else:
            out.append(p)
    from . import paths
    seen, uniq = set(), []
    for p in out:
        k = os.path.abspath(p).lower()
        if k in seen:
            continue
        # 自家产出树整棵跳过：否则第二次跑目录会把上一轮的对照图/成片当素材再处理一遍
        if paths.TOP.lower() in [s.lower() for s in os.path.abspath(p).split(os.sep)]:
            continue
        seen.add(k)
        uniq.append(p)
    return uniq


def build_parser():
    ap = argparse.ArgumentParser('svFilm', description='数码仿胶片管线（重写版）')
    sub = ap.add_subparsers(dest='cmd', required=True)

    def common(p):
        p.add_argument('--src', default=None, choices=[None, 'auto', 'raw', 'jpg'])
        p.add_argument('--max-side', type=int, default=None, dest='max_side')
        p.add_argument('--stock', default=None,
                       help='胶片卷：%s；不指定 = config.STOCK' % '/'.join(stocks.names()))
        p.add_argument('--base', default=None,
                       help='基准成色：%s；不指定 = config.BASE' % '/'.join(stocks.base_names()))

    ap.add_argument('--version', action='version', version='svFilm ' + C.VERSION)

    p = sub.add_parser('stocks', help='列出所有胶片卷')
    p.add_argument('--stock', default=None)
    p.set_defaults(func=cmd_stocks)

    p = sub.add_parser('bases', help='列出基准成色候选（中性路径对齐大师平均的三档）')
    p.add_argument('--base', default=None)
    p.set_defaults(func=cmd_bases)

    p = sub.add_parser('probe', help='只分析不写文件')
    p.add_argument('inputs', nargs='+')
    p.add_argument('--recursive', action='store_true')
    p.add_argument('--after', action='store_true', help='同时打印出片后的数')
    common(p)
    p.set_defaults(func=cmd_probe)

    p = sub.add_parser('one', help='处理单张')
    p.add_argument('input')
    p.add_argument('-o', '--out', default=None)
    p.add_argument('--json', default=None, help='把报告写成 json')
    common(p)
    p.set_defaults(func=cmd_one)

    p = sub.add_parser('dir', help='批处理目录')
    p.add_argument('inputs', nargs='+')
    p.add_argument('-o', '--out', required=True)
    p.add_argument('--jobs', type=int, default=4, help='重活建议 <=4（内存）')
    p.add_argument('--glob', default='*.jpg', help='占位，实际按内置扩展名表扫描')
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--raw-only', action='store_true', dest='raw_only',
                   help='只要 RAW（目录里 RAW+JPG 混放时用）')
    p.add_argument('--pair', default=None, metavar='DIR',
                   help='给 JPG 去这个目录找同名 RAW 来跑（选片目录只有 JPG 时用）')
    common(p)
    p.set_defaults(func=cmd_dir)

    p = sub.add_parser('bake', help='把 L2 风格层烘成 .cube')
    p.add_argument('out', help='文件名；给了 --purpose 时只当文件名，落进效果debug 树')
    p.add_argument('--size', type=int, default=33)
    p.add_argument('--stock', default=None, help='烘哪个卷（不指定 = config 默认风格）')
    p.add_argument('--base', default=None, help='叠哪一档基准成色')
    p.add_argument('--purpose', default=None, help='三级目录名，给了就走效果debug 规范')
    p.add_argument('--root', default=None, help='效果debug 的上级目录')
    p.add_argument('--date', default=None)
    p.set_defaults(func=cmd_bake)

    p = sub.add_parser('calib', help='量机型表要填的基线增益（需 RAW+JPG 同名对）')
    p.add_argument('inputs', nargs='+')
    p.set_defaults(func=cmd_calib)
    return ap


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
