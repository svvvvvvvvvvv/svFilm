# -*- coding: utf-8 -*-
"""量我们自己成片的"颜色指纹"（和量大师那把尺子同口径）。

用途两件：
  1) 拿**基线**（不套卷，只有 L1 的成片）—— 标定卷的时候要拿它当起点；
  2) **验收**：套了卷之后，量成片有没有落进大师带
     （近中性占比 47~75%、彩度形状 c90/c50 2.65~3.41、彩度P90 11.7~25.9、黑位地板 0~13）。

用法（工程根目录）：
  PY _debug/lab_ours_fingerprint.py --stocks neutral
  PY _debug/lab_ours_fingerprint.py --stocks neutral,fuji_c200,ektar100 --tag 验收
"""
import argparse
import concurrent.futures as cf
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from svFilm import config as C, io, metrics, paths, pipeline, stocks  # noqa: E402

KEYS = ['L50', 'span90', 'black', 'c50', 'c90', 'c_ratio', 'gray_pct',
        'a_med', 'b_med', 'b_sh', 'b_hi', 'split', 'fade_lin', 'noise', 'hi90']


def _job(t):
    path, stock_name, max_side = t
    st = stocks.get(stock_name)
    res = pipeline.run(path, src='raw', max_side=max_side, stock=st, keep_stages=True)
    stg = res.report['stages']
    out = dict(
        sample=res.sample.name, stock=stock_name,
        base=metrics.fingerprint(stg['base']),          # RAW 线性直出的样子
        after_l1=metrics.fingerprint(stg['after_tone']),  # 只做完影调修正（= 卷的输入）
        final=metrics.fingerprint(res.disp),            # 出片
    )
    return res.sample.name, stock_name, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stems', nargs='*', default=[
        'D:/照片库/svFilm效果/DSCF2328.RAF',
        'D:/照片库/svFilm效果/DSCF4504.RAF',
        'D:/照片库/svFilm效果/DSCF6407.RAF'])
    ap.add_argument('--stocks', default='neutral')
    ap.add_argument('--purpose', default='卷标定_我方指纹')
    ap.add_argument('--root', default='D:/照片库/svFilm效果')
    ap.add_argument('--tag', default=None)
    ap.add_argument('--date', default=None)
    ap.add_argument('--max-side', type=int, default=2048, dest='max_side')
    ap.add_argument('--jobs', type=int, default=3)
    a = ap.parse_args()

    raws = []
    for st in a.stems:
        rp = io.find_pair(os.path.splitext(st)[0], prefer='raw')
        raws.append(rp)
    names = [s.strip() for s in a.stocks.split(',') if s.strip()]

    tasks = [(rp, n, a.max_side) for n in names for rp in raws]
    print('共 %d 张渲染（%d 图 x %d 卷）' % (len(tasks), len(raws), len(names)))
    sys.stdout.flush()
    got = {}
    with cf.ProcessPoolExecutor(max_workers=max(1, a.jobs)) as ex:
        for smp, stk, o in ex.map(_job, tasks):
            got[(smp, stk)] = o
            print('  ok  %-10s %-12s 中灰%.0f 彩度50 %.1f 彩度90 %.1f 形状%.2f 中性%.0f%%'
                  % (smp, stk, o['final']['L50'], o['final']['c50'], o['final']['c90'],
                     o['final']['c_ratio'], o['final']['gray_pct'] * 100))
            sys.stdout.flush()

    # 每卷汇总：跨样片取中位；彩度形状用"先取中位再相除"（与大师画像表口径一致）
    print('\n%-14s %-6s %-7s %-7s %-7s %-7s %-7s %-7s %-7s %s'
          % ('卷', 'L*中位', '反差', '彩度50', '彩度90', '形状', '中性占比', '暗部b*', '亮部b*', '黑位'))
    summary = {}
    for n in names:
        sub = [got[k]['final'] for k in got if k[1] == n]
        if not sub:
            continue
        vals = {k: float(np.median([s[k] for s in sub])) for k in KEYS}
        vals['c_ratio'] = vals['c90'] / max(vals['c50'], 1e-6)
        vals['split'] = vals['b_hi'] - vals['b_sh']
        summary[n] = vals
        print('%-14s %-6.0f %-7.0f %-7.1f %-7.1f %-7.2f %-7.0f%% %-7.1f %-7.1f %.1f'
              % (n, vals['L50'], vals['span90'], vals['c50'], vals['c90'], vals['c_ratio'],
                 vals['gray_pct'] * 100, vals['b_sh'], vals['b_hi'], vals['black']))

    band = metrics.BAND
    print('\n落带检查（大师带）：')
    for n in names:
        v = summary[n]
        ok = (band['gray_pct'][0] <= v['gray_pct'] <= band['gray_pct'][1]
              and band['c_ratio'][0] <= v['c_ratio'] <= band['c_ratio'][1]
              and band['c90'][0] <= v['c90'] <= band['c90'][1]
              and band['black'][0] <= v['black'] <= band['black'][1])
        flags = []
        if not band['gray_pct'][0] <= v['gray_pct'] <= band['gray_pct'][1]:
            flags.append('中性占比 %.0f%% 不在 47~75%%' % (v['gray_pct'] * 100))
        if not band['c_ratio'][0] <= v['c_ratio'] <= band['c_ratio'][1]:
            flags.append('彩度形状 %.2f 不在 2.65~3.41' % v['c_ratio'])
        if not band['c90'][0] <= v['c90'] <= band['c90'][1]:
            flags.append('彩度P90 %.1f 不在 11.7~25.9' % v['c90'])
        if not band['black'][0] <= v['black'] <= band['black'][1]:
            flags.append('黑位 %.1f 不在 0~13' % v['black'])
        print('  %-14s %s' % (n, '✅ 四项全落带' if ok else '❌ ' + '；'.join(flags)))

    p = paths.debug_path(a.purpose, (a.tag or 'ours_fingerprint') + '.json', a.root, a.date)
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(dict(samples={'%s|%s' % k: v for k, v in got.items()},
                       summary=summary), f, ensure_ascii=False, indent=1)
    print('\n-> %s' % p)
    return got


if __name__ == '__main__':
    main()
