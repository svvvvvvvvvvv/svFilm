# -*- coding: utf-8 -*-
"""IDT 机型表 —— **加机型不改代码**，只在这里加一行。

为什么要这张表：
  RAW 解码后是"传感器线性"，拿它当 18% 中灰标定，各家的偏置不一样。
  这里存的就是"这台机器的解码结果，中灰偏了多少档"，把它补回来，
  之后下游看到的所有图都处在同一个基准上（中灰 ≈ 0.18、白点 = 1.0）。

baseline_ev 的物理含义：
  需要额外乘 2^baseline_ev，才能让这张机器的线性中灰回到 0.18。

怎么填：
  用 `python -m svFilm.cli calib <RAW+JPG 同名的若干张>` 会打印建议值。
  优先拿"曝光标准"的样张（中灰落在 18% 上的那种），别拿夜景。

实测记录（2026-09-12，`D:\\照片库\\2026-09-0x_互勉约拍_深圳园岭新村`）：
  FUJIFILM X-T30 III —— 从 RAF 解码出来的线性，中灰本身就落在 0.18 标定上
  （DSCF0119 / DSCF0156 两张标准曝光的图，算出来增益 1.00）。
  所以这台机器 baseline_ev = 0.0；
  同批其余图 RAW 中灰低 3 档，是**拍摄时按保护高光欠曝**，不是解码偏置。
"""
from __future__ import annotations

# key = (make, model) 小写；只写 model 也能命中
TABLE = {
    'x-t30 iii': dict(baseline_ev=0.0, note='libraw 解码已含 0.18 中灰标定'),
    'x-t30': dict(baseline_ev=0.0, note='同 X-T30 III，待复核'),
    'x100vi': dict(baseline_ev=0.0, note='40MP，未做基准对照，按 0 处理'),
    'ilce-7m3': dict(baseline_ev=0.0, note='待用 calib 复核'),
    'ilce-7m4': dict(baseline_ev=0.0, note='待用 calib 复核'),
}

DEFAULT = dict(baseline_ev=0.0, note='未知机型，按 0 处理（多数机型 libraw 已标定）')


def lookup(make=None, model=None):
    keys = []
    if make and model:
        keys.append(('%s %s' % (make, model)).strip().lower())
    if model:
        keys.append(str(model).strip().lower())
    for k in keys:
        if k in TABLE:
            return dict(TABLE[k], key=k, hit=True)
    return dict(DEFAULT, key=None, hit=False)
