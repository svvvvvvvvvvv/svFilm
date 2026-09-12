# -*- coding: utf-8 -*-
"""IDT 机型表 —— **加机型不改代码**，只在这里加一行。

为什么要这张表：
  相机厂商**故意让 RAW 欠曝**，把"把中点抬回来"这一步留给转换器做。
  这一步在 DNG 规范里叫 **baseline exposure**；Adobe 系（ACR / Lightroom）都是"静默"做掉的
  （滑块显示 0，其实已经抬过）。RawTherapee 官方文档：**目的就是让亮度匹配机内 JPEG**。
  LibRaw 作者 Alex Tutubalin 的原话：相机灰点通常只在满量程的 7~12%，转换器要把它抬回 18%，
  「0.7~1.5EV gray point move is usual」，而且**要用一条自定义 tone curve 把高光压下去**。

表里存两样东西，**别混**：
  1) `baseline_ev`：与该**机型**绑定的常数基线。
  2) 富士的 DR 档额外欠曝 —— 这一项**每张都可能不同**，逐张从 RAF 的 MakerNote 里读
     （`rawmeta.fuji_development_dr`），量在 `FUJI_DR_BIAS`。
     最终补量 = baseline_ev + FUJI_DR_BIAS[dr]；若机身写了精确的
     `0x9650 RawExposureBias`，**直接用那个值覆盖**（它是含基础偏移的总量）。

实测记录（2026-09-13）：
  园岭 32 张全是 X-T30 III + DR400 ⇒ 补 +2.72EV。补完底与机内 JPEG 对齐
  （DSCF1010 中位 59 vs 66、DSCF1032 71 vs 79；批量 16 张中位差从 −3.8 档收到 −0.84 档）。
  ⚠ 同一台机身 DR 会变：整库 759 张里 **DR100 118 张 / DR200 66 张 / DR400 575 张** ⇒
    逐张读。按机型写死 DR400 会错掉那 184 张。
  ⚠ 旧记录「X-T30 III 解码本身已含 0.18 中灰标定，baseline_ev = 0」是**错的**：
    当时只看了两张"标准曝光"的样张，恰好是 DR100 或恰好落在 0.18，样本太少把结论带偏了。

怎么填 `baseline_ev`：
  用 `python -m svFilm.cli calib <RAW+JPG 同名的若干张>` 会打印建议值。
  优先拿"曝光标准"的样张（中灰落在 18% 上的那种），别拿夜景。
"""
from __future__ import annotations

# key = (make, model) 小写；只写 model 也能命中
# 0.72 = 富士的基础偏移（darktable：even at 100DR there is an EV shift −0.72）。
TABLE = {
    'x-t30 iii': dict(baseline_ev=0.72, note='富士基底 0.72；DR 额外量逐张读 tag'),
    'x-t30': dict(baseline_ev=0.72, note='同 X-T30 III，待复核'),
    'x100vi': dict(baseline_ev=0.72, note='40MP；富士基底 0.72，DR 逐张读'),
    'ilce-7m3': dict(baseline_ev=0.0, note='▲ 未标定。LibRaw 说多数机身偏 0.7~1.5EV，别猜，跑 calib'),
    'ilce-7m4': dict(baseline_ev=0.0, note='▲ 未标定。同上'),
}

DEFAULT = dict(baseline_ev=0.0, note='未知机型 ⇒ 不猜，常数项按 0；有没有 DR tag 由 rawmeta 决定')

# 富士「动态范围」档位的**额外**欠曝（不含那 0.72 的基底）。
# 来源：darktable 官方 lua 脚本 fujifilm_dynamic_range 的实测值
#   RawExposureBias 总量：100 DR -> −0.72 EV ／ 200 DR -> −1.72 EV ／ 400 DR -> −2.72 EV
# 减去基底 0.72 ⇒ 额外量就是 0 / 1 / 2 EV。
# 同口径：RapidRAW issue #711（DevelopmentDynamicRange 100/200/400 -> 0/+1/+2 EV）。
FUJI_DR_BIAS = {100: 0.0, 200: 1.0, 400: 2.0}


def dr_bias_ev(dr):
    """富士 DR 档 -> 要**额外**补的 EV（不含机型基底）。None / 不认识 -> None。"""
    if dr is None:
        return None
    try:
        return FUJI_DR_BIAS.get(int(dr))
    except (TypeError, ValueError):
        return None


# ==================== 入口曲线（Q1，2026-09-13）====================
# 为什么不再用"一个恒定乘数"：
#   恒定乘数在 log 域是一条**直线**；实测相机曲线是**弓形** ——
#   暗部抬 +2.6 档、中间调峰 +3.33 档（70% 分位）、高光塌回 +0.85 档（99.5%）。
#   全程 +2.72 ⇒ 中间调偏暗 0.4~0.6 档、高光多推 0.5~1.9 档（0805 路面死白 92%，相机才 6.9%）。
# 出处（三处独立印证"零点和形状是两件事"）：
#   * DNG 规范：BaselineExposure "specifies ... to move **the zero point**"；
#     Adobe 相机 raw 团队 Eric Chan：「it's there to match to LR's sliders, nothing else」。
#   * Adobe DCP 的 tone curve 负责"和机内 JPEG 差不多的亮度与对比度"。
#   * RawTherapee 5.4 起默认的 Auto-Matched Tone Curve：拿内嵌 JPEG 做**直方图匹配**算曲线。
# ⇒ **曝光零点（一个 EV 数）+ 曲线形状**，两项都从真实 RAW+机内 JPG 对里量。
#
# 量化口径：`_debug/calib_entry_curve_from_pairs.py`
#   「纯底」= io.load_raw 关掉入口补偿拿到的线性；「相机」= 同名机内 JPEG（已 exif_transpose）。
#   逐分位配对（Q-Q 匹配）⇒ 同一分位上的 (纯底, 相机) 就是曲线的一个采样点；
#   按 **机型 × DR 档** 分组取中位（DR 会改曲线形状，不能混着平均）。
#
# 数据格式：ENTRY_CURVE[model][str(dr)] = dict(mid_ev=..., anchors=[[输入线性, 形状倍率], ...])
#   `mid_ev`  = 中位分位处的实测增益档 = **曝光零点**
#   `anchors` = 形状，**已归一到中灰**（中位分位处 = 1.0），按输入线性升序
# 用法：out_lin = in_lin * 2**mid_ev * shape(in_lin)，shape 用 anchors 线性插值。
# ⚠ 曲线按 **亮度** 套（`color.luma(lin)` 驱动），三通道同一个增益 ⇒ 不改色相；
#   这跟契约里"只在亮度域做曲线"一致。
ENTRY_CURVE = {}


def entry_curve(model=None, dr=None):
    """取实测入口曲线 -> (mid_ev, xs, ys) 或 None（没有就退回老的常数补偿）。"""
    tab = ENTRY_CURVE.get(str(model or '').strip().lower())
    if not tab:
        return None
    c = tab.get(str(dr)) or tab.get('None') or tab.get('default')
    if not c:
        return None
    xs = [float(p[0]) for p in c['anchors']]
    ys = [float(p[1]) for p in c['anchors']]
    return float(c['mid_ev']), xs, ys


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
