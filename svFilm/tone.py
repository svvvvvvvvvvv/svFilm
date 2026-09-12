# -*- coding: utf-8 -*-
"""L1 —— 修正层。影调，lin 域进、lin 域出。

三条与"边搭边补"最本质的区别：
  1) 靶是绝对靶（黑 0.01 / 中灰 0.55 / 白 0.86 显示域），不是"这张图自己的均值"。
     靶自指 == 输入等于输出 == 永远 ev 0 == 死代码。
  2) 曲线只作用在"亮度"上，RGB 同步按比例缩放。
     逐通道曲线会在高光端把颜色放大好几倍（粉裙变鲜红就是这么来的）。
  3) 因为 1)+2)，中灰落点可以一次解析算出，不需要二分闭环。

曲线骨架（log2 域，5 个锚点 + 地板）：
    地板(黑底) → 黑点 → P25(阴影填充) → 中灰靶 → 膝点(在中灰-白直线上) → 白点靶
"""
import numpy as np

from . import color
from . import config as C


def _t(y):
    """lin 亮度 -> log2 域（曲线的定义域）。log 域插值才不会把暗部压扁。"""
    return np.log2(np.maximum(np.asarray(y, np.float64), C.NOISE_FLOOR))


class ToneCurve:
    """单调分段线性曲线，定义在 log2 亮度域。"""

    def __init__(self, t_in, t_out):
        t_in = np.maximum.accumulate(np.asarray(t_in, np.float64))
        t_out = np.maximum.accumulate(np.asarray(t_out, np.float64))
        # 严格递增：把重合的结点用极小量推开（不能让它们合并，
        # 合并会悄悄丢掉"黑点到靶"这一步，等于曲线上少了一段）
        t_in = t_in + np.arange(t_in.shape[0]) * 1e-7
        self.t_in = t_in
        self.t_out = t_out

    def __call__(self, y):
        return np.exp2(np.interp(_t(y), self.t_in, self.t_out))

    def gain(self, y):
        """输出/输入 比值。用于 RGB 同步缩放。"""
        return self(y) / np.maximum(np.asarray(y, np.float64), C.NOISE_FLOOR)


def build_curve(rep, cfg=C, allow_lift=False):
    """由 L0 报告造曲线。返回 (curve, info)。"""
    lp = rep['lin_pcts']
    yb, y25, ym, yk, yw = (lp[cfg.PCT_BLACK], lp[25.0], lp[cfg.PCT_MID],
                           lp[cfg.PCT_KNEE], lp[cfg.PCT_WHITE])

    # ---- 绝对靶：显示域 -> lin 域 ----
    Tb = float(color.s2l(cfg.TGT_BLACK))
    Tm = float(color.s2l(cfg.TGT_MID))
    Tw = float(color.s2l(cfg.TGT_WHITE))

    # ---- 输入位置（log2） ----
    tb_in, t25_in, tm_in, tk_in, tw_in = _t(yb), _t(y25), _t(ym), _t(yk), _t(yw)

    # ---- 中灰增益：一个量，双向限幅 ----
    g_raw = Tm / max(ym, cfg.NOISE_FLOOR)
    if g_raw >= 1.0:                                   # 方向：提亮
        g = min(g_raw, 2.0 ** cfg.EV_CAP_UP) if allow_lift else 1.0
    else:                                              # 方向：压暗
        g = max(g_raw, 2.0 ** -cfg.EV_CAP_DOWN)
    Tm_eff = ym * g
    tm_out = _t(Tm_eff)
    capped = bool(abs(np.log2(g) - np.log2(max(g_raw, cfg.NOISE_FLOOR))) > 1e-3)

    # 黑点：朝绝对靶压，但 ① 最多压 BLACK_PULL 档（防"画面本来没有暗部"被压死）
    #              ② 绝不反过来把黑提亮（否则纯黑像素会顶到 1% 灰，暗部出现台阶）
    tb_out = min(tb_in, max(_t(Tb), tb_in - cfg.BLACK_PULL))

    tw_out = _t(Tw)

    def line_bm(y):
        """黑点 -> 中灰 的直线（log2 域），给阴影填充当参照"""
        return float(np.interp(_t(y), [tb_in, tm_in], [tb_out, tm_out]))

    # P25：朝直线靠 SHADOW_LIFT 比例（是"填暗部"不是"整体提亮"）
    t25_out = t25_in + cfg.SHADOW_LIFT * (line_bm(y25) - t25_in)

    # 膝点：落在中灰->白点直线上。肩部由"白点靶低于输入白点"自然形成，不人工造软肩。
    tk_out = float(np.interp(tk_in, [tm_in, tw_in], [tm_out, tw_out]))

    t_in = [_t(C.NOISE_FLOOR), tb_in, t25_in, tm_in, tk_in, tw_in]
    t_out = [_t(C.NOISE_FLOOR), tb_out, t25_out, tm_out, tk_out, tw_out]

    curve = ToneCurve(t_in, t_out)
    info = dict(
        applied=True, capped=capped,
        allow_lift=bool(allow_lift), gain_mid=float(g), gain_wanted=float(g_raw),
        target_mid_lin=Tm_eff, target_white_lin=Tw,
        ev_mid=float(np.log2(curve.gain(max(ym, C.NOISE_FLOOR)))),
        y_in=[float(yb), float(y25), float(ym), float(yk), float(yw)],
        y_out=[float(v) for v in np.exp2([tb_out, t25_out, tm_out, tk_out, tw_out])],
    )
    return curve, info


def fit_gamut(lin_rgb, Y=None):
    """线性域色域适配：通道超 1 时朝亮度方向收彩度，绝不硬裁（硬裁会变色相）。"""
    r = np.asarray(lin_rgb, np.float64)
    if Y is None:
        Y = color.Y_of(r)
    Y = np.clip(Y, 0.0, 0.999999)
    mx = r.max(axis=-1)
    over = mx > 1.0
    if not np.any(over):
        return np.clip(r, 0.0, 1.0)
    k = (mx - Y) / np.maximum(1.0 - Y, 1e-6)
    k = np.where(over, np.maximum(k, 1.0), 1.0)
    out = Y[..., None] + (r - Y[..., None]) / k[..., None]
    return np.clip(out, 0.0, 1.0)


def highlight_desat(lin_rgb, Y, cfg=C):
    """高光去饱和：胶片真实行为，同时天然避免单通道先削顶。"""
    if cfg.HILIGHT_DESAT <= 0.0:
        return lin_rgb
    lo = float(color.s2l(0.70))
    hi = float(color.s2l(0.97))
    m = color.smoothstep(Y, lo, hi) * cfg.HILIGHT_DESAT
    return lin_rgb + (Y[..., None] - lin_rgb) * m[..., None]


def correct(lin, rep, cfg=C):
    """lin 进 lin 出。返回 (lin_out, info)。

    动不动手，由"方向 + 源的提亮许可"两件事一起决定：
      hold   -> 本来就在靶上，不动
      below  -> 偏暗。RAW 允许提亮，JPG 不动（一放大就出伪色）
      compress -> 偏亮，压回去
    """
    dec = rep['decision']
    allow = bool(cfg.ALLOW_LIFT_RAW if rep.get('kind') == 'raw' else cfg.ALLOW_LIFT_JPG)

    if dec == 'hold' or (dec == 'below' and not allow):
        return lin.copy(), dict(applied=False, reason=dec, ev_mid=0.0, capped=False,
                                allow_lift=allow, gain_mid=1.0, gain_wanted=1.0,
                                target_mid_lin=float(color.s2l(cfg.TGT_MID)),
                                target_white_lin=float(color.s2l(cfg.TGT_WHITE)),
                                y_in=[], y_out=[])

    curve, info = build_curve(rep, cfg, allow_lift=allow)
    Y = color.Y_of(lin)
    out = lin * curve.gain(Y)[..., None]
    out = highlight_desat(out, color.Y_of(out), cfg)
    out = fit_gamut(out, color.Y_of(out))
    return out, info
