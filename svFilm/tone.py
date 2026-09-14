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
    # 中灰落点分三种（★ P0-2：阈值一律走 **L\***，与 analyze 同一把尺子）：
    #   compress（真的亮得离谱）-> 落到**过亮护栏线** GUARD_MID_L(87.0)：只把"离谱的那一截"收回来，
    #                             不再拽到中灰 TGT_MID_L(58.0)（那会把正常的亮片压闷）。
    #   dark（★ 有界兜底提亮，09-13 SV 拍板「乙」第 2 步）：靶 = **大师·高反差带的 L*50 下沿**。
    #       为什么不是 TGT_MID_L：那是**大师中位**，拽过去就是 09-13 园岭那个老病
    #       （每张都被拽到同一个中间灰）；而且实测把上限放开后会连高光一起压（L95 掉、亮点塌）。
    #       "只补进带、不追中位"既够用又天然有界。
    #   其余（below 的老兜底：入口没补过基线曝光的源）-> 落到 TGT_MID_L，行为不变。
    dark = bool(rep.get('dark_lift')) and bool(allow_lift)
    if rep.get('decision') == 'compress':
        # ★ P0-2：落点也切到 **L\***（GUARD_MID_L 87.0），与 analyze 的判据同一把尺子。
        Tm = float(color.lin_of_L(float(getattr(cfg, 'GUARD_MID_L', 87.0))))
        # 白点的上限也走"过曝专属"那一档，见下面 tw_out 的注释（09-13 SV 拍板「开顶」）。
        Tw = float(color.s2l(getattr(cfg, 'WHITE_CEIL', cfg.TGT_WHITE)))
    elif dark:
        Tm = float(color.lin_of_L(float(getattr(cfg, 'LIFT_DARK_FLOOR_L', 33.0))))
        Tw = float(color.s2l(cfg.TGT_WHITE))
    else:
        Tm = float(color.lin_of_L(float(getattr(cfg, 'TGT_MID_L', 58.0))))
        Tw = float(color.s2l(cfg.TGT_WHITE))

    # ---- 输入位置（log2） ----
    tb_in, t25_in, tm_in, tk_in, tw_in = _t(yb), _t(y25), _t(ym), _t(yk), _t(yw)

    # ---- 中灰增益：一个量，双向限幅 ----
    g_raw = Tm / max(ym, cfg.NOISE_FLOOR)
    if g_raw >= 1.0:                                   # 方向：提亮
        cap = float(cfg.EV_CAP_UP)
        if dark:                                       # 有界兜底提亮：上限收紧，防"图极暗时冲过头"
            cap = min(cap, float(getattr(cfg, 'LIFT_DARK_CAP_EV', cap)))
        g = min(g_raw, 2.0 ** cap) if allow_lift else 1.0
    else:                                              # 方向：压暗
        g = max(g_raw, 2.0 ** -cfg.EV_CAP_DOWN)
    Tm_eff = ym * g
    tm_out = _t(Tm_eff)
    capped = bool(abs(np.log2(g) - np.log2(max(g_raw, cfg.NOISE_FLOOR))) > 1e-3)

    # 黑点：朝绝对靶压，但 ① 最多压 BLACK_PULL 档（防"画面本来没有暗部"被压死）
    #              ② 绝不反过来把黑提亮（否则纯黑像素会顶到 1% 灰，暗部出现台阶）
    tb_out = min(tb_in, max(_t(Tb), tb_in - cfg.BLACK_PULL))

    # ---- 白点：也分两条路（与中灰同一个道理；「必须等于」是护栏层最贵的错误） ----
    # compress（真的过曝）：**只设上限** `WHITE_CEIL`（灰阶 245），不再"必须等于灰阶 220"。
    #   为什么这条路放开是安全的：过曝片的输入白点本来就 >= 落点 ⇒ 放开只可能"少压一点"，
    #   **永远不会放大颜色** —— 而"落点高于输入白点会把高光连色一起放大 3.4 倍"正是当年
    #   把落点定在 220 的原因。⇒ 09-13 SV 拍板的「开顶」：过曝片的高光不再被挤成 218~220 的平板。
    #   `max(..., tm_out)` 是单调性兜底：上限再低也不许压到中灰以下。
    # 其余（兜底提亮）：白点的输入常常远低于落点，落点必须守 TGT_WHITE，否则高光连色一起放大。
    # ★★ 09-14 早（SV：「继续」查 L90 为什么低 4）：**"没提亮"时也只设上限。**
    #   原来 fallback 路一律 `tw_out = _t(TGT_WHITE)`（灰阶 220 = **L\* 87.7** = "必须等于"），
    #   但 `ENTRY_SETTLE_ENABLE=True` 之后 L1 的增益被夹在 1.0（**不提亮**）⇒
    #   那条"必须有"的理由（防高光连色放大）已经不成立，只剩坏处：
    #   逐层追踪实测（0805/2328/0791/0999）——**入口刚交出来的 L90 是 91~98**
    #   （≈ 大师真胶片的 90.0），**L1 一刀砍到 74~80**，L2 只补回一部分 ⇒
    #   最终 L90 83~86，比大师低 4~7。⇒ 没提亮时改成**只设上限** `WHITE_CEIL`（灰阶 245 = L\* 96.5）。
    lift_on = bool(g > 1.0 + 1e-9)
    if rep.get('decision') == 'compress' or not lift_on:
        _cap_w = float(color.s2l(getattr(cfg, 'WHITE_CEIL', cfg.TGT_WHITE)))
        tw_out = max(min(tw_in, _t(_cap_w)), tm_out)
    else:
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
    # ★ P2-10 观测（不改行为，只记账）：非过曝路径白点是**绝对靶** `_t(Tw)`，
    #   对很暗的图（输入白点远低于靶）等于"把最亮端抬起来"。lab 验过 7 帧无损，
    #   但全库回归要专门看"暗片高光有没有被吹" ⇒ 把这件事量出来记进报告，别再靠印象。
    _path = 'compress' if rep.get('decision') == 'compress' else ('dark' if dark else 'fallback')
    _white_raise = float(np.log2(Tw / max(yw, C.NOISE_FLOOR))) if Tw > yw else 0.0
    info = dict(
        applied=True, capped=capped, dark=dark, path=_path,
        allow_lift=bool(allow_lift), gain_mid=float(g), gain_wanted=float(g_raw),
        target_mid_lin=Tm_eff, target_white_lin=Tw,
        white_raise_ev=_white_raise,               # >0 = 这张图的"最亮端"被曲线上抬了这么多档
        gain_white=float(np.exp2(tw_out) / max(yw, C.NOISE_FLOOR)),
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


def correct(lin, rep, cfg=C, allow_lift=None):
    """lin 进 lin 出。返回 (lin_out, info)。

    动不动手，由"方向 + 源的提亮许可"两件事一起决定：
      hold     -> 没超出上限护栏，不动
      below    -> 偏暗。**入口补过基线曝光的不再一律不亮**（09-13 SV 拍板「乙」第 2 步）：
                  判据改看这张图自己 —— 中间调低于「大师·高反差带下沿」L*33 时才允许
                  **有界**提亮（落点 = 带下沿，上限 LIFT_DARK_CAP_EV 档）；
                  入口补不了（非富士 / 读不到 tag）的老兜底照旧。
      compress -> 偏亮，压回来（这就是"救过曝"）

    ⚠ 这里的中灰只当**上限护栏**，不是"提亮靶"。拿整图中位数当靶会把每张图都拽到
      同一个中间灰（内容量冒充曝光量），09-13 园岭实测就是这个病。
    """
    dec = rep['decision']
    if allow_lift is None:
        allow = bool(cfg.ALLOW_LIFT_RAW if rep.get('kind') == 'raw' else cfg.ALLOW_LIFT_JPG)
    else:
        allow = bool(allow_lift)

    if dec == 'hold' or (dec == 'below' and not allow):
        return lin.copy(), dict(applied=False, reason=dec, ev_mid=0.0, capped=False,
                                dark=False, path='skip', white_raise_ev=0.0, gain_white=1.0,
                                allow_lift=allow, gain_mid=1.0, gain_wanted=1.0,
                                target_mid_lin=float(color.lin_of_L(float(getattr(cfg, 'TGT_MID_L', 58.0)))),
                                target_white_lin=float(color.s2l(cfg.TGT_WHITE)),
                                y_in=[], y_out=[])

    curve, info = build_curve(rep, cfg, allow_lift=allow)
    Y = color.Y_of(lin)
    out = lin * curve.gain(Y)[..., None]
    out = highlight_desat(out, color.Y_of(out), cfg)
    out = fit_gamut(out, color.Y_of(out))
    return out, info
