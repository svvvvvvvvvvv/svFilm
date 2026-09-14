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


def build_curve(rep, cfg=C):
    """由 L0 报告造曲线。返回 (curve, info)。**只压不提**（09-14：提亮交给锚点）。"""
    lp = rep['lin_pcts']
    yb, y25, ym, yk, yw = (lp[cfg.PCT_BLACK], lp[25.0], lp[cfg.PCT_MID],
                           lp[cfg.PCT_KNEE], lp[cfg.PCT_WHITE])

    # ---- 绝对靶：显示域 -> lin 域 ----
    Tb = float(color.s2l(cfg.TGT_BLACK))
    # ★★ 09-14 评审后重写（SV：「这些全修了」）：**L1 不再提亮**。
    #   位置的来源只剩两个 —— ① 入口的「听相机」落点 ② `io.anchor_ev` 的脸部锚点（只提不压）。
    #   所以这里**不再有 dark / 兜底提亮那条路**（它与锚点职责重叠，已删）。中灰只做一件事：
    #     真的过亮（`decision == 'compress'`）时把"离谱的那一截"收回到**过亮护栏线**；
    #     其余情况中灰**保持原位**（`g` 被夹在 ≤ 1，不再拽向任何靶）。
    if rep.get('decision') == 'compress':
        Tm = float(color.lin_of_L(float(getattr(cfg, 'GUARD_MID_L', 87.0))))
    else:
        Tm = float(color.lin_of_L(float(getattr(cfg, 'TGT_MID_L', 58.0))))
    Tw = float(color.s2l(getattr(cfg, 'WHITE_CEIL', cfg.TGT_WHITE)))

    # ---- 输入位置（log2） ----
    tb_in, t25_in, tm_in, tk_in, tw_in = _t(yb), _t(y25), _t(ym), _t(yk), _t(yw)

    # ---- 中灰增益：**只压不提**（上限锁死 1.0，下限仍由 EV_CAP_DOWN 兜住） ----
    g_raw = Tm / max(ym, cfg.NOISE_FLOOR)
    g = float(np.clip(g_raw, 2.0 ** -float(cfg.EV_CAP_DOWN), 1.0))
    Tm_eff = ym * g
    tm_out = _t(Tm_eff)
    capped = bool(abs(np.log2(g) - np.log2(max(g_raw, cfg.NOISE_FLOOR))) > 1e-3)

    # 黑点：朝绝对靶压，但 ① 最多压 BLACK_PULL 档（防"画面本来没有暗部"被压死）
    #              ② 绝不反过来把黑提亮（否则纯黑像素会顶到 1% 灰，暗部出现台阶）
    tb_out = min(tb_in, max(_t(Tb), tb_in - cfg.BLACK_PULL))

    # ---- 白点：**只设上限**（「必须等于」是护栏层最贵的错误） ----
    # ★★ 09-14 评审后：L1 不再提亮 ⇒ 那条"落点必须守 TGT_WHITE（否则高光连色一起放大）"的
    #   理由**彻底消失**，两条路合并成一条 —— 无论过不过曝，都只给白点一个上限 `WHITE_CEIL`
    #   （灰阶 245 = L\* 96.5，≈ 大师真胶片 L99 97.0）。`max(..., tm_out)` 是单调性兜底：
    #   上限再低也不许压到中灰以下。
    tw_out = max(min(tw_in, _t(Tw)), tm_out)

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
    # ★ P2-10 观测（不改行为，只记账）：白点现在**只会被下压、永不抬**
    #   （`tw_out = min(tw_in, 上限)`）。所以 `white_raise_ev` 改成记**实际抬了多少档**，
    #   口径 = `tw_out − tw_in`（恒 ≤ 0）—— 旧口径拿"靶 vs 输入"算，在"靶只是上限"之后会假报正数。
    _white_raise = max(0.0, float(tw_out - tw_in))
    info = dict(
        applied=True, capped=capped,
        path=('compress' if rep.get('decision') == 'compress' else 'pass'),
        gain_mid=float(g), gain_wanted=float(g_raw),
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


def correct(lin, rep, cfg=C):
    """lin 进 lin 出。返回 (lin_out, info)。

    动不动手只看一件事：**要不要往下压**。
      hold     -> 没超出上限护栏，不动
      below    -> 偏暗 —— **L1 不提亮**（09-14：位置归入口 settle + 脸锚点，兜底提亮那套已删）
      compress -> 偏亮，压回来（这就是"救过曝"）

    ⚠ 这里的中灰只当**上限护栏**，不是"提亮靶"。拿整图中位数当靶会把每张图都拽到
      同一个中间灰（内容量冒充曝光量），09-13 园岭实测就是这个病。
    """
    dec = rep['decision']
    if dec in ('hold', 'below'):
        return lin.copy(), dict(applied=False, reason=dec, ev_mid=0.0, capped=False,
                                path='skip', white_raise_ev=0.0, gain_white=1.0,
                                gain_mid=1.0, gain_wanted=1.0,
                                target_mid_lin=float(color.lin_of_L(float(getattr(cfg, 'TGT_MID_L', 58.0)))),
                                target_white_lin=float(color.s2l(cfg.TGT_WHITE)),
                                y_in=[], y_out=[])

    curve, info = build_curve(rep, cfg)
    Y = color.Y_of(lin)
    out = lin * curve.gain(Y)[..., None]
    out = highlight_desat(out, color.Y_of(out), cfg)
    out = fit_gamut(out, color.Y_of(out))
    return out, info
