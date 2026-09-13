# -*- coding: utf-8 -*-
r"""L3 —— 局部层。三件事，顺序 = **色度 → 位置 → 形状**：
  ① 肤色保护 —— 风格层会把颜色往胶片方向收，肤色最容易被收坏。
     拿 L1 修正完的成片当参考，哪里是肤色、就把那里的色度按比例还给参考值。
     明度用当前的，色度用参考的 —— 只还"色"，不还"亮"。

  ② 肤色正向达标 —— 只在"保护"补不回来时起作用（09-13 SV 拍板「全」档 + 过亮门）。
     目标 = 大师脸区实测中位（a* 16.3 / b* 18.5，`_debug/analysis/skin_master_ruler.md`）。
     **逐像素**：每个皮肤像素自己看"离目标还差多少"，差多少补多少；**只向上补**；
     单像素位移设上限（a* 8 / b* 12，护唇妆/腮红）；**两轴都低才补**（某轴已高过目标
     就完全不碰 —— 挡粉衣服/木色被顺手推暖）；过亮门**逐像素**收力
     （L* ≤ 73.3 全强度 / ≥ 83.7 不补，逆光高调脸的高光部分自己不补、暗部照补）。

  ③ ★ **脸**（`face_tone`，09-13 深夜 SV 拍板；出厂 `FACE_ENABLE=False` ⇒ 逐位不变）——
     **位置**（这张脸该多亮）+ **形状**（这张脸有多立体）。这两件事要的是"人在哪、脸在哪"，
     只能靠分割 + 五官点拿（`face.py`），**不是** ①② 用的那个颜色软掩膜。详见 `face_tone` 的 docstring。
     ⚠ 排在最末：它的锚点（脸中位）必须在①②和影调曲线都定完之后才取。

（真正的大局：局部后面还会长柔化/颗粒/黑柔，那是空间域的活，
  用独立一层做，不塞进颜色层。）
"""
import numpy as np

from . import color
from . import face
from . import config as C


def skin_mask(disp):
    """软掩膜 0~1：色相 5~55 度、有彩度、明度在中段。门限见 `config.SKIN_MASK_*`（P1-3）。"""
    lab = color.to_lab(np.clip(disp, 0.0, 1.0))
    h = color.hue_deg(lab)
    c = color.chroma(lab)
    L = lab[..., 0]
    _hlo = tuple(getattr(C, 'SKIN_MASK_HUE_LO', (2.0, 14.0)))
    _hhi = tuple(getattr(C, 'SKIN_MASK_HUE_HI', (46.0, 66.0)))
    _clo = tuple(getattr(C, 'SKIN_MASK_C_LO', (5.0, 13.0)))
    _chi = tuple(getattr(C, 'SKIN_MASK_C_HI', (70.0, 95.0)))
    _llo = tuple(getattr(C, 'SKIN_MASK_L_LO', (12.0, 22.0)))
    _lhi = tuple(getattr(C, 'SKIN_MASK_L_HI', (86.0, 95.0)))
    m = color.smoothstep(h, _hlo[0], _hlo[1]) * (1.0 - color.smoothstep(h, _hhi[0], _hhi[1]))
    m *= color.smoothstep(c, _clo[0], _clo[1]) * (1.0 - color.smoothstep(c, _chi[0], _chi[1]))
    m *= color.smoothstep(L, _llo[0], _llo[1]) * (1.0 - color.smoothstep(L, _lhi[0], _lhi[1]))
    return m


def _protect(ref_disp, disp, cfg):
    """肤色保护：色度按 SKIN_PROTECT_STRENGTH 往 L1 的参考值还。"""
    cur = np.clip(disp, 0.0, 1.0)
    ref = np.clip(ref_disp, 0.0, 1.0)
    m = skin_mask(cur)                     # 只这一次是全图 Lab
    cov = float(np.mean(m))
    if cov < float(getattr(C, 'SKIN_PROTECT_MIN_COV', 1.0e-4)):
        return cur, dict(skin_cov=0.0)
    sel = m > float(getattr(C, 'SKIN_PROTECT_SEL', 1.0e-3))
    if not np.any(sel):
        return cur, dict(skin_cov=cov)
    lab_c = color.to_lab(cur[sel])
    lab_r = color.to_lab(ref[sel])
    w = (m[sel] * cfg.SKIN_PROTECT_STRENGTH)[:, None]
    lab_c[:, 1:] = lab_c[:, 1:] * (1.0 - w) + lab_r[:, 1:] * w
    out = cur.copy()
    out[sel] = np.clip(color.from_lab(lab_c), 0.0, 1.0)
    return out, dict(skin_cov=cov)


def skin_floor(disp, cfg=C):
    """肤色正向达标：**逐像素**把"低于大师脸的皮肤"抬到档上。只动色度。

    每个皮肤像素自己算"离目标还差多少"，差多少补多少；三个门决定它使多大劲：
      · 软掩膜 m           —— 越像皮肤越使劲；
      · 过亮门（逐像素）    —— 亮度超过大师脸区分布上端就收力（逆光/高调脸不硬补）；
      · 两轴都低门          —— 某一轴已经高过目标（粉衣服/木色/唇妆）就完全不碰。

    ⚠ 不用"全图皮肤中位"这类整体统计量。试过，不行：掩膜会把粉衣服/木色/路面算进来
    （实测一张 15.3% 的像素被判成皮肤），中位被拉低之后**人脸会被推过头**
    （实测 b* 冲到 20.4，目标 18.5）。逐像素做就没有这个问题。
    """
    if not getattr(cfg, 'SKIN_FLOOR', False):
        return np.clip(disp, 0.0, 1.0), dict(applied=False, reason='off')

    d = np.clip(disp, 0.0, 1.0)
    m = skin_mask(d)
    if int((m > 0.5).sum()) < int(getattr(C, 'SKIN_MASK_MIN_PX', 200)):
        return d, dict(applied=False, reason='no_skin')
    sel = m > float(getattr(C, 'SKIN_MASK_SEL', 0.02))

    lab = color.to_lab(d[sel])
    a, b, L = lab[:, 1], lab[:, 2], lab[:, 0]
    aT, bT = cfg.SKIN_FLOOR_A, cfg.SKIN_FLOOR_B

    gate_L = 1.0 - color.smoothstep(L, cfg.SKIN_FLOOR_L_LO, cfg.SKIN_FLOOR_L_HI)
    excess = np.maximum(a - aT, b - bT)
    both_low = 1.0 - color.smoothstep(excess, 0.0, cfg.SKIN_FLOOR_EXCESS)
    w = m[sel] * gate_L * both_low

    info = dict(applied=False, a_med=float(np.median(a)), b_med=float(np.median(b)),
                L_med=float(np.median(L)), gate=float(np.mean(gate_L)),
                touched=float(np.mean(w > 1e-3)))
    if float(np.max(w)) < 1e-4:
        info['reason'] = 'nothing_to_do'
        return d, info

    lab[:, 1] = a + np.clip(aT - a, 0.0, cfg.SKIN_FLOOR_A_MAX) * w
    lab[:, 2] = b + np.clip(bT - b, 0.0, cfg.SKIN_FLOOR_B_MAX) * w
    out = d.copy()
    out[sel] = np.clip(color.from_lab(lab), 0.0, 1.0)
    info['applied'] = True
    return out, info


def _region_weight(sel, cfg):
    r"""把选中的**脸区域**羽化成力道权重（和"正脸框那套"同构：σ = `FACE_FEATHER_REL` × 区域宽）。

    为什么要它：形状那一下（A1 拉开明暗）要落在**脸这一块**上，硬边界会在脸上留下台阶。
    老路径用 `face.parse` 给的"框羽化 × person"，新路径没有框 ⇒ 用选中的区域自己羽化。
    返回 `None` 表示区域是空的（调用方退回直接用 `person_weight`）。
    """
    import cv2
    if sel is None or not np.any(sel):
        return None
    _ys, xs = np.where(sel)
    bw = float(xs.max() - xs.min() + 1)
    sig = max(3.0, float(getattr(cfg, 'FACE_FEATHER_REL', 0.05)) * bw)
    return np.clip(cv2.GaussianBlur(sel.astype(np.float32), (0, 0), sig), 0.0, 1.0)


def _lift_weight(pw, m, edge_px):
    r"""提亮的**力道权重** = 人物权重 × **到人物轮廓的淡出**（09-14 SV 拍板「中」）。

    为什么需要它（SV 实测：提亮后"在环境中亮度会有点割裂"）：
      `person_weight` 的淡出是由**分割置信度**定的，而置信度在轮廓上是"啪"地从 1 掉到 0
      ⇒ `_debug/lab_person_falloff.py` 实测**淡出只有 3~6 px**（画面 1365~2048 宽）= 一刀切。
    做法：**按几何距离**补一道淡入 —— 轮廓处力道 = 0，往里 `edge_px` 像素涨到满力（smoothstep 收两端）。
      `edge_px` 由调用方按 `FACE_LIFT_EDGE_REL × 画面短边` 算好传进来（便于单测直接钉死宽度）。
    ★ **只在人物掩膜内部衰减**（算完再乘 `pw`，而 `pw` 在背景恒为 0）
      ⇒ 「背景一个像素不碰」这条铁律不破。
    ⚠ 代价：越宽，人身上**细的地方**（手臂/发丝/肩外沿）提得越少。
    """
    import cv2
    pm = (np.asarray(m['person'], np.float32) > 0.5).astype(np.uint8)
    if int(pm.sum()) < 16:                     # 掩膜太小/拿不到 ⇒ 不做衰减（别把力道全吃掉）
        return pw
    dist = cv2.distanceTransform(pm, cv2.DIST_L2, 5)
    t = np.clip(dist / max(float(edge_px), 1e-6), 0.0, 1.0)
    t = t * t * (3.0 - 2.0 * t)                # smoothstep：靠轮廓更平、中段更陡
    return pw * t


def face_tone(disp, cfg=C):
    r"""L3 的「脸」：① **位置**（只提人物，背景零改动）② **形状**（A1 放大已有的立体感）。

    为什么排在这一层、这个位置：
      * 影调曲线（L2）和黑柔都在压脸 ⇒ 这一层排在它们**后面**，量到的才是"最终要交出去的那张脸"。
      * 排在肤色那两件事**之后**：那两件事的过亮门读的是当前 L\*，先动明度会把它们的判据改掉。
        ⇒ 层内顺序 = **色度 → 位置 → 形状**。
      * ★ **只碰明度**（`color.retone_L` 同步缩放 RGB，色相/彩度关系不动）。
      * ★ 两个动作都只在**人物权重**上做，而该权重在**背景掩膜里恒为 0** ⇒ 背景一个像素不碰。

    两句规则（SV 定；09-14 把位置靶提到 68）：
      ① 位置：`补光量 = max(0, 靶 − 脸中位)`，**只提不压**。
         靶 = 鹿井「脸 L*」的 **中位 67.9 ⇒ 取 68**（09-14 SV 拍板「乙」；原来用的是 p25 = 62）。
         脸本来就在靶之上的片子（0791 脸 90.6 / 2328 脸 89.8）⇒ 补光量 = 0 ⇒ 一格不动。
         ★ 力道**到人物轮廓要淡出去**（09-14 SV 拍板「中」）：见 `_lift_weight` —— 不然人一亮、
           环境不亮，交界处会"割裂"（实测原来的淡出只有 3~6 px）。
      ② 形状：以**脸自己的中位**为锚，把已有的明暗拉开 `k = 线/现状`（≤2.4 倍），**不编光**。
         线只有一条 = 鹿井「脸内部跨度（框内皮肤 P90−P10）」的 p25 = **35**（`FACE_TGT_SPAN`）。
         ⚠ **光方向（左右差）那条线只量、不参与放大**（`FACE_DIR_MEASURE`，09-14 定）。
           09-14 试过把它当"形状的第二条线"（两条线取更高那个），实测**会把已经很立体的脸整张炸开**：
           DSCF2236 跨度 44.8 → 69.4（它本来就远超 35 ⇒ 跨度那条不发力，全是方向那条干的），左右差几乎没动。
           根因：那条线要的是**左右方向**的差，A1 的 k 却是**各向同性**放大 ⇒ 拿一个横向指标去拉总反差。
           要用它只能"只放大左右那个分量"（= A2 编光，SV 已否）⇒ **这条线到此为止**。证据写在 `config.py`。
         变亮那部分最多到 L\*97（不动暗部）。
      ⚠ **顺序不可换**：形状的锚点（脸中位）必须在**位置定好之后**才取，否则锚点是错的。
    ★ 这一层会被 `pipeline.run` 在 **L2 之后 / 空间层之后 / L3** 各调一次
      （第 4 条「每层护脸」，`FACE_GUARD_LAYERS`）—— 因为把脸压平的主力是**黑柔 + 颗粒**，
      在末尾补一道到不了靶。靶是绝对值 ⇒ 后层压下去、下一道就补回来，重复调用是**幂等收敛**的。

    ★ 「脸在哪」（09-14 SV 选「甲」）：两个动作用脸的**方式不一样**，这点容易混 ——
      · **提亮** = 提**整个人**（力道落 `person_weight`，脸/手/衣服一起动）。脸在这里是**尺子**：
        用"脸有多亮"代表"这个人有多亮"（用人的平均会被暗衣服、背影、背景漏进来的人边带偏）
        ⇒ 脸只贡献**一个数**。
      · **立体感** = **只落在脸上**（力道落 脸区域 × `person_weight`）。因为这是在脸上改**局部反差**，
        圈到手臂就等于给胳膊编造光影 ⇒ 脸必须有**准确的位置**。
      · 定位优先走 `face.face_region()`（**单独的脸皮肤类 + 头窗口**，不依赖正脸检测框，
        正侧脸/背影/小脸都能拿到）；拿不到才回落老的正脸框路径。`info['how']` 记录用了哪条。
    """
    d = np.clip(disp, 0.0, 1.0)
    info = dict(applied=False)
    if not getattr(cfg, 'FACE_ENABLE', False):
        info['reason'] = 'off'
        return d, info
    try:
        st = face.parse(d)
    except face.FaceUnavailable as e:                      # 缺依赖/缺模型 ⇒ 优雅降级
        info.update(reason='no_dep', err=str(e))
        return d, info
    pw = st['person_weight']
    if pw is None or float(np.max(pw)) < 1e-3:
        info['reason'] = 'no_person'
        return d, info
    f = st['face']

    lin = color.s2l(d)
    L = color.L_of_lin(color.Y_of(lin))
    # ★★ 脸在哪（09-14 SV 选「甲」）：优先 **"脸皮肤 + 头窗口"** —— 不依赖正脸检测框。
    #   正脸检测器对**正侧脸 / 背影 / 小脸**基本给不出框（1954 正侧脸机内 0 个候选；
    #   0999 小脸被眼距闸挡掉，而它框里明明有 2025 个皮肤像素）。拿不到才回落老的"正脸框"
    #   路径 ⇒ **绝不会比改动前更差**。详见 `face.face_region()`。
    sel, how = face.face_region(st['masks'], cfg)
    if sel is not None:
        w_geo = _region_weight(sel, cfg)
        w = pw if w_geo is None else (w_geo * pw)   # 再乘人物权重 ⇒ 背景处仍恒为 0
    elif f is not None:
        x0, y0, x1, y1 = f['box']
        boxm = np.zeros(L.shape, bool)
        boxm[y0:y1, x0:x1] = True
        sel = boxm & (st['masks']['skin'] > 0.5)
        if int(sel.sum()) < int(getattr(cfg, 'FACE_MIN_SKIN_PX', 200)):
            sel = boxm
        w = f['weight']
        how = 'box'
    else:
        info['reason'] = 'no_face(%s)' % how
        return d, info
    Lb = float(np.median(L[sel]))

    # ---- ① 位置：只提不压（★ 力道到人物轮廓要淡出去，见 `_lift_weight`）----
    dL = float(np.clip(float(getattr(cfg, 'FACE_TGT_L', 62.0)) - Lb, 0.0,
                       float(getattr(cfg, 'FACE_LIFT_MAX', 40.0))))
    edge_px = max(2.0, float(getattr(cfg, 'FACE_LIFT_EDGE_REL', 0.045)) * float(min(L.shape)))
    L1 = (L + _lift_weight(pw, st['masks'], edge_px) * dL) if dL > 1e-3 else L

    # ---- ② 形状：以脸中位为锚拉开已有的明暗（A1：各向同性，只放大已有的，不编光）----
    q10, q90 = np.percentile(L1[sel], [10.0, 90.0])
    span = float(q90 - q10)
    k = float(np.clip(float(getattr(cfg, 'FACE_TGT_SPAN', 35.0)) / max(span, 1e-6),
                      1.0, float(getattr(cfg, 'FACE_SPAN_KMAX', 2.4))))
    # ★ 光方向（左右差）：**只量、不改画面**。逐张把「哪半边脸更亮」记进 info，供将来用。
    #   ⚠ 它**不参与 k** —— 09-14 试过把它当"形状的第二条线"，实测会把已经很立体的脸整张炸开
    #   （DSCF2236 跨度 44.8 → 69.4，而左右差几乎没动）。根因：那条线要的是**左右方向**的差，
    #   而这里的 k 是**各向同性**放大 ⇒ 拿一个横向指标去拉整张脸的总反差。证据与结论写在 config 那一段。
    asym, n_pair, dir_how = None, 0, 'off'
    if bool(getattr(cfg, 'FACE_DIR_MEASURE', True)):
        try:
            asym, n_pair, dir_how = face.lr_asym(
                L1, st['masks']['face_skin'], sel,          # ★ 传"脸皮肤"那一类，别传 skin（skin = 脸+身体）
                lm=(f or {}).get('lm'), box=(f or {}).get('box'),
                eyed=(f or {}).get('eyed'), cfg=cfg)
        except Exception:                                       # noqa: BLE001
            asym, n_pair, dir_how = None, 0, 'err'
        if asym is not None and abs(asym) < float(getattr(cfg, 'FACE_DIR_MIN', 0.030)):
            asym = None                    # 量出来太小 ⇒ 认为"这张脸没方向" ⇒ **拒答**，不报方向
    if dL <= 1e-3 and k <= 1.0 + 1e-9:
        info.update(reason='nothing_to_do', face_L=Lb, span=span, k=k, lift=0.0, how=how,
                    asym=asym, n_pair=n_pair, dir_how=dir_how, edge_px=float(edge_px))
        return d, info
    Ls = float(np.median(L1[sel]))
    cap = float(getattr(cfg, 'FACE_TOP_CAP', 97.0))
    Lx = Ls + k * (L1 - Ls)
    Lx = np.where(Lx > L1, np.minimum(Lx, np.maximum(L1, cap)), Lx)
    L2 = L1 + w * (Lx - L1)
    # ★ 只在有人物权重的像素上重调亮度 ⇒ 权重恒 0 的地方（**整个背景**）原样搬过来，
    #   连浮点残差都不留（否则 retone_L 会把"权重 0"的像素挪动 ~0.02 L*，肉眼不可见但不再是"一个像素不碰"）。
    #   门用 `pw`（人物权重）而不是 `w`：位置那一下是在 `pw` 上做的，`w ⊆ pw > 0`，用 pw 才不会漏掉位置。
    out = d.copy()
    wsel = pw > 1.0e-6
    if np.any(wsel):
        out[wsel] = color.retone_L(lin[wsel], L2[wsel])
    info.update(applied=True, how=how, face_L_before=Lb, face_L=Ls, lift=dL,
                span=span, k=k, asym=asym, n_pair=n_pair, dir_how=dir_how,
                edge_px=float(edge_px),
                person_cov=float(np.mean(pw)), feather_cov=float(np.mean(w)))
    return out, info


def apply(ref_disp, disp, cfg=C):
    """ref_disp = L1 修正后的成片；disp = 当前（过完风格 + 空间域）的成片。"""
    out = np.clip(disp, 0.0, 1.0)
    info = {}
    if cfg.SKIN_PROTECT and cfg.SKIN_PROTECT_STRENGTH > 0.0:
        out, pinfo = _protect(ref_disp, out, cfg)
        info.update(pinfo)
    else:
        info['skin_cov'] = 0.0
    if getattr(cfg, 'SKIN_FLOOR', False):
        out, finfo = skin_floor(out, cfg)
        info['skin_floor'] = finfo
    if getattr(cfg, 'FACE_ENABLE', False):          # ★ 色度做完才动明度（见 face_tone 的注释）
        out, tinfo = face_tone(out, cfg)
        info['face_tone'] = tinfo
    return out, info
