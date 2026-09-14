# -*- coding: utf-8 -*-
"""入口关口 + 出口。**整个工程里唯一允许出现"机型"三个字的地方。**

入口要做完三件事，做完之后下游不许再关心这张图是谁拍的：
  1) 解码：RAW 走 rawpy（线性、相机白平衡、关自动亮度）；JPG 走 PIL。
  2) IDT（输入设备变换）：白平衡 + 色彩矩阵 + 黑白电平 —— RAW 由 rawpy 一次做完；
     JPG 已经在相机里做完，拿不回来，所以是"显式降级"。
  3) 归一：对外只吐 lin（白点 1.0）、disp（0~1）两个域。
"""
from __future__ import annotations

import io as _stdlib_io
import os

import numpy as np
from PIL import Image, ImageOps

from . import color
from . import config as C
from . import rawmeta

try:
    import cv2
except Exception:      # pragma: no cover
    cv2 = None

RAW_EXT = {'.raf', '.arw', '.cr2', '.cr3', '.nef', '.dng', '.orf', '.rw2', '.pef', '.srw', '.raw'}
STD_EXT = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.webp'}


class Sample:
    __slots__ = ('lin', 'disp', 'kind', 'path', 'name', 'exif', 'cam')

    def __init__(self, lin, disp, kind, path, exif=None, cam=None):
        self.lin = lin
        self.disp = disp
        self.kind = kind
        self.path = path
        self.name = os.path.splitext(os.path.basename(path))[0]
        self.exif = exif
        self.cam = cam if cam is not None else {}


def kind_of(path):
    e = os.path.splitext(path)[1].lower()
    if e in RAW_EXT:
        return 'raw'
    if e in STD_EXT:
        return 'std'
    raise ValueError('不认识的扩展名: %s' % path)


def _resize(arr, max_side):
    """缩到长边 max_side。大图先缩后转浮点 —— 40MP 转 float32 再缩要多花好几秒。"""
    h, w = arr.shape[:2]
    long_side = max(h, w)
    if max_side is None or long_side <= max_side:
        return arr
    s = float(max_side) / float(long_side)
    nw, nh = int(round(w * s)), int(round(h * s))
    if nw < 1 or nh < 1:
        raise ValueError('缩得太小')
    if cv2 is not None and arr.dtype == np.uint16:
        return cv2.resize(arr, (nw, nh), interpolation=cv2.INTER_AREA)
    if cv2 is not None:
        return cv2.resize(arr.astype(np.float32), (nw, nh),
                          interpolation=cv2.INTER_AREA).astype(np.float64)
    ch = [np.asarray(Image.fromarray(arr[..., i].astype(np.float32), mode='F')
                     .resize((nw, nh), Image.BILINEAR), np.float64) for i in range(arr.shape[-1])]
    return np.stack(ch, axis=-1)


def _exif_orientation_fixed(im):
    """exif_transpose 之后把 orientation 标签抹成 1，否则存回去会被转两次。"""
    ex = im.getexif()
    if not ex:
        return None
    if 274 in ex:
        ex[274] = 1
    try:
        return ex.tobytes()
    except Exception:
        return None


def idt_wb(lin, cfg=C):
    """IDT 的最后一步：白平衡。默认关。

    为什么默认关：RAW 走相机白平衡、JPG 是相机已经定死的，绝大多数情况本来就是对的。
    乱做自动白平衡会把黄金时刻的暖调当成"偏色"抹平。
    只在输入明显拍歪（白炽灯下没改色温）时手动打开。

    做法（灰世界，但只在"近中性像素"上算）：取低饱和度像素的线性均值，
    求一组限幅增益。不做则返回原图。
    """
    if not cfg.WB_ENABLE:
        return lin, dict(applied=False)
    disp = np.clip(color.l2s(np.clip(lin, 0.0, None)), 0.0, 1.0)
    s = color.sat_hsv(disp)
    g = color.gray_of(disp)
    sel = (s < cfg.WB_NEUTRAL_SAT) & (g > 0.05) & (g < 0.95)
    cov = float(np.mean(sel))
    if cov < cfg.WB_MIN_COVER:
        return lin, dict(applied=False, cover=cov)
    m = lin[sel].mean(axis=0)
    m = np.maximum(m, cfg.NOISE_FLOOR)
    gain = float(m.mean()) / m
    gain = 1.0 + (gain - 1.0) * float(cfg.WB_STRENGTH)
    gain = np.clip(gain, 1.0 - cfg.WB_MAX_GAIN, 1.0 + cfg.WB_MAX_GAIN)
    return lin * gain, dict(applied=True, cover=cov, gain=[float(v) for v in gain])


def load_std(path, max_side=C.MAX_SIDE):
    im = Image.open(path)
    im = ImageOps.exif_transpose(im).convert('RGB')
    arr = np.asarray(im, np.float64) / 255.0
    arr = _resize(arr, max_side)
    disp = np.clip(arr, 0.0, 1.0)
    lin, wb_info = idt_wb(color.s2l(disp))
    disp = np.clip(color.l2s(lin), 0.0, 1.0)
    return Sample(lin, disp, 'jpg', path, _exif_orientation_fixed(im),
                  dict(kind='jpg', wb=wb_info, note='无 IDT 余量（相机曲线已压过）'))


def apply_entry_curve(lin, curve):
    """按实测相机曲线补入口：三通道乘同一个"亮度 → 增益"，**只动亮度、不改色相**。

    契约里"曲线只在亮度域做"指的就是这条。曲线本身由
    `_debug/calib_entry_curve_from_pairs.py` 从真实 RAW+机内 JPEG 对量出来。
    """
    _, xs, gs = curve
    Y = color.luma(np.clip(lin, 0.0, None))
    return lin * np.interp(Y, xs, gs)[..., np.newaxis]


def scene_exposure_index(lin):
    """场景曝光指数 `e = log2(场景线性亮度中位 / 0.18)`。

    「这张图整体有多少档高于/低于 18% 灰」。与 `_debug/lab_pos_law_fit.py` 的口径逐字一致：
    那边写的是 `L_of_lin(median(luma(lin)))` 再 `log2(lin_of_L(·)/0.18)`；因 `L_of_lin` 与
    `lin_of_L` 互逆、且**单调变换与中位可交换** ⇒ 等于本式。**改口径会让规律偏掉**。

    只为省时间对大图抽稀（中线数对抽稀不敏感）。
    """
    y = color.luma(np.clip(lin, 0.0, None))
    if y.size > 3_000_000:
        k = int(np.ceil(np.sqrt(y.size / 3_000_000.0)))
        y = y[::k, ::k]
    return float(np.log2(max(float(np.median(y)), 1e-9) / 0.18))


def entry_tone(lin, ev, cfg=C, level=None):
    """入口成形（09-13 SV 拍板「乙」）：**场景线性光 + 一条固定的亮度/反差/肩部/趾部曲线**。

    与 `apply_entry_curve` 的分工：
      * 旧：按 机型×DR 实测的「RAW → 机内 JPEG」逐亮度增益 —— **把相机的机内风格一起搬进来**
        （实测把动态范围压掉 ~2.5 倍、2328 有 13% 像素被顶穿 1.0 ⇒ 高光砸成平板）。
      * 新：只做两件与口味无关的事 ——
          ① 零点：`× 2^ev`（ev = 实测 mid_ev，只补"相机故意欠曝"那几档）⇒ 近似**场景线性光**；
          ② 形状：`y' = a · y^ENTRY_GAMMA` + **软肩部**（超过 ENTRY_KNEE 平滑压向 ENTRY_CEIL）
             + **趾部**（`ENTRY_TOE`：y 低于"出口亮度中位 × ENTRY_TOE_HI_REL"那一段收下来，
               越低收得越多，中灰以上**逐位不动**）。

    落点 `a`：默认 = 全局 `cfg.ENTRY_LEVEL`；**`level` 传了就用它** ——
    「位置逐张听相机」（`cameras.entry_settle_level`）走的就是这条路，
    把"全库一个落点"换成"逐张查机型×DR 实测规律"（见 cameras.py `SETTLE_LAW`）。

    **趾部（v0.3.8，09-13 深夜 SV 拍板「对齐鹿井」）**：上面的形状原来**只有肩部没有趾部**
    ⇒ `Y^0.886 · a` 在 Y→0 时增益无上限，实测把黑位放大 ×12（相机给的 L\\*3.0 → 出口 L\\*21.8），
    而中灰/高光本来是准的 ⇒ 整片"发灰发奶"。
    趾部只压 **自己最低那一小段**：断点 = `出口亮度 y 的中位 × ENTRY_TOE_LO/HI_REL`（**内容归一**，
    不是绝对亮度 —— 绝对断点在暗片上会让整张图都掉进趾部，见 config 注释）。
    `ENTRY_TOE = 1.0` = 关掉（逐位等于改之前）。

    仍然**只动亮度、不改色相**（三通道乘同一个"亮度→增益"，与 `apply_entry_curve` 同契约）。
    """
    g = float(getattr(cfg, 'ENTRY_GAMMA', 1.0))
    a = float(getattr(cfg, 'ENTRY_LEVEL', 1.0) if level is None else level)
    knee = float(getattr(cfg, 'ENTRY_KNEE', 1.0))
    ceil = float(getattr(cfg, 'ENTRY_CEIL', 1.0))
    Y = np.maximum(color.luma(np.clip(lin, 0.0, None)), 1e-9)
    y = np.power(np.maximum(Y * (2.0 ** float(ev)), 1e-9), g) * a
    d = max(ceil - knee, 1e-6)
    # ★ 肩的形状族（09-14 晚）：`log` 比 `exp` 在高光段保得住层次（见 config 那段注释）。
    #   两者都严格有界于 ceil ⇒ 都不会触发入口裁切护栏。
    if str(getattr(cfg, 'ENTRY_SHOULDER_KIND', 'exp')).lower() == 'log':
        umax = max(float(getattr(cfg, 'ENTRY_SHOULDER_UMAX', 12.0)), 1e-6)
        # ⚠ 必须把 u 夹到 UMAX —— 不夹的话 u>UMAX 时 log1p(u)/log1p(UMAX) > 1
        #   ⇒ **输出会冲破 ENTRY_CEIL**（实测 knee.80/ceil.985/UMAX2 会跑到 L* 105），
        #   那就不是"有界不裁"了，还会顶出真正的死白。
        u = np.clip((y - knee) / d, 0.0, umax)
        np.copyto(y, knee + d * np.log1p(u) / np.log1p(umax), where=(y > knee))
    else:
        np.copyto(y, knee + d * (1.0 - np.exp(-(y - knee) / d)), where=(y > knee))
    # ---- 趾部（09-13 深夜 SV 拍板「对齐鹿井」）：最低那一段按 TOE 倍走，到 HI 之上完全不动 ----
    # ★ 断点走**内容归一**（= 出口亮度 y 的中位 × 固定倍数），不是绝对亮度。
    #   为什么要这样：固定绝对断点在亮场上正好，在**暗片**上整张图都落在断点之下
    #   ⇒ 趾部退化成"全图乘 0.32"（实测 0071 中灰 28.8 → 16.5，那不是"只动最底部"）。
    #   与项目既有原则一致：「形状抄大师**内容归一后**的形状，不能抄绝对亮度」。
    toe = float(getattr(cfg, 'ENTRY_TOE', 1.0))
    if toe < 1.0 - 1e-9:
        _k = max(1, int(y.size // 3_000_000))
        ym = float(np.median(y.reshape(-1)[::_k]))
        if ym > 1e-9:
            t0 = float(getattr(cfg, 'ENTRY_TOE_LO_REL', 0.08)) * ym
            t1 = float(getattr(cfg, 'ENTRY_TOE_HI_REL', 0.84)) * ym
            t = np.clip((y - t0) / max(t1 - t0, 1e-9), 0.0, 1.0)
            y = y * (toe + (1.0 - toe) * t * t * (3.0 - 2.0 * t))
    return lin * (y / Y)[..., np.newaxis]


# ========== 「脸的锚点决定位置」＝ 把入口那条曲线**重打一个曝光偏移**（09-14 SV 选「乙」） ==========
# 为什么能这么做：入口曲线上任意像素的输出**只依赖 `Y·2^ev`** ⇒ 想换一个 `ev`，
# 不需要回到原始线性、更不需要重新解码 —— 把当前输出**反解回"过肩之前"**、乘上
# `2^(γ·Δev)`、再正向过一次即可。**全程一条 1D 曲线、不分区域、不用掩膜。**

def _shoulder(t, knee, ceil):
    """入口的软肩（`entry_tone` 里那一段；**不含趾部**）。"""
    d = max(float(ceil) - float(knee), 1e-6)
    return np.where(t > knee, knee + d * (1.0 - np.exp(-(t - knee) / d)), t)


def _shoulder_inv(y, knee, ceil):
    """上面那条肩的**逆**（单调 ⇒ 可逆）。"""
    d = max(float(ceil) - float(knee), 1e-6)
    r = np.clip(1.0 - (np.asarray(y, np.float64) - knee) / d, 1e-9, None)
    return np.where(y > knee, knee - d * np.log(r), y)


def refocus(lin, d_ev, cfg=C):
    r"""把入口那条曲线**整体重打 `d_ev` 档** —— 等价于"一开始就用 `ev + d_ev` 过入口"。

    只动亮度、三通道乘同一个倍率（与 `entry_tone` 同契约）。
    ⚠ 近似：反解时**忽略趾部**（趾部只作用于 `y < 0.84×中位` 的暗部；而锚点是按脸算的，
    脸 / 背景 / 高光都在趾部之上）⇒ 暗部会有一点偏差，实测要报出来。
    """
    if abs(float(d_ev)) < 1e-6:
        return lin
    g = float(getattr(cfg, 'ENTRY_GAMMA', 1.0))
    knee = float(getattr(cfg, 'ENTRY_KNEE', 1.0))
    ceil = float(getattr(cfg, 'ENTRY_CEIL', 1.0))
    Y = np.maximum(color.luma(np.clip(lin, 0.0, None)), 1e-9)
    # 反解回"过肩之前" ⇒ 换 ev 等价于 t × 2^(γ·Δev) ⇒ 再正向过一次（肩部）
    t = _shoulder_inv(Y, knee, ceil) * (2.0 ** (g * float(d_ev)))
    t2 = _shoulder(np.maximum(t, 1e-9), knee, ceil)
    return lin * (t2 / Y)[..., np.newaxis]


def anchor_ev(disp, cfg=C, masks=None):
    r"""由**脸**算出"位置"要补多少档 —— 让脸中位落到 `ANCHOR_FACE_L`。

    闭式解（不拟合、不迭代）：入口曲线肩部以下 `显示线性 ∝ u^γ`（u = Y·2^ev），
    而 `L*+16 ∝ Y^(1/3)` ⇒ `u ∝ (L*+16)^(3/γ)` ⇒

        d_ev = (3/γ) · log2( (L_靶 + 16) / (L_脸 + 16) )

    返回 `(d_ev, info)`。**拿不到脸 ⇒ 返回 0（逐位不变）。**
    """
    info = dict(applied=False)
    if not bool(getattr(cfg, 'ANCHOR_ENABLE', True)):
        info['reason'] = 'off'
        return 0.0, info
    if masks is None:
        try:
            from . import face
            masks = face.parse(np.clip(disp, 0.0, 1.0))['masks']
        except Exception as e:                              # noqa: BLE001
            info.update(reason='no_mask', err='%s: %s' % (type(e).__name__, e))
            return 0.0, info
    sel = np.asarray(masks.get('face_skin', 0.0)) > 0.5
    n = int(sel.sum())
    if n < int(getattr(cfg, 'ANCHOR_MIN_FACE_PX', 300)):
        info.update(reason='no_face', n=n)
        return 0.0, info
    lin = color.s2l(np.clip(disp, 0.0, 1.0))
    L = color.L_of_lin(color.Y_of(lin))
    Lf = float(np.median(L[sel]))
    tgt = float(getattr(cfg, 'ANCHOR_FACE_L', 68.0))
    g = max(float(getattr(cfg, 'ENTRY_GAMMA', 1.0)), 1e-6)
    raw = (3.0 / g) * float(np.log2(max(tgt + 16.0, 1e-6) / max(Lf + 16.0, 1e-6)))
    # ★ 只提不压（09-14 SV 选「甲」）：脸已经够亮 ⇒ 一个像素都不动。
    if bool(getattr(cfg, 'ANCHOR_ONLY_UP', True)) and raw <= 0.0:
        info.update(reason='already_bright', face_L_before=Lf, face_L_target=tgt, n_face=n)
        return 0.0, info
    cap = float(getattr(cfg, 'ANCHOR_EV_MAX', 2.0))
    d_ev = float(np.clip(raw, -cap, cap))
    info.update(applied=True, face_L_before=Lf, face_L_target=tgt,
                d_ev=d_ev, capped=bool(abs(raw) > cap), n_face=n)
    return d_ev, info


def finish_anchor(disp, cfg=C, masks=None):
    r"""锚点**收尾**：把**当前**画面里的脸挪到 `ANCHOR_FACE_L`（线性域乘一个**全局**增益）。

    为什么需要：`anchor_ev` 那一步是在 **L1** 把脸放到靶上，但后面的 **L2 影调曲线会再把它抬上去**
    （实测 +8.4 L\*）。这一步在 L2 之后量一次脸、把它挪回靶 ⇒ **最终脸真的落在靶上**。

    仍然是"一条曲线"：**整张乘同一个增益**，不分区、不用掩膜决定力道（掩膜只用来**量**脸）。
    """
    info = dict(applied=False)
    if masks is None:
        try:
            from . import face
            masks = face.parse(np.clip(disp, 0.0, 1.0))['masks']
        except Exception as e:                              # noqa: BLE001
            info.update(reason='no_mask', err='%s: %s' % (type(e).__name__, e))
            return disp, info
    sel = np.asarray(masks.get('face_skin', 0.0)) > 0.5
    n = int(sel.sum())
    if n < int(getattr(cfg, 'ANCHOR_MIN_FACE_PX', 300)):
        info.update(reason='no_face', n=n)
        return disp, info
    lin = color.s2l(np.clip(disp, 0.0, 1.0))
    Y = color.Y_of(lin)
    L = color.L_of_lin(Y)
    Lf = float(np.median(L[sel]))
    tgt = float(getattr(cfg, 'ANCHOR_FACE_L', 68.0))
    tol = float(getattr(cfg, 'ANCHOR_FINISH_TOL_L', 0.6))
    if abs(Lf - tgt) < tol:
        info.update(reason='on_target', face_L=Lf)
        return disp, info
    k = float(color.lin_of_L(tgt)) / max(float(np.median(Y[sel])), 1e-9)
    out = np.clip(color.l2s(np.clip(lin * k, 0.0, None)), 0.0, 1.0)
    info.update(applied=True, face_L_before=Lf, face_L_target=tgt, gain=k,
                n_face=n, clip_frac=float(np.mean(color.luma(np.clip(lin * k, 0, None)) > 1.0)))
    return out, info


def clip_guard(lin, cfg=C):
    """入口高光护栏：给入口增益设一个**只往下**的上限（按"允许裁切的像素比例"）。

    为什么必须在**入口**做：高光的梯度只要过了下游的 `np.clip(lin, 0, 1)` 就没了，
    L4 的 `guard.CAP_WHITE_FRAC` 只能把一块平板压暗，**救不回形状**（见 SKILL.md §19.2）。

    依据（三方一致，见 SKILL.md §19.3）：
      * Adobe DNG 规范：`BaselineExposure` = "高光还能往回捞多少 EV 而不真裁切" —— 是**余量**，
        **不是"该加的增益"**；
      * RawTherapee：EV=0 = **增益刚好让最亮的通道不裁切**，Auto 用 **Clip%**（默认 0.2%）定白点；
      * darktable filmic："把中间调调对，**高光别管**"。

    返回 (lin_out, k)。**k ≤ 1，不裁切的图 k=1 ⇒ 逐位不变。** 判据用线性亮度
    （`color.luma`，与 `apply_entry_curve` 同一个量），"裁切"阈值 = 显示域 `ENTRY_CLIP_LEVEL` 的线性值。
    """
    if not getattr(cfg, 'ENTRY_CLIP_GUARD', False):
        return lin, 1.0
    allow = float(getattr(cfg, 'ENTRY_CLIP_ALLOW', 0.0))
    lvl = float(color.s2l(float(getattr(cfg, 'ENTRY_CLIP_LEVEL', 254.0 / 255.0))))
    y = color.luma(np.clip(lin, 0.0, None))
    if float(np.mean(y >= lvl)) <= allow:
        return lin, 1.0
    # ★ P2-9：下界是**保险丝**（config.ENTRY_CLIP_GUARD_LO，默认 0.60 ⇒ 最多压 −0.74EV），
    #   原来写死 0.02（≈ −5.6EV）—— 遇到大面积真过曝会把正常曝光的部分也一起拖黑。
    lo = float(getattr(cfg, 'ENTRY_CLIP_GUARD_LO', 0.60))
    hi = 1.0                              # 裁切比例随 k 单调不减 ⇒ 二分
    for _ in range(int(getattr(cfg, 'CLIP_GUARD_ITERS', 28))):
        mid = 0.5 * (lo + hi)
        if float(np.mean(y * mid >= lvl)) <= allow:
            lo = mid
        else:
            hi = mid
    return lin * lo, float(lo)


def load_raw(path, max_side=C.MAX_SIDE):
    """RAW 解码 + IDT（白平衡 / 色彩矩阵由 rawpy 完成；**基线曝光**按机型表 + DR tag 补回）。"""
    import rawpy
    from . import cameras

    kw = dict(C.RAW_DECODE)
    exif = None
    make = model = None
    thumb = None
    with rawpy.imread(path) as raw:      # 一次打开：解码 + 缩略图（拿 EXIF 和机型）
        try:
            th = raw.extract_thumb()
            if th.format == rawpy.ThumbFormat.JPEG:
                thumb = th.data
                tmp = ImageOps.exif_transpose(Image.open(_stdlib_io.BytesIO(th.data)))
                ex = tmp.getexif()
                make = str(ex.get(271) or '') or None
                model = str(ex.get(272) or '') or None
                exif = _exif_orientation_fixed(tmp)
        except Exception:
            pass
        rgb = raw.postprocess(use_camera_wb=kw['use_camera_wb'],
                              no_auto_bright=kw['no_auto_bright'],
                              output_bps=kw['output_bps'],
                              gamma=kw['gamma'],
                              half_size=kw['half_size'],
                              output_color=rawpy.ColorSpace.sRGB)

    rgb = _resize(rgb, max_side)           # 先在 uint16 上缩，省一次 40MP 的浮点转换
    lin = rgb.astype(np.float64) / 65535.0
    lin = np.clip(lin, 0.0, None)          # 只挡负值，不夹上限（高光余量要留着给 L1）

    # ---- 基线曝光（baseline exposure）：相机故意欠曝那几档，在这里一次补回来 ----
    # 相机厂商把"把中点抬到 18%"的活留给转换器做（DNG 里叫 baseline exposure），
    # Adobe 系静默做掉。富士还要按 DR 档额外欠曝 1~2 档，档位写在 RAF 的 MakerNote 里。
    cam = dict(cameras.lookup(make, model), make=make, model=model)
    # ---- 元数据（**只读**，不受开关影响）：开关管的是"补不补"，不是"读不读" ----
    dr = None
    raw_ev = None
    if thumb is not None:
        try:
            mn = rawmeta.thumb_makernote(thumb)
            dr = rawmeta.fuji_development_dr(mn)
            raw_ev = rawmeta.fuji_raw_ev(mn)
        except Exception:
            dr, raw_ev = None, None
    cam['fuji_dr'] = dr

    # ---- 入口补偿（受开关管）：零点 + 曲线形状 ----
    bias = 0.0
    curve = None
    # ⚠ `ENTRY_BIAS_ENABLE` 必须管住**整件事**（机型基底 + DR 额外量 + 曲线）。
    #   之前把"×2^baseline_ev"写在开关外面 ⇒ 关掉开关照样补 0.72 档，
    #   A/B 实验的"未补偿"组其实已经带了补偿，结论会被带偏。
    if C.ENTRY_BIAS_ENABLE:
        bias = float(cam['baseline_ev'])
        # ★ 优先：实测相机曲线（逐 机型×DR 一组；增益锚点 = 线性亮度 → 增益倍数）
        #   ⚠ P1-6：默认路径（ENTRY_TONE=True）**只取 curve[0]（零点）**，
        #     `anchors`（形状）是死数据 —— 别以为它还在参与运算（见 cameras.py 文件头）。
        curve = cameras.entry_curve(model, dr)
        if curve is not None:
            bias = float(cameras.entry_zero_ev(model, dr))   # 零点 = 18% 灰处的实测增益（一个 EV 数）
            cam['bias_source'] = '实测相机曲线 DR%s' % dr
        elif raw_ev is not None:
            # 机身直接写了精确 EV（已含基础偏移）—— 次优，只有零点没有形状
            bias = float(raw_ev)
            cam['bias_source'] = 'tag 0x9650'
        else:
            dr_ev = cameras.dr_bias_ev(dr)
            if dr_ev is not None:
                bias += float(dr_ev)
                cam['dr_bias_ev'] = float(dr_ev)
                cam['bias_source'] = 'DR%s 查表' % dr
            else:
                cam['bias_source'] = '仅机型基底（无 DR tag）'
    cam['idt_bias_ev'] = bias              # 下游据此判断"入口补过了没有"
    if C.ENTRY_BIAS_ENABLE and getattr(C, 'ENTRY_TONE', False):
        # ★ 入口成形（09-13 SV 拍板「乙」）：零点(一个 EV) + 固定的亮度/反差/肩部曲线。
        #   不再复现机内 JPEG ⇒ 不会把"相机的机内风格 + 测光偏亮"一起搬进来（见 config.py）。
        #   对"没量过曲线的机身"也是同一条路（bias = 机型基底 + DR 查表）⇒ 全机型口径统一。
        # ★ 位置逐张听相机（09-13 深夜）：落点 `a` 改由「机型×DR 实测落点规律」**逐张**定。
        #   拿不到（机型没标定 / DR 不认 / e 算不出）⇒ 回落 ENTRY_LEVEL，**逐位同旧行为**。
        a_settle = None
        if getattr(C, 'ENTRY_SETTLE_ENABLE', False):
            _e = scene_exposure_index(lin)
            a_settle = cameras.entry_settle_level(model, dr, _e)
            if a_settle is not None:
                # 「往大师那头靠多少」= 全局偏移（档），默认 0 ⇒ 纯"位置听相机的"
                _shift = float(getattr(C, 'ENTRY_SETTLE_SHIFT_EV', 0.0))
                if abs(_shift) > 1e-9:
                    a_settle *= 2.0 ** _shift
                cam['entry_settle'] = 'e=%+.2f DR%s → 落点 ×%.3f' % (_e, dr, a_settle)
                cam['entry_settle_e'] = _e
                cam['entry_settle_shift_ev'] = _shift
        lin = entry_tone(lin, bias, C, level=a_settle)
        # 报告用：此时曲线只贡献**零点**（形状已由固定成形负责），措辞别让人以为还在复现相机。
        cam['bias_source'] = cam.get('bias_source', '').replace('实测相机曲线', '实测零点')
        cam['entry_shape'] = '胶片成形 γ%.3f %s %s肩%.2f→%.3f %s' % (
            C.ENTRY_GAMMA,
            ('落点×%.3f(听相机)' % a_settle) if a_settle is not None
            else ('落点%.2f(全局)' % C.ENTRY_LEVEL),
            str(getattr(C, 'ENTRY_SHOULDER_KIND', 'exp')),
            C.ENTRY_KNEE, C.ENTRY_CEIL,
            ('趾×%.2f@中位×%.2f~%.2f' % (C.ENTRY_TOE, C.ENTRY_TOE_LO_REL,
                                        C.ENTRY_TOE_HI_REL))
            if float(getattr(C, 'ENTRY_TOE', 1.0)) < 1.0 - 1e-9 else '无趾部')
    elif curve is not None:
        # 实测曲线 = 一条"亮度 → 增益"的曲线：三通道乘同一个增益 ⇒ 只动亮度、不改色相
        # （契约：曲线只在亮度域做）。按输入亮度查，暗部/中间调/高光各自有自己的倍数。
        lin = apply_entry_curve(lin, curve)
        cam['entry_curve'] = '增益锚点 %d 个' % len(curve[1])
    else:
        gain = float(2.0 ** bias)
        if gain != 1.0:
            lin = lin * gain

    # ---- ★ 入口高光护栏（09-13 SV 拍板）：「按高光不裁切定零点」 ----
    # 放在这里 = **在 `np.clip(lin, 0, 1)` 之前**，所以高光的梯度还救得回来。
    # k ≤ 1 只往下压；不裁切的图 k=1 ⇒ 逐位不变（X-T30 III 三张实测 k≈1.0）。
    if C.ENTRY_CLIP_GUARD:
        lin, _gk = clip_guard(lin, C)
        cam['entry_clip_guard_k'] = _gk
        if _gk < 0.999:
            cam['bias_source'] = '%s +高光护栏 ×%.3f' % (cam.get('bias_source', ''), _gk)

    lin, wb_info = idt_wb(lin)
    disp = np.clip(color.l2s(lin), 0.0, 1.0)
    cam = dict(cam, wb=wb_info)
    return Sample(lin, disp, 'raw', path, exif, cam)


def load(path, max_side=C.MAX_SIDE, src=None):
    """src: None/'auto' | 'raw' | 'jpg'（jpg = 走解码器出来的显示域，不是机内 JPEG）"""
    k = kind_of(path)
    if src in (None, 'auto'):
        src = 'raw' if k == 'raw' else 'jpg'
    if src == 'raw':
        if k != 'raw':
            raise ValueError('要求 RAW，但 %s 不是 RAW' % path)
        return load_raw(path, max_side)
    return load_std(path, max_side)


def find_pair(path, prefer='raw'):
    """给一个 stem 或任一路径，找同名的 RAW/JPG 对。返回实际存在的路径。"""
    stem, ext = os.path.splitext(path)
    ext = ext.lower()
    if prefer == 'raw' and ext in RAW_EXT:
        return path
    if prefer == 'jpg' and ext in STD_EXT:
        return path
    order = (RAW_EXT, STD_EXT) if prefer == 'raw' else (STD_EXT, RAW_EXT)
    for exts in order:
        for e in sorted(exts):
            cand = stem + e
            if os.path.exists(cand):
                return cand
    return path


def save(disp, path, exif=None, quality=C.JPEG_QUALITY):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
    im = Image.fromarray(color.display_to_u8(disp))
    kw = dict(quality=int(quality), subsampling=0, optimize=True)
    if exif:
        kw['exif'] = exif
    im.save(path, **kw)
    return path
