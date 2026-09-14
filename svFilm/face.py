# -*- coding: utf-8 -*-
r"""脸 / 人 —— 进 L3 之前**算一次**，位置（脸多亮）和形状（脸多立体）两层共用。

为什么需要这一层（而不是继续用 `local.skin_mask`）：
  `local.skin_mask` 是**颜色**软掩膜（色相 + 彩度 + 亮度三重门），它回答的是
  "这个像素像不像皮肤的颜色" —— 粉衣服、木色、粉墙都会中招，而且它**不知道脸在哪**。
  而"这张脸该多亮""这张脸有多立体"要的是**人的位置**和**五官的位置**，
  只能靠分割 + 五官关键点拿。两者不是一回事，别混。

★ 算一次、共用：分割是这条路上最贵的一步（256×256 前向 + 上采样），
  位置层和形状层各跑一次等于白付两遍。所以这里做成"一次解析、一个 dict"。

★★ 挑脸的闸（不加会出事）：`cv2.FaceDetectorYN` 会**在手上/花束上编出假脸**，
  而且假脸的眼距可能比真脸还大（实测 DSCF2328 机内：3 个候选，假脸 eyed 233 最大，
  但框只有 0.44 落在人物掩膜里）。⇒ **光"取最大眼距"必错**，要按下面的闸筛：
    ① 框 ∩ 人物掩膜 ≥ FACE_GATE_OVP（框必须长在人身上）
    ② 眼距 ≥ FACE_GATE_EYED × 图宽（脸太小量不准）
    ③ 框高 ≥ FACE_GATE_H × 人的外接框高（假框通常很小）

⚠ 模型路径不能带中文（mediapipe 的硬限制），本工程路径里含中文 ⇒ 走 `_ascii()` 复制到临时目录。
"""
from __future__ import annotations

import os
import shutil
import tempfile

import numpy as np

from . import config as C

HERE = os.path.dirname(os.path.abspath(__file__))
MDIR = os.path.join(HERE, '_models')
YUNET = os.path.join(MDIR, 'yunet.onnx')
SELFIE = os.path.join(MDIR, 'selfie_multiclass_256x256.tflite')
SEG_SIDE = 256                      # 分割模型的输入边长（固定）

_SEG = None                         # (segmenter, mp)　模块级缓存
_DET = {}                           # 尺寸 -> YuNet detector


class FaceUnavailable(RuntimeError):
    """依赖/模型缺失。调用方应当**优雅降级**（当作"没人脸"），不是崩。"""


def _ascii(path):
    """mediapipe 只吃 ASCII 路径；本工程路径含中文 ⇒ 复制到临时目录再用。"""
    try:
        path.encode('ascii')
        return path
    except UnicodeEncodeError:
        pass
    tp = os.path.join(tempfile.gettempdir(), os.path.basename(path))
    if not os.path.exists(tp) or os.path.getsize(tp) != os.path.getsize(path):
        shutil.copy(path, tp)
    return tp


def _segmenter():
    global _SEG
    if _SEG is None:
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mpp
            from mediapipe.tasks.python import vision
        except Exception as e:                                  # noqa: BLE001
            raise FaceUnavailable('mediapipe 不可用：%s' % e)
        for p in (YUNET, SELFIE):
            if not os.path.exists(p):
                raise FaceUnavailable('缺模型文件：%s' % p)
        opt = vision.ImageSegmenterOptions(
            base_options=mpp.BaseOptions(model_asset_path=_ascii(SELFIE)),
            output_confidence_masks=True, output_category_mask=False)
        _SEG = (vision.ImageSegmenter.create_from_options(opt), mp)
    return _SEG


def _detector(w, h):
    key = (int(w), int(h))
    if key not in _DET:
        import cv2
        d = cv2.FaceDetectorYN.create(_ascii(YUNET), '', (64, 64),
                                      float(getattr(C, 'FACE_DET_SCORE', 0.55)),
                                      float(getattr(C, 'FACE_DET_NMS', 0.30)))
        d.setInputSize(key)
        _DET[key] = d
    return _DET[key]


def masks(disp):
    """`disp` = 显示域 (H,W,3) 0~1 的 **RGB**。返回同尺寸的软掩膜 dict：
       bg / person / skin(=脸皮肤+身体皮肤) / face_skin(=只有脸皮肤) / hair(=头发)。
    ⚠ 语义保证：**person == 1 − bg**（分割是 6 类互斥）。这条被权重层用到了，别改。

    ★ 09-14 补两个键：原来只暴露 bg/person/skin，而 **`skin` 是"脸皮肤 + 身体皮肤"合起来的**
    （`cm[3] + cm[2]`）⇒ 拿它当"脸"必然把**手臂/手**一起圈进来（DSCF1954 / DSCF0791 实测）。
    分割本来就是分开的两类，只是没暴露。现在把 **`face_skin`（cm[3]）** 和
    **`hair`（cm[1]，用来定"头在哪"）** 单独拿出来 —— `face_region()` 靠它们工作。
    """
    import cv2
    seg, mp = _segmenter()
    u8 = np.ascontiguousarray((np.clip(disp, 0.0, 1.0) * 255).astype(np.uint8))
    H, W = u8.shape[:2]
    small = np.ascontiguousarray(cv2.resize(u8, (SEG_SIDE, SEG_SIDE),
                                            interpolation=cv2.INTER_AREA))
    r = seg.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=small))
    # 6 类顺序：0=背景 1=头发 2=身体皮肤 3=脸皮肤 4=衣服 5=其他
    cm = [np.asarray(r.confidence_masks[i].numpy_view(), np.float32) for i in range(6)]
    up = lambda a: np.clip(cv2.resize(a, (W, H), interpolation=cv2.INTER_LINEAR), 0, 1)  # noqa: E731
    return dict(bg=up(cm[0]), person=up(1.0 - cm[0]),
                skin=up(cm[3] + cm[2]),        # 原语义（脸+身），别动
                face_skin=up(cm[3]),           # ★ 单独的脸皮肤
                hair=up(cm[1]))                # ★ 头发（定"头在哪"的锚）


def face_region(m, cfg=C):
    r"""★ 「脸在哪」—— **不靠正脸检测框**。返回 `(bool 掩膜 或 None, 说明)`。

    为什么需要它（09-13 深夜 SV 逼出来的）：**正脸检测器对"正侧脸 / 背影 / 小脸"基本给不出框**
      · DSCF1954 正侧脸：机内 JPEG **0 个候选**，成片只给一个 26×32 的垃圾框 ⇒ 整张被跳过；
      · DSCF0999 小脸：被 `FACE_GATE_EYED`（眼距 ≥3% 图宽）挡掉 —— 而它框里**明明有 2025 个皮肤
        像素**，"脸太小量不准"这个理由**根本不成立**；
      · DSCF1782：眼距 2.7% 图宽，差一点点。
    ⇒ 而"脸在哪"**其实是知道的**：分割的 `face_skin`（单独那一类脸皮肤）**对侧脸也认**。

    规则：`face_skin > 0.5 且 person > 0.3` → 连通域 → 只留**落在"头窗口"里的块**。
      头窗口 = `[头发外接框顶, 头发顶 + FACE_HEAD_K × 头发宽]`
      （**头发一定长在头上**，侧脸/背影都有 ⇒ 它是比正脸检测器稳得多的锚）。
      头发太少（< `FACE_MIN_HAIR_PX`）就退化成"人物外接框的上部 `FACE_HEAD_FALLBACK_FRAC`"。
    ⇒ 手臂/手即便被误判成脸皮肤，也会因为**落在头窗口之外**被丢掉（这是 v1 用"皮肤总类"时栽的坑：
      DSCF1954 的脸和手臂皮肤连成同一块 97275 px，靠"质心在上半身"根本筛不掉）。

    ⚠ 拿不到就返回 `None` —— 调用方**必须**回落老的"正脸框"路径，绝不比改动前更差。
    """
    import cv2
    fs = (m['face_skin'] > 0.5) & (m['person'] > 0.3)
    H, W = fs.shape
    if int(fs.sum()) < int(getattr(cfg, 'FACE_REGION_MIN_PX', 300)):
        return None, 'no_face_skin'
    pm = m['person'] > 0.5
    if int(pm.sum()) == 0:
        return None, 'no_person'
    ys, _xs = np.where(pm)
    y0p, y1p = float(ys.min()), float(ys.max())
    ph = max(y1p - y0p, 1.0)

    hair = m['hair'] > 0.5
    if int(hair.sum()) >= int(getattr(cfg, 'FACE_MIN_HAIR_PX', 200)):
        hy, hx = np.where(hair)
        hy0, hx0, hx1 = float(hy.min()), float(hx.min()), float(hx.max())
        hw = float(hx1 - hx0 + 1)
        y_hi = hy0
        y_lo = min(hy0 + float(getattr(cfg, 'FACE_HEAD_K', 1.35)) * hw,
                   y0p + float(getattr(cfg, 'FACE_HEAD_BOT_FRAC', 0.65)) * ph)
        x_lo, x_hi = hx0 - 0.5 * hw, hx1 + 0.5 * hw
        how = 'hair_window'
    else:
        y_hi = y0p
        y_lo = y0p + float(getattr(cfg, 'FACE_HEAD_FALLBACK_FRAC', 0.45)) * ph
        x_lo, x_hi = 0.0, float(W)
        how = 'top_window'

    wm = np.zeros_like(fs)
    wm[int(max(y_hi, 0)):int(min(y_lo, H)), int(max(x_lo, 0)):int(min(x_hi, W))] = True
    n, lab = cv2.connectedComponents((fs & wm).astype(np.uint8), connectivity=8)
    keep = np.zeros_like(fs)
    for i in range(1, n):
        blk = lab == i
        if int(blk.sum()) < int(getattr(cfg, 'FACE_REGION_MIN_PX', 300)):
            continue
        keep |= blk
    if int(keep.sum()) < int(getattr(cfg, 'FACE_REGION_MIN_PX', 300)):
        return None, 'window_empty'
    return keep, how


def landmarks(disp, person):
    """挑出**那一个人**的脸：返回 (lm, box, eyed, ovp)。`lm` = 5 个关键点 (re, le, no, rml, lml)。
    过不了闸就返回 (None, None, 0, 0) —— **宁可不做，也不要在假脸上动手**。"""
    import cv2
    u8 = np.ascontiguousarray((np.clip(disp, 0.0, 1.0) * 255).astype(np.uint8))
    H, W = u8.shape[:2]
    bgr = u8[:, :, ::-1]
    pm = person > 0.5
    ph = 0.0
    if int(pm.sum()) > int(getattr(C, 'FACE_MIN_PERSON_PX', 500)):
        ys, _xs = np.where(pm)
        ph = float(ys.max() - ys.min())
    try:
        _n, fs = _detector(W, H).detect(np.ascontiguousarray(bgr))
    except Exception:                                           # noqa: BLE001
        fs = None
    if fs is None or not len(fs):
        return None, None, 0.0, 0.0
    best = None
    for f in fs:
        lm = [np.array([float(f[4 + 2 * i]), float(f[5 + 2 * i])]) for i in range(5)]
        bb, eyed = lm_box(lm)
        if bb is None:
            continue
        x, y, w, h = [int(round(v)) for v in bb]
        x0, y0 = max(x, 0), max(y, 0)
        x1, y1 = min(x + w, W), min(y + h, H)
        if x1 - x0 < 16 or y1 - y0 < 16:
            continue
        m = np.zeros((H, W), bool)
        m[y0:y1, x0:x1] = True
        ovp = float((m & pm).sum()) / max(int(m.sum()), 1)
        if ovp < float(getattr(C, 'FACE_GATE_OVP', 0.85)):
            continue
        if eyed < float(getattr(C, 'FACE_GATE_EYED', 0.03)) * W:
            continue
        if ph > 0 and (y1 - y0) < float(getattr(C, 'FACE_GATE_H', 0.08)) * ph:
            continue
        if best is None or eyed > best[2]:
            best = (lm, bb, float(eyed), ovp)
    if best is None:
        return None, None, 0.0, 0.0
    return best


def lm_box(lm, wid=None, top=None, bot=None):
    """★ 用五官自己推框（以眼距为尺度）—— 比 YuNet 那个又大又偏的框稳。
    返回 ([x, y, w, h], 眼距)。"""
    re, le, no = np.asarray(lm[0], np.float64), np.asarray(lm[1], np.float64), np.asarray(lm[2], np.float64)
    eyed = float(np.linalg.norm(le - re))
    if not np.isfinite(eyed) or eyed < 1.0:
        return None, 0.0
    wid = float(getattr(C, 'FACE_BOX_WID', 2.40)) if wid is None else wid
    top = float(getattr(C, 'FACE_BOX_TOP', 1.40)) if top is None else top
    bot = float(getattr(C, 'FACE_BOX_BOT', 1.55)) if bot is None else bot
    cx, cy = 0.5 * (re[0] + le[0]), float(no[1])
    return [cx - wid / 2.0 * eyed, cy - top * eyed, wid * eyed, (top + bot) * eyed], eyed


def parse(disp):
    """**一次解析**：返回 L3 要的全部材料。没人脸时 `face` 为 None（调用方照着降级）。

    dict(
      masks  = dict(bg, person, skin, face_skin, hair)   # 软掩膜，和 disp 同尺寸
      face   = None 或 dict(lm, box=[x0,y0,x1,y1], eyed, ovp, weight=框羽化×person, mid=cx)
      person_weight = 软权重（**背景处恒为 0**）
    )

    ★ 09-14：`face`（正脸框）只当**回落路径 + 排假脸**用了 —— "脸在哪"优先走 `face_region(masks)`，
    它不依赖正脸检测器，侧脸 / 背影 / 小脸都能拿到。详见 `face_region` 的 docstring。
    """
    import cv2
    m = masks(disp)
    H, W = disp.shape[:2]
    lm, box, eyed, ovp = landmarks(disp, m['person'])
    out = dict(masks=m, face=None, person_weight=None)
    out['person_weight'] = person_weight(m['person'], (H, W))
    if lm is None:
        return out
    x, y, w, h = box
    x0, y0 = max(int(round(x)), 0), max(int(round(y)), 0)
    x1, y1 = min(int(round(x + w)), W), min(int(round(y + h)), H)
    if x1 - x0 < 16 or y1 - y0 < 16:
        return out
    wf = np.zeros((H, W), np.float32)
    wf[y0:y1, x0:x1] = 1.0
    wf = cv2.GaussianBlur(wf, (0, 0), max(3.0, float(getattr(C, 'FACE_FEATHER_REL', 0.05)) * (x1 - x0)))
    out['face'] = dict(lm=lm, box=[x0, y0, x1, y1], eyed=eyed, ovp=ovp,
                       weight=wf * np.clip(m['person'], 0.0, 1.0),
                       box_weight=wf,
                       mid=0.5 * (float(lm[0][0]) + float(lm[1][0])))
    return out


def person_weight(person, shape=None, sigma_rel=None):
    """★ 人物权重：**在背景掩膜里必须恰好 = 0**。

    分割里 `person == 1 − bg` 恒成立 ⇒ 只在人身边界做高斯羽化会让力道**漏到背景一侧**
    （实测背景被顺手带亮 1.7~5.3 个 L*，而"背景不动"是 SV 明确定下的）。
    `min(blur(person), person)` 也堵不住：`bg > 0.6` 的掩膜像素对应 `person < 0.4`，权重还剩 0.4。
    ⇒ 正解 = 再乘一道 `smoothstep(person, lo, hi)`，让 `person ≤ lo`（⟺ `bg ≥ 1−lo`，覆盖整个背景掩膜）处为 0。
    """
    import cv2
    p = np.clip(np.asarray(person, np.float32), 0.0, 1.0)
    if shape is not None:
        p = cv2.resize(p, (int(shape[1]), int(shape[0])), interpolation=cv2.INTER_LINEAR)
    H, W = p.shape
    sig = max(3.0, (0.006 if sigma_rel is None else float(sigma_rel)) * W)
    soft = cv2.GaussianBlur(p, (0, 0), sig)
    lo = float(getattr(C, 'PERSON_W_LO', 0.45))
    hi = float(getattr(C, 'PERSON_W_HI', 0.80))
    t = np.clip((p - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return soft * (t * t * (3.0 - 2.0 * t))
