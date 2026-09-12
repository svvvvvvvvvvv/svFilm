# -*- coding: utf-8 -*-
"""RAW 元数据层：读出"转换器必须替相机补的那点亮度"。

**纯格式层，不认识机型名** —— 机型相关的数值表在 `cameras.py`。
为什么要单独一层：这件事由"相机厂商故意让 RAW 欠曝、把中点抬回来的活留给转换器"
（DNG 规范里叫 baseline exposure）决定；对富士来说，DR200/DR400 会额外欠曝 1/2 档，
并把该补多少写进 MakerNote。我们要做的就是把那个数读出来。

★ 富士这四个 tag 的确切语义（别混）：
    0x1402 DynamicRangeSetting    —— **模式开关**：1 = Manual/Raw，0 = Auto。**不是档位**。
    0x1403 DevelopmentDynamicRange —— 档位 100/200/400，**仅 0x1402 == 1 时有效**
    0x140b AutoDynamicRange        —— 档位 100/200/400，**仅 0x1402 == 0 时有效**
    0x9650 RawExposureBias         —— 精确 EV（4 字节 = 两个 signed short 的比值）。
                                      有就直接用，没有才查表；多数机身**不写**这个 tag。
  出处：darktable 官方 lua 脚本文档 fujifilm_dynamic_range（"even at 100DR there is an EV
  shift"；且明确写了 DevelopmentDynamicRange 只在 Manual/Raw 存在、Auto 时等价数据在
  AutoDynamicRange）。0x9650 因为编码是 stdlib 没有的"比值"类型，exiv2 至今读不了。

⚠ 实测（2026-09-13，X-T30 III / 759 张）：0x1402 全是 1、0x9650 与 0x140b 一律不写；
  **档位是混的：DR100 118 张 / DR200 66 张 / DR400 575 张** ⇒ 必须逐张读，按机型写死必错。

为什么不用 PIL：`Image.getexif()` 拿得到 MakerNote 的原始字节（tag 0x927C），
但**不解析富士自己那套 IFD**。所以这里手工走一遍：
  "FUJIFILM" + 4 字节偏移 → 标准 TIFF IFD（uint16 条数 + 12 字节条目）。
"""
from __future__ import annotations

import io as _stdlib_io
import struct

TAG_DR_SETTING = 0x1402      # DynamicRangeSetting：1=Manual/Raw，0=Auto
TAG_DEV_DR = 0x1403          # DevelopmentDynamicRange（Manual/Raw 模式下的档位）
TAG_AUTO_DR = 0x140b         # AutoDynamicRange（Auto 模式下的档位）
TAG_RAW_EV = 0x9650          # RawExposureBias（精确 EV，可选）

DR_VALID = (100, 200, 400)

_TYPES = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8, 13: 4}


def fuji_entries(mn):
    """把富士 MakerNote 字节解析成 [(tag, type, count, value), ...]。失败返回 []。"""
    if not mn or len(mn) < 12 or mn[:8] != b'FUJIFILM':
        return []
    base = struct.unpack('<I', mn[8:12])[0]
    if base + 2 > len(mn):
        return []
    n = struct.unpack('<H', mn[base:base + 2])[0]
    out = []
    for i in range(n):
        p = base + 2 + i * 12
        if p + 12 > len(mn):
            break
        tag, typ, cnt = struct.unpack('<HHI', mn[p:p + 8])
        size = _TYPES.get(typ, 1) * cnt
        if size <= 4:
            vb = mn[p + 8:p + 8 + size]
        else:
            off = struct.unpack('<I', mn[p + 8:p + 12])[0]
            if off + size > len(mn):
                continue
            vb = mn[off:off + size]
        try:
            if typ in (3, 8):
                val = struct.unpack('<%dh' % cnt, vb.ljust(2 * cnt, b'\0'))
            elif typ in (4, 9):
                val = struct.unpack('<%di' % cnt, vb.ljust(4 * cnt, b'\0'))
            elif typ == 10:      # SRATIONAL
                v = struct.unpack('<%di' % (cnt * 2), vb.ljust(4 * cnt, b'\0'))
                val = tuple((v[i * 2], v[i * 2 + 1]) for i in range(cnt))
            elif typ in (1, 7):
                val = vb
            else:
                continue
        except struct.error:
            continue
        out.append((tag, typ, cnt, val[0] if cnt == 1 else list(val)))
    return out


def _entry_map(mn):
    return dict((t, v) for t, _ty, _cnt, v in fuji_entries(mn))


def fuji_development_dr(mn):
    """取"开发用动态范围"档位（100 / 200 / 400）。读不到返回 None。

    0x1402 是模式开关，两个档位 tag 互斥：
      Manual/Raw(1) -> 0x1403；Auto(0) -> 0x140b。
    模式本身读不到时两个都试（取第一个值域合法的），但**绝不把 0x1402 的值当档位**
    —— 它的值是 0/1，混进来会静默地少补 1~2 档。
    """
    e = _entry_map(mn)
    mode = e.get(TAG_DR_SETTING)
    if mode == 1:
        cands = (e.get(TAG_DEV_DR),)
    elif mode == 0:
        cands = (e.get(TAG_AUTO_DR),)
    else:
        cands = (e.get(TAG_DEV_DR), e.get(TAG_AUTO_DR))
    for c in cands:
        if c in DR_VALID:
            return int(c)
    return None


def fuji_raw_ev(mn):
    """0x9650 RawExposureBias —— 机身若写了，直接给出要补的 EV（正数）。读不到返回 None。

    编码是"两个 signed short 的比值"（darktable 文档原话），所以按 SRATIONAL 读；
    有些资料按 RATIONAL 记，所以两种都试一遍。值域卡在 0.05~4 EV —— 不合法就丢弃，
    宁可退回查表，也不要因为格式猜错而给出离谱的补量。
    """
    e = _entry_map(mn)
    raw = e.get(TAG_RAW_EV)
    if raw is None:
        return None
    num = den = None
    if isinstance(raw, tuple) and len(raw) == 2:
        num, den = raw
    if not den:
        return None
    try:
        v = abs(float(num) / float(den))
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return v if 0.05 <= v <= 4.0 else None


def thumb_makernote(thumb_bytes):
    """从内嵌 JPEG 字节里取 MakerNote 原始字节（tag 0x927C）。"""
    from PIL import Image, ImageOps
    ex = ImageOps.exif_transpose(Image.open(_stdlib_io.BytesIO(thumb_bytes))).getexif()
    ifd = ex.get_ifd(0x8769) or {}
    return ifd.get(0x927C)
