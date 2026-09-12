# -*- coding: utf-8 -*-
"""svFilm —— 唯一调参入口。

约定：其它文件里不允许出现魔法数字，全部从这里取。
改动前先 git commit。
"""

VERSION = '0.1.0'

# ========== 输入 ==========
MAX_SIDE = 2048                 # 工作分辨率（长边）
RAW_DECODE = dict(              # rawpy 解码参数（固定，尽量保证可复现）
    use_camera_wb=True,
    no_auto_bright=True,
    output_bps=16,
    gamma=(1, 1),               # 线性输出：0~1 场景线性
    half_size=False,
    demosaic_algorithm=None,    # None = rawpy 默认
)
CACHE_LINEAR = False            # True: 线性底缓存成 16bit PNG 复用（RAW 非确定性时）

# ========== L0 分析（判据：只用分位，绝不用均值） ==========
PCT_BLACK = 0.2                 # 黑点分位
PCT_MID = 50.0                  # 中灰分位
PCT_KNEE = 98.0                 # 高光膝点分位（肩部起点）
PCT_WHITE = 99.8                # 白点分位

TGT_BLACK = 0.010               # 绝对靶（显示域 0~1）= 灰度 2.5
TGT_MID = 0.550                 # 中灰靶：0.55 ≈ 灰度 140 ≈ L*58（大师带中位）
TGT_WHITE = 0.863               # 白点靶 ≈ 灰度 220（220 是本批素材实测最优；250 会把颜色放大 3.4 倍）
BLACK_PULL = 1.00               # 黑点最多下压的档数（防"画面本来没有暗部"被压死）

MID_DEADZONE = 0.030            # 中灰与靶差小于此值 => 判为"本来就对"，不动
EV_CAP_DOWN = 1.8               # 最多压档数
EV_CAP_UP = 4.00                # 最多提档数（只对允许提亮的源生效）
                                # ⚠ 调大 = 把欠曝的 RAW 一律拉回中灰靶（夜景会被拉亮）；
                                #   调小 = 欠得多的图永远到不了靶。这条要 SV 拍板。

# 允不允许"提亮"。这是 RAW 与 JPG 最本质的分界：
#   RAW 线性有高光余量，提亮是"重新冲洗"；JPG 是相机曲线压过的成品，
#   提亮等于把被压缩的颜色按斜率放大（粉裙变鲜红就是这么来的）。
ALLOW_LIFT_RAW = True
ALLOW_LIFT_JPG = False
ALLOW_LIFT = False              # 兼容用旧开关，别再动它

NOISE_FLOOR = 1.0e-6            # log2 域保护

# ========== L1 修正（绝对靶 + 亮度域） ==========
WB_ENABLE = True
WB_STRENGTH = 1.0               # 0~1
WB_MAX_GAIN = 0.25              # 单通道增益限幅 ±
WB_NEUTRAL_SAT = 0.22           # 判"近中性像素"的饱和度上限（HSV S）
WB_MIN_COVER = 0.02             # 近中性像素占比低于此 => 不做 WB

SHADOW_LIFT = 0.18              # 阴影段上提比例（0 = 不动）
CONTRAST = 1.00                 # 以中灰为支点的对比（log2 域，1.0 = 不动）
CONTRAST_MAX = 1.35
HILIGHT_DESAT = 0.35            # 高光去饱和（胶片的真实行为，同时防通道削顶）

# ========== L2 风格 ==========
LUT_PATH = None                 # 指定 .cube 就用它；None = 用下面内置参数化胶片色
LUT_DOMAIN = 'display'          # 'display' = 期望输入为 0~1 显示域（电影 LUT 常见）
LUT_STRENGTH = 1.0              # 0~1 混合
LOCK_MID = True                 # LUT 之后把中灰拉回靶，并记录漂移

FILM_MATRIX = [                 # 内置占位：线性域 3x3（负片交调，很轻）
    [1.030, -0.022, -0.008],
    [-0.014, 1.020, -0.006],
    [-0.006, -0.012, 1.018],
]
FILM_SHADOW_TINT = [0.000, 0.000, 0.018]   # 暗部微冷
FILM_HILIGHT_TINT = [0.010, 0.003, -0.007]  # 亮部微暖
TINT_LO = 0.18                  # 加性 tint 的作用范围（低于此视为暗部）
TINT_HI = 0.75                  # 高于此视为亮部
CHROMA_SCALE = 1.00
CONTRAST = 1.00                 # 明度对比（只动 L*，1.0 = 不动；>1 是 S 形，中灰不移动）
CONTRAST_MAX = 1.35             # 上限（超过 0.12 的 S 幅度会失去单调性）

# ========== L3 局部 ==========
SKIN_PROTECT = True
SKIN_PROTECT_STRENGTH = 0.55    # 肤色区彩度少降的比例

# ========== L4 护栏（只做"不许超过"，不做"必须等于"） ==========
CAP_WHITE_FRAC = 0.030          # 死白（>=254）占比上限
CAP_CHROMA_C90 = 45.0           # 彩度 P90 上限（Lab C）
GUARD_MAX_PASS = 3

# ========== 输出 ==========
JPEG_QUALITY = 95
