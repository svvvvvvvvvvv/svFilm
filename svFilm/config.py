# -*- coding: utf-8 -*-
"""svFilm —— 唯一调参入口。

约定：其它文件里不允许出现魔法数字，全部从这里取。
改动前先 git commit。
"""

VERSION = '0.2.2'   # 0.2.2 = 加"基准成色"（中性路径对齐大师平均）+ 降噪层 + 批量 --pair/--raw-only

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

# ========== L2 风格（颜色模型 —— 卷往这里填"量出来的数"） ==========
# 这套模型是照"大师那把尺子"设计的：尺子量什么，这里就有什么旋钮。
#   a / b / b_sh / b_hi  ←→  尺子的 a*中位 / b*中位 / 暗部b* / 亮部b*（冷暖分离）
#   chroma_p / chroma_s  ←→  尺子的 彩度中位 + 彩度P90（形状 + 量级）
#   contrast             ←→  尺子的 反差 span90
# 换卷 = 改这几组数，见 stocks.py。
LUT_PATH = None                 # 指定 .cube 就用它，叠加在内置颜色之上
LUT_DOMAIN = 'display'          # 'display' = 期望输入为 0~1 显示域（电影 LUT 常见）
LUT_STRENGTH = 1.0              # 0~1 混合
LOCK_MID = True                 # LUT/风格之后把中灰拉回 L1 交出来的那个中灰，并记录漂移

FILM_MATRIX = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]   # 线性域 3x3（负片交调，默认恒等）
COL_A = 0.0                     # 整体 a* 偏移（+ 偏红/洋红，− 偏绿）
COL_B = 0.0                     # 整体 b* 偏移（+ 偏黄，− 偏蓝）
COL_B_SH = 0.0                  # 暗部额外 b*（负 = 暗部偏冷/青蓝）
COL_B_HI = 0.0                  # 亮部额外 b*（正 = 亮部偏暖）
COL_TINT_LO = 25.0              # b_sh 的作用上界（L*，对齐大师口径 L≤P25）
COL_TINT_HI = 90.0              # b_hi 的作用下界（L*，对齐大师口径 L≥P90）
CHROMA_P = 1.00                 # 彩度 gamma：>1 压中低彩度、保高彩度（更"局部化"）；<1 反之
CHROMA_S = 1.00                 # 彩度整体倍率
CHROMA_REF = 20.0               # 彩度 gamma 的参考点
CONTRAST = 1.00                 # 明度对比（只动 L*，1.0 = 不动；>1 是 S 形，中灰不移动）
CONTRAST_MAX = 1.35             # 上限（超过 0.12 的 S 幅度会失去单调性）

# ========== L2 基准成色（不属于任何卷；把"我方中性路径"对齐到"大师平均"） ==========
# 分工（这是路 B 定的口径，别混）：
#   卷   = 这条作者线**相对大师平均**的性格偏移      → stocks.py
#   基准 = 我方中性路径**相对大师平均**的系统性差    → 这里
#   最终颜色 = cfg 默认 → 基准 → 卷（偏移类相加，彩度/对比类相乘）
# 出处：效果debug/<日期>/卷标定_大师颜色聚类/卷标定报告.md §一 锚点 / §四 系统性差。
# 四条候选（SV 挑一条当默认；BASE_NONE = 现状，向后兼容）：
#   fog      线性光域黑位抬升（"雾"）：黑位 L* 1.2 → 6.6、雾量 0.0009 → 0.0235
#   b        整体 b* 偏移：+3.26（偏黄）→ +1.0
#   chroma   彩度形状/量级：形状 3.61 → ~2.98、中位 5.57 → 7.16（闭式解 p/s）
#   contrast 明度对比（只动 L*）：反差 75 → ~78
BASE = 'BASE_NONE'
BASE_TABLE = {
    'BASE_NONE': dict(
        label='现状（不动）', desc='基线：整体偏黄 +3.3、黑位死黑、几乎无雾',
        b=0.0, chroma_p=1.0, chroma_s=1.0, contrast=1.0, fog=0.0),
    'BASE_FOG': dict(
        label='① 只加雾', desc='黑位从死黑抬到胶片的灰黑、画面通透度略降；色偏与彩度不动',
        b=0.0, chroma_p=1.0, chroma_s=1.0, contrast=1.0, fog=0.0063),
    'BASE_DEYELLOW': dict(
        label='② 退黄+加雾', desc='在①之上把整体黄味从 +3.3 收到 +1 左右（画面明显"没那么黄"）',
        b=-2.26, chroma_p=1.0, chroma_s=1.0, contrast=1.0, fog=0.0063),
    'BASE_FULL': dict(
        label='③ 全对齐', desc='在②之上收彩度形状（高彩不再冲那么猛）+彩度略提+反差略增，最接近大师平均',
        b=-2.26, chroma_p=0.849, chroma_s=1.059, contrast=1.30, fog=0.0063),
}

# ========== 降噪（RAW 提亮后暗部色斑/噪点） ==========
# 位置：L1 修正之后、L2 风格之前（此时噪点被影调放大出来，最该收拾）。
# 原则：只在"暗部 + 平坦区"下手，边缘/细节区不动；用引导滤波（保边）逐通道做，
#       加回的量按掩膜加权 ⇒ 近似零均值 ⇒ 中灰/黑位几乎不漂移（selftest 有这条不变量）。
DENOISE_ENABLE = False
DENOISE_LUMA = 0.45             # 亮度通道降噪强度（0~1，会吃掉一点细节，别给大）
DENOISE_CHROMA = 0.85           # 色度通道降噪强度（色斑主要在这，可以给大）
DENOISE_DARK_LO = 0.00          # 掩膜：L* 低于此 = 全量降噪（暗部噪点最脏）
DENOISE_DARK_HI = 45.0          # 掩膜：L* 高于此 = 不降噪（亮部本来就干净）
DENOISE_EDGE_LO = 4.0           # 掩膜：L* 梯度低于此 = 判为平坦区（全量降噪）
DENOISE_EDGE_HI = 14.0          # 掩膜：梯度高于此 = 判为边缘/纹理（不降噪）
DENOISE_RADIUS = 4              # 引导滤波半径 px（@2048 长边）
DENOISE_EPS = 90.0              # 引导滤波 eps（L* 域，越大越平滑/越糊边）

# ========== 卷（stocks.py 是数据表） ==========
# 选一个卷 = 一次定下"颜色性格 + 空间效果强度"。None = 用上面 L2 的 config 默认（= 轻度风格）。
# 可选：neutral / portra400 / pro400h / fuji_c200 / ektar100 / cinestill800t / air
STOCK = None

# ========== 空间域（颗粒 / 黑柔 Bloom / Halation）==========
# 为什么单独一层：LUT 只能装"颜色 + 影调"，这三个是光学/物理空间效果，塞不进 LUT。
# 位置：L2 风格之后、L3 局部之前。卷会给各自的强度（见 stocks.py），这里是"不选卷时"的默认。
GRAIN_ENABLE = False
GRAIN_AMOUNT = 0.024            # 颗粒强度（乘性，值越大越"糙"）
GRAIN_SIZE = 1.2                # 颗粒尺度（高斯半径 px，@2048 长边）
GRAIN_CHROMA = 0.20             # 彩噪占比（0 = 纯单色颗粒；彩噪多会显脏）
GRAIN_SKIN_SUPPRESS = 0.62      # 肤色区颗粒抑制比例（别把脸磨出麻点）
GRAIN_DETAIL_SUPPRESS = 0.32    # 高细节区颗粒抑制比例（别叠在纹理上糊成一片）
GRAIN_DARK_FLOOR = 0.03         # 比这更暗 => 颗粒淡出（暗部本来就脏，再撒就是噪点）
GRAIN_SEED = 20260912           # 固定种子：同一张图每次结果一致（可复现）

BLOOM_ENABLE = False
BLOOM_AMOUNT = 0.075            # 辉光强度（亮部外溢的光，线性域加回）
BLOOM_RADIUS = 22.0             # 扩散半径 px（@2048 长边，越大越"糊"）
BLOOM_THR_LO = 0.74             # 软阈值下界（显示域灰度，比这低的不发光）
BLOOM_THR_HI = 0.93             # 软阈值上界
BLOOM_WARMTH = 0.30             # 辉光偏暖比例（0 = 中性白）
BLOOM_VEIL = 0.045              # 黑柔特征：整体轻微提灰/降对比（Black Pro Mist 那口气）

HALATION_ENABLE = False
HALATION_AMOUNT = 0.110         # 晕圈强度（线性域加回）
HALATION_RADIUS = 18.0          # 晕圈扩散半径 px
HALATION_THR_LO = 0.78          # 从多亮的高光开始往外散
HALATION_THR_HI = 0.99
HALATION_COLOR = [1.000, 0.300, 0.120]   # 红橙（电影卷片基把红光散射回来）

# ========== L3 局部 ==========
SKIN_PROTECT = True
SKIN_PROTECT_STRENGTH = 0.55    # 肤色区彩度少降的比例

# ========== L4 护栏（只做"不许超过"，不做"必须等于"） ==========
CAP_WHITE_FRAC = 0.030          # 死白（>=254）占比上限
CAP_CHROMA_C90 = 45.0           # 彩度 P90 上限（Lab C）
GUARD_MAX_PASS = 3

# ========== 输出 ==========
JPEG_QUALITY = 95
