# -*- coding: utf-8 -*-
"""svFilm —— 唯一调参入口。

约定：其它文件里不允许出现魔法数字，全部从这里取。
改动前先 git commit。
"""

VERSION = '0.3.1'   # 0.3.1 = L1 过亮护栏改「折中」（阈值从大师分布量，不再拽到中灰）+ 空间/颜色三处调研修正

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

# ---- 入口基线曝光（baseline exposure）----
# 相机厂商故意让 RAW 欠曝，把"把中点抬回来"留给转换器。这一层必须在入口一次补掉，
# 否则下游的曝光模块只好用"整张图的中位数"去补，而中位数是内容量、不是曝光量 ——
# 结果就是每张都被拽到同一个中间灰，亮的压暗、暗的提亮、暗部抬成灰（09-13 园岭实测）。
ENTRY_BIAS_ENABLE = True        # 总开关：入口按机型表 + RAF 的 DR tag 补回来
# 富士 DR 档的补量在 cameras.FUJI_DR_BIAS（100→0.72EV / 200→1.72 / 400→2.72）。
# ⚠ 逐张读 tag，不许按机型写死：同一台机身 DR 会变（实测 0191=DR200 / 0351=DR400）。

# 入口已经补过基线曝光的图，曝光层**不再允许提亮**（否则等于把补好的又拽一次）。
# 只有"入口补不了"（非富士 / 读不到 tag）时，才退回让曝光层兜底提亮。
AUTO_LIFT_ONLY_WHEN_NO_ENTRY_BIAS = True

# ========== L0 分析（判据：只用分位，绝不用均值） ==========
PCT_BLACK = 0.2                 # 黑点分位
PCT_MID = 50.0                  # 中灰分位
PCT_KNEE = 98.0                 # 高光膝点分位（肩部起点）
PCT_WHITE = 99.8                # 白点分位

TGT_BLACK = 0.010               # 绝对靶（显示域 0~1）= 灰度 2.5
TGT_MID = 0.550                 # 「偏暗」判据的参考（= 大师全体逐图『中位 L*』的中位 58.0）：
                                # 低于它 MID_DEADZONE 才判 'below'，而 'below' 只在
                                # 入口补不了基线曝光时才兜底提亮。**不再当"过亮护栏"**。
TGT_WHITE = 0.863               # 白点靶 ≈ 灰度 220（220 是本批素材实测最优；250 会把颜色放大 3.4 倍）
BLACK_PULL = 1.00               # 黑点最多下压的档数（防"画面本来没有暗部"被压死）

# ★ 过亮护栏（09-13 起走「折中」档：只有真的亮得离谱才管）
# 出处：1172 张大师成片逐图『中位 L*』的分布 —— P50=58.0 / P90=82.7 / **P95=87.0** / P99=94.1 / max=99.7
#   （`_debug/analysis/master_resurvey.json`，量法见 `_debug/lab_master_resurvey.py`）
# 取 P95：只有比 95% 的大师成片还亮，才算"亮得过分"；**压回的落点也是这条线**，不是拽到 TGT_MID。
# L*87.0 ↔ 中性灰显示域 0.854（= 灰阶 218）。
# ⚠ 为什么不能拽到 TGT_MID（灰阶 140）：那是"内容量当曝光量"的老病 —— 大师本来就有一半的片子中位在 58 以上。
GUARD_MID = 0.854

MID_DEADZONE = 0.030            # 「偏暗」判据的容差（低于 TGT_MID 这么多才算 below）
EV_CAP_DOWN = 1.8               # 最多压档数
EV_CAP_UP = 4.00                # 最多提档数（**只对"入口补不了基线曝光"的源生效**）
                                # ⚠ 别再为了"把欠曝的图拉回中灰靶"去调大它 —— 那是治标；
                                #   欠曝该在入口按 DR tag 补（见 ENTRY_BIAS_ENABLE）。

# 允不允许"提亮"。这是 RAW 与 JPG 最本质的分界：
#   RAW 线性有高光余量，提亮是"重新冲洗"；JPG 是相机曲线压过的成品，
#   提亮等于把被压缩的颜色按斜率放大（粉裙变鲜红就是这么来的）。
# ⚠ 09-13 起 RAW 的提亮只在"入口没能补基线曝光"时兜底用（AUTO_LIFT_ONLY_WHEN_NO_ENTRY_BIAS）。
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
# 抽色饱和（两端掉彩）：胶片在密度两端本来就保不住彩，人眼在极暗/极亮也钝。
# 公式出处 Emulsifier（`_debug/_rs/emul/engine.py`）：sat_mult = 1 - subsat*(2*luma-1)^2。
# 中间调（L*50）倍率 = 1 不动；纯黑/纯白 = 1-CHROMA_ENDS。
# ⚠ 高光那一端 L1 的 HILIGHT_DESAT 已经在管了，所以这里**主要是把暗部的彩压下来**
#   （09-13 园岭暗部泛紫红斑，就是这一块）。
CHROMA_ENDS = 0.20

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
# 四条候选（09-13 SV 拍板：默认走 ③ 全对齐；BASE_NONE 保留只作 A/B 对照，别当默认）：
#   fog      线性光域黑位抬升（"雾"）：黑位 L* 1.2 → 6.6、雾量 0.0009 → 0.0235
#   b        整体 b* 偏移：+3.26（偏黄）→ +1.0
#   chroma   彩度形状/量级：形状 3.61 → ~2.98、中位 5.57 → 7.16（闭式解 p/s）
#   contrast 明度对比（只动 L*）：反差 75 → ~78
# ⚠ 口径（别搞反）：**基准 = 把我方中性路径挪到「大师平均」**，**卷 = 相对「大师平均」的性格偏移**。
#   两者相加/相乘才落在作者线上（见 stocks.color_params）。所以改这个默认**不需要**重标卷。
# ★ ③ 全对齐 已于 09-13 **由数据重解**（入口补基线曝光之后，老那组手写数已作废）：
#   解它的脚本 = `_debug/calib_base_from_masters.py`（目标 = 大师 1170 张整体中位），
#   输入 = `效果debug/2026-09-13/卷标定_我方指纹/基线14_入口补偿后.json`（同一批 14 张探针）。
#   重解结果：整体退黄 1.80、彩度只需把中位从 5.9 提到 7.2（形状 3.52→2.98）、反差 75→79.6、
#   **雾量归零**（我们本来就比大师更雾：0.0494 vs 0.0235，fog 只能往上抬、压不下来）。
#   老那组手写数（b=-1.90 / p=0.770 / s=1.150 / contrast=1.35 / fog=0.0085）是按**老基线**
#   （彩度中位 5.57 / 形状 3.61）解的，照搬到入口补偿之后就是**补两遍**，故替换。
#   ★ v0.3.1 又重解了一次：新建了 CHROMA_ENDS（抽色饱和，两端掉彩）改动了彩度形状
#   （我方基线 形状 3.52→3.30），所以 chroma_p/s 要跟着重解。
#   输入 = `效果debug/2026-09-13/卷标定_我方指纹/基线14_抽色饱和后.json`，
#   重解结果：整体退黄 1.73、chroma_p 0.866→**0.913**、chroma_s 1.037→**0.959**、contrast 1.048 不变。
#   验证：14 探针（BASE_NONE 与 BASE_FULL 各自）四项全落带。
BASE = 'BASE_FULL'
BASE_TABLE = {
    'BASE_NONE': dict(
        label='不套基准（A/B 对照用）', desc='什么都不做：整体偏黄 +3.3、黑位死黑、几乎无雾',
        b=0.0, chroma_p=1.0, chroma_s=1.0, contrast=1.0, fog=0.0),
    'BASE_FOG': dict(
        label='① 只加雾', desc='黑位从死黑抬到胶片的灰黑、画面通透度略降；色偏与彩度不动',
        b=0.0, chroma_p=1.0, chroma_s=1.0, contrast=1.0, fog=0.0063),
    'BASE_DEYELLOW': dict(
        label='② 退黄+加雾', desc='在①之上把整体黄味从 +3.3 收到 +1 左右（画面明显"没那么黄"）',
        b=-1.90, chroma_p=1.0, chroma_s=1.0, contrast=1.0, fog=0.0063),
    'BASE_FULL': dict(
        label='③ 全对齐', desc='退黄 + 彩度微调 + 反差微调（09-13 v0.3.1 由当前基线重解；不再加雾）',
        a=0.47, b=-1.73, b_sh=-0.02, b_hi=1.73,
        chroma_p=0.913, chroma_s=0.959, contrast=1.048, fog=0.0),
}

# ========== 降噪（RAW 提亮后暗部色斑/噪点） ==========
# 位置：L1 修正之后、L2 风格之前（此时噪点被影调放大出来，最该收拾）。
# 原则：只在"暗部 + 平坦区"下手，边缘/细节区不动；用引导滤波（保边）逐通道做，
#       加回的量按掩膜加权 ⇒ 近似零均值 ⇒ 中灰/黑位几乎不漂移（selftest 有这条不变量）。
DENOISE_ENABLE = False
DENOISE_LUMA = 0.50             # 亮度通道降噪强度（0~1，会吃掉一点细节，别给大）
DENOISE_CHROMA = 0.90           # 色度通道降噪强度（色斑主要在这，可以给大）
DENOISE_DARK_LO = 55.0          # 掩膜：L* 低于此 = 全量降噪（暗部/中间调噪点最脏）
DENOISE_DARK_HI = 90.0          # 掩膜：L* 高于此 = 不降噪（高光本来就干净，别动皮肤高光）
DENOISE_EDGE_LO = 4.0           # 亮度掩膜：L* 梯度低于此 = 平坦区（全量降噪）
DENOISE_EDGE_HI = 18.0          # 亮度掩膜：梯度高于此 = 边缘/纹理（不降噪，别糊细节）
DENOISE_EDGE_LO_C = 12.0        # 色度掩膜同义（门槛更松）：色斑不产生 L* 梯度，卡太紧就白降了
DENOISE_EDGE_HI_C = 45.0
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
# ★ 高光端**精确归零**（09-13 调研修 bug）：原来写成"压掉 70%"，白墙/天空还留 30% 颗粒在动。
#   物理：密度饱和区没有可显影的银盐。外面（LIMO `applyGrainAsExposure`、
#   Emulsifier `grain_mask=(luma^0.5)(1-luma)^1.5`）两端都精确为 0。
#   注意我们的包络 `4g(1-g)` 顶端本来就偏肥（g=0.9 处还有峰值的 57%），所以窗口越到顶端越要收紧。
GRAIN_HI_LO = 0.86              # 开始收高光的位置（显示域灰度）
GRAIN_HI_HI = 0.98              # 到这里 = 0（★ 不许留残余）
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
# 分通道扩散半径比（R 最远 / G 中 / B 近零）—— 09-13 调研修正：
#   红光穿透片基散射得最远，蓝光几乎不散。出处：
#     LIMO `FilmShaderCommon.h`  RED/GREEN/BLUE_PENETRATION = 0.88/0.10/0.02
#     spektrafilm 三通道各自独立的散射 sigma
#   ⇒ 扩散半径按 [R, G, B] 缩放。红边因此是"一圈红、里层偏白"，而不是整块叠橙。
HALATION_RADIUS_RATIOS = [1.0, 0.45, 0.15]

# ========== L3 局部 ==========
SKIN_PROTECT = True
SKIN_PROTECT_STRENGTH = 0.55    # 肤色区彩度少降的比例

# ========== L4 护栏（只做"不许超过"，不做"必须等于"） ==========
CAP_WHITE_FRAC = 0.030          # 死白（>=254）占比上限
CAP_CHROMA_C90 = 45.0           # 彩度 P90 上限（Lab C）
GUARD_MAX_PASS = 3

# ========== 输出 ==========
JPEG_QUALITY = 95
