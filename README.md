# svFilm

数码仿胶片管线（重写版）。**代码与 `auto_pipe` 完全独立，只沿用原理，不共用任何一行。**

## 分层契约（改任何一层前先看这张表）

| 层 | 文件 | 输入 → 输出 | 职责 | 铁律 |
|---|---|---|---|---|
| L0 | `analyze.py` | lin/disp → dict | 只读分析 | **判据只用分位，绝不用均值** |
| L1 | `tone.py` | lin → lin | 影调修正 | **靶是绝对靶**；曲线只作用在亮度上；可解析求中灰 |
| ┃ | `denoise.py` | disp → disp | **降噪** | **只在暗部×平坦区下手**；近似零均值（分位不漂）；不打平边缘 |
| L2 | `style.py` | disp → disp | 胶片风格、外部 .cube | **不许把 L1 定下的中灰改回去**（`lock_ref` 保证） |
| ┃ | `spatial.py` | disp → disp | **颗粒 / 黑柔 Bloom / Halation** | 邻域运算，**塞不进 LUT**，必须单独一层 |
| L3 | `local.py` | disp → disp | 肤色等局部 | 只改色度，不改明度 |
| L4 | `guard.py` | disp → disp | 护栏 | **只做"不许超过"，永不做"必须等于"** |
| 入口 | `io.py` | 文件 → lin/disp | 解码 + IDT + 归一 | **全工程唯一允许出现"机型"的地方**（`cameras.py` 是数据表） |
| 路径 | `paths.py` | 目的 → 目录 | 产出落点 | **所有效果图一律走这里，不许在别处拼路径** |
| 数据 | `stocks.py` | 卷名 → 参数 | **胶片卷表** | 换卷只改数据不改代码，新卷加一行 |

一切可调参数只在 `config.py`。改核心文件前先 commit。

### 基准成色（不属于任何卷）vs 卷

颜色分两层，职责别混：

* **基准成色**（`config.BASE_TABLE`）= **我方中性路径相对"大师平均"的系统性差**。
  四档：`BASE_NONE`（什么都不做，只留作 A/B 对照）/ `BASE_FOG`（只加雾）/ `BASE_DEYELLOW`（退黄+加雾）/
  `BASE_FULL`（全对齐：退黄 + 彩度微调 + 反差微调）。`cli bases` 能列出来。
  **默认 = `BASE_FULL`（09-13 SV 拍板）**；`config.BASE` 改这一行即可换默认，命令行 `--base` 可逐次覆盖。
  ⚠ `BASE_FULL` 的数**不是手写的**：09-13 入口补基线曝光之后，
  老那组手写数（照老基线解的）会"补两遍"，所以用 `_debug/calib_base_from_masters.py`
  **按当前基线重解了一遍**（目标 = 大师 1170 张整体中位）。改基线就得重跑它。
* **卷**（`stocks.py`）= **这条作者线相对"大师平均"的性格偏移**。

组装顺序 `cfg 默认 → 基准 → 卷`：偏移类（a/b/b_sh/b_hi）**相加**，彩度与对比**相乘**。
基准负责把我们的中性路径挪到"大师平均"上，卷再往作者线上偏 —— 换基准不会废掉卷。


### 卷（甲）与空间域（乙丙丁）

一个"卷"就是一份数据，两半：

* **颜色**（L2 用）：a\*/b\* 偏移、暗部·亮部 b\* 分离、彩度(p, s)、明度对比
* **空间**（`spatial.py` 用）：颗粒 / 黑柔 / Halation 的强度

现有 7 卷：`neutral`（什么都不做的基准）、`portra400`、`pro400h`、`fuji_c200`、`ektar100`、
`cinestill800t`（带 Halation）、`air`。

**数值是量出来的，不是手编的**（v0.2.1 起）：卷名/方向继承 09-10《胶片卷映射与分组策略》，
颜色数值由 `_debug/calib_stocks_from_masters.py` 从 **1170 张大师成片**（`master_resurvey.json`）量出。
口径 =「取神不取形」：**卷 = 这条作者线相对"大师全体中位"的性格偏移**。
逐卷证据 → `效果debug/<日期>/卷标定_大师颜色聚类/卷标定报告.md`。
只有 `pro400h` 没标定（数据里没有"青绿粉彩"那条线）→ `calibrated=False`，仍是手写。

为什么空间三件必须单独一层：`.cube` 是**逐像素查表**，只能装颜色和影调；
颗粒、黑柔（光学扩散）、Halation（片基红光散射）都是**看邻域**的运算，LUT 装不下。

## 目录规范

```
svFilm/                    工程（git 仓库，只放产品代码）
  svFilm/                  包：L0~L4 + 空间域 + 入口 + 路径 + 卷表
  <样片目录>/效果debug/<日期 YYYY-MM-DD>/<目的说明>/<文件>

E:\WorkBuddy\摄影助手\
  _debug/                  调试/出图脚本 + 分析数据（**在仓库外**：含隐私路径/网盘路径，不进 git）
```

- 一级 `效果debug`：所有"给人看效果"的产物归到这里，不再散落在样片旁边。
- 二级日期默认今天，三级是这批图**为了看什么**（例：`三张RAW_分段对照`）。
- `cli dir` 扫描时**整棵跳过 `效果debug`**，避免把上一轮的产出当素材再处理一遍。
- 调试脚本**不在工程里**：它们写着样片/大师作品的绝对路径和作者名，进仓库等于泄露。
  在工程根执行时用 `../_debug/xxx.py`；脚本内部靠 `SVFILM_ROOT` 环境变量找工程根（默认 `../svFilm`）。

## 常用命令

```bash
PY=C:/Users/psw99/.workbuddy/binaries/python/envs/default/Scripts/python.exe   # 在本目录下执行
GIT=C:/Users/psw99/.workbuddy/binaries/PortableGit/versions/1.2.0/cmd/git.exe  # 本机 git 用 WorkBuddy 自带那份

$PY -m svFilm.selftest                                  # 动完任何一层都要跑
$PY -m svFilm.cli stocks                                # 列出所有胶片卷（人话说明）
$PY -m svFilm.cli bases                                 # 列出基准成色候选（中性路径对齐大师平均）
$PY -m svFilm.cli probe <img...> --after [--stock 卷名] [--base 基准名]  # 只看数，不写文件
$PY -m svFilm.cli one   <img> -o out.jpg --stock 卷名 --base 基准名   # 单张
$PY -m svFilm.cli dir   <in> -o <out> --jobs 4 --stock 卷名 --base 基准名   # 批（重活 jobs<=4）
$PY -m svFilm.cli dir   <选片目录> -o <out> --pair <拍摄目录> --raw-only   # 选片只有 JPG 时：去拍摄目录取同名 RAW
$PY -m svFilm.cli bake  x.cube --size 33 --stock 卷名 --base 基准名 --purpose <目的> --root <样片目录>
$PY -m svFilm.cli calib <RAW+JPG 同名对...>              # 量机型表要填的 baseline_ev

$PY ../_debug/make_sheet.py        --purpose <目的> <stem...>   # 分段：底/只修正/出片
$PY ../_debug/make_stock_sheet.py  --purpose <目的> <stem...>   # 卷对照：一张图 x 各卷
$PY ../_debug/make_spatial_sheet.py --purpose <目的> <stem...>  # 空间域：逐个开颗粒/黑柔/Halation（默认 1:1 切图）
$PY ../_debug/make_crop.py         --purpose <目的> <stem...>   # 100% 细节切图
$PY ../_debug/lab_ours_fingerprint.py --stocks neutral --bases BASE_NONE,BASE_FOG --tag 基线  # 量我方指纹 / 验收落带
$PY ../_debug/calib_stocks_from_masters.py --ours <基线.json>   # 从大师作品量出卷参数 → 抄回 stocks.py
```

### 卷的标定链（路 B，改口径必须整条重跑）

```
lab_ours_fingerprint.py --stocks neutral  →  基线14.json        （我方起点）
master_resurvey.json（1170 张大师成片）   →  大师九条线的指纹
        ↓ calib_stocks_from_masters.py
   stock_calib.json + 卷标定报告.md  →  抄进 svFilm/stocks.py 的 TABLE
        ↓
lab_ours_fingerprint.py --stocks <各卷> --tag 验收  →  看有没有落进大师带
```

## 还没做（下一步）

- 卷要"名副其实"（叫 Portra 就真是 Portra）得走**路 A：色卡实拍标定**；现在量到的是"这些摄影师后期风格"（取神不取形）。
- 外层这版只是"一次冲洗、三锚点"；胶片真实的欠曝/正常/过曝三态（同一卷三种性格）没做。
- 档位/强度还是定档，没有"跟随外部 LUT 动态"的默认档。

## 已做完（别重复做）

- **降噪层**（`denoise.py`）：暗部色斑降 30~52%，亮度只在平坦区降 2~7%（纹理区不动）。
- **基准成色三档**（`config.BASE_TABLE`）：`BASE_FULL` 让 14 张探针**四项全落大师带**。
- **`contrast` 方向修正**：曾把 S 形符号写反（+a·sin ⇒ 实际在降对比），
  导致所有卷"标定说反差大、落地在降对比"。现已改对，selftest 有专门一条守着。
- **`cli dir --pair / --raw-only`**：选片目录只有 JPG 时，去拍摄目录取同名 RAW 再跑。
