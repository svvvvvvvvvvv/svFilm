# svFilm

数码仿胶片管线（重写版）。**代码与 `auto_pipe` 完全独立，只沿用原理，不共用任何一行。**

## 分层契约（改任何一层前先看这张表）

| 层 | 文件 | 输入 → 输出 | 职责 | 铁律 |
|---|---|---|---|---|
| L0 | `analyze.py` | lin/disp → dict | 只读分析 | **判据只用分位，绝不用均值** |
| L1 | `tone.py` | lin → lin | 影调修正 | **靶是绝对靶**；曲线只作用在亮度上；可解析求中灰 |
| L2 | `style.py` | disp → disp | 胶片风格、外部 .cube | **不许把 L1 定下的中灰改回去**（`lock_ref` 保证） |
| ┃ | `spatial.py` | disp → disp | **颗粒 / 黑柔 Bloom / Halation** | 邻域运算，**塞不进 LUT**，必须单独一层 |
| L3 | `local.py` | disp → disp | 肤色等局部 | 只改色度，不改明度 |
| L4 | `guard.py` | disp → disp | 护栏 | **只做"不许超过"，永不做"必须等于"** |
| 入口 | `io.py` | 文件 → lin/disp | 解码 + IDT + 归一 | **全工程唯一允许出现"机型"的地方**（`cameras.py` 是数据表） |
| 路径 | `paths.py` | 目的 → 目录 | 产出落点 | **所有效果图一律走这里，不许在别处拼路径** |
| 数据 | `stocks.py` | 卷名 → 参数 | **胶片卷表** | 换卷只改数据不改代码，新卷加一行 |

一切可调参数只在 `config.py`。改核心文件前先 commit。

### 卷（甲）与空间域（乙丙丁）

一个"卷"就是一份数据，两半：

* **颜色**（L2 用）：色交叉矩阵 / 暗部色调 / 亮部色调 / 彩度 / 明度对比
* **空间**（`spatial.py` 用）：颗粒 / 黑柔 / Halation 的强度

现有 7 卷：`neutral`（什么都不做的基准）、`portra400`、`pro400h`、`fuji_c200`、`ektar100`、
`cinestill800t`（带 Halation）、`air`。

为什么空间三件必须单独一层：`.cube` 是**逐像素查表**，只能装颜色和影调；
颗粒、黑柔（光学扩散）、Halation（片基红光散射）都是**看邻域**的运算，LUT 装不下。

## 目录规范

```
svFilm/                    工程（git 仓库）
  svFilm/                  包：L0~L4 + 空间域 + 入口 + 路径 + 卷表
  _debug/                  调试/出图脚本（不进生产链，产物全走 paths.py）
  <样片目录>/效果debug/<日期 YYYY-MM-DD>/<目的说明>/<文件>
```

- 一级 `效果debug`：所有"给人看效果"的产物归到这里，不再散落在样片旁边。
- 二级日期默认今天，三级是这批图**为了看什么**（例：`三张RAW_分段对照`）。
- `cli dir` 扫描时**整棵跳过 `效果debug`**，避免把上一轮的产出当素材再处理一遍。

## 常用命令

```bash
PY=C:/Users/psw99/.workbuddy/binaries/python/envs/default/Scripts/python.exe   # 在本目录下执行
GIT=C:/Users/psw99/.workbuddy/binaries/PortableGit/versions/1.2.0/cmd/git.exe  # 本机 git 用 WorkBuddy 自带那份

$PY -m svFilm.selftest                                  # 动完任何一层都要跑
$PY -m svFilm.cli stocks                                # 列出所有胶片卷（人话说明）
$PY -m svFilm.cli probe <img...> --after [--stock 卷名]  # 只看数，不写文件
$PY -m svFilm.cli one   <img> -o out.jpg --stock 卷名    # 单张
$PY -m svFilm.cli dir   <in> -o <out> --jobs 4 --stock 卷名   # 批（重活 jobs<=4，输出名带 _svFilm_卷名）
$PY -m svFilm.cli bake  x.cube --size 33 --stock 卷名 --purpose <目的> --root <样片目录>   # 把某卷烘成 .cube
$PY -m svFilm.cli calib <RAW+JPG 同名对...>              # 量机型表要填的 baseline_ev

$PY _debug/make_sheet.py        --purpose <目的> <stem...>   # 分段：底/只修正/出片
$PY _debug/make_stock_sheet.py  --purpose <目的> <stem...>   # 卷对照：一张图 x 各卷
$PY _debug/make_spatial_sheet.py --purpose <目的> <stem...>  # 空间域：逐个开颗粒/黑柔/Halation（默认 1:1 切图）
$PY _debug/make_crop.py         --purpose <目的> <stem...>   # 100% 细节切图
```

## 还没做（下一步）

- **没有降噪层**：RAW 提 3~4 档会把暗部传感器色斑/噪点一起放大（看 100% 细节图最明显）。
- 卷的颜色参数目前是"按各卷公开性格手写"的近似，还没用色卡实拍标定。
- 外层这版只是"一次冲洗、三锚点"；胶片真实的欠曝/正常/过曝三态（同一卷三种性格）没做。
