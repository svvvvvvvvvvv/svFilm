# svFilm

数码仿胶片管线（重写版）。**代码与 `auto_pipe` 完全独立，只沿用原理，不共用任何一行。**

## 分层契约（改任何一层前先看这张表）

| 层 | 文件 | 输入 → 输出 | 职责 | 铁律 |
|---|---|---|---|---|
| L0 | `analyze.py` | lin/disp → dict | 只读分析 | **判据只用分位，绝不用均值** |
| L1 | `tone.py` | lin → lin | 影调修正 | **靶是绝对靶**；曲线只作用在亮度上；可解析求中灰 |
| L2 | `style.py` | disp → disp | 胶片风格、外部 .cube | **不许把 L1 定下的中灰改回去**（`lock_ref` 保证） |
| L3 | `local.py` | disp → disp | 肤色等局部 | 只改色度，不改明度 |
| L4 | `guard.py` | disp → disp | 护栏 | **只做"不许超过"，永不做"必须等于"** |
| 入口 | `io.py` | 文件 → lin/disp | 解码 + IDT + 归一 | **全工程唯一允许出现"机型"的地方**（`cameras.py` 是数据表） |
| 路径 | `paths.py` | 目的 → 目录 | 产出落点 | **所有效果图一律走这里，不许在别处拼路径** |

一切可调参数只在 `config.py`。改核心文件前先 commit。

## 目录规范

```
svFilm/                    工程（git 仓库）
  svFilm/                  包：L0~L4 + 入口 + 路径
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
$PY -m svFilm.cli probe <img...> --after                # 只看数，不写文件
$PY -m svFilm.cli one   <img> -o out.jpg                # 单张
$PY -m svFilm.cli dir   <in> -o <out> --jobs 4          # 批（重活 jobs<=4，输出名带 _svFilm 后缀）
$PY -m svFilm.cli bake  x.cube --size 33 --purpose <目的> --root <样片目录>   # 烘 L2
$PY -m svFilm.cli calib <RAW+JPG 同名对...>              # 量机型表要填的 baseline_ev

$PY _debug/make_sheet.py --purpose <目的> <stem...>        # 分段对照图（底/只修正/出片）
$PY _debug/make_crop.py  --purpose <目的> <stem...>        # 100% 细节切图
```

## 还没做（下一步）

- L2 的胶片性格目前只是很轻微的一层占位（色交叉 + 分裂色调），真正的"卷"要靠外部 `.cube` 或离线烘焙。
- 空间域的三件（颗粒 / 黑柔 = 光学扩散 / Halation）还没写，它们塞不进 LUT，必须单独一层。
- 暗部提亮 3 档以上会带出传感器色斑，还没有降噪层。
