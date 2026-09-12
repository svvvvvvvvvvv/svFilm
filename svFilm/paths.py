# -*- coding: utf-8 -*-
"""产出路径规范。**所有效果图/调试产物一律走这里，不许在别处拼路径。**

约定（SV 定的）：

    <root>/效果debug/<日期 YYYY-MM-DD>/<目的说明>/<文件名>

  * 一级 `效果debug`  —— 所有"给人看效果"的产物都归到这下面，不再散落在样片旁边
  * 二级 日期          —— 默认取今天
  * 三级 目的说明      —— 这一批图是为了看什么（例：三张RAW_分段对照）

`root` 默认 = 样片所在目录（效果图和样片放一起，方便对照着看）。
"""
import os
import time

TOP = '效果debug'


def today():
    return time.strftime('%Y-%m-%d')


def slug(s):
    """把目的说明收拾成安全的目录名：去首尾空格、把路径分隔符换成下划线。"""
    s = (s or '').strip()
    for ch in '\\/:*?"<>|':
        s = s.replace(ch, '_')
    return s or '未命名'


def debug_dir(purpose, root='.', date=None):
    """建好并返回 <root>/效果debug/<日期>/<目的>/。"""
    p = os.path.join(root, TOP, date or today(), slug(purpose))
    os.makedirs(p, exist_ok=True)
    return p


def debug_path(purpose, filename, root='.', date=None):
    return os.path.join(debug_dir(purpose, root, date), filename)
