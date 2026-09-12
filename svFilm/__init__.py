# -*- coding: utf-8 -*-
"""svFilm —— 数码仿胶片管线（重写版）。

设计契约（改任何模块前先读这段）：
  L0 analyze  只读。判据只用分位（百分位），绝不用均值。
  L1 correct  影调修正。靶 = 绝对靶（0.01 / 0.55 / 0.86 显示域），
              只在"亮度域"上做曲线（RGB 同步缩放），不做逐通道曲线。
  L2 style    胶片风格。色交叉 / 色调分离 / 对比 / 外部 .cube。
              越过中灰就把它拉回 L1 的靶（风格不许改回修正）。
  L3 local    局部（肤色等）。可参考 L1 的成片做保护。
  L4 guard    护栏。只做"不许超过"，永不做"必须等于"。

另外两条：
  * 机器差异只关在 io.py（入口一个关口）里，其余模块必须与机型无关。
  * 一切可调参数只在 config.py，其它文件不许出现魔法数字。
"""
from . import config  # noqa: F401
from .pipeline import run, Result  # noqa: F401

__version__ = config.VERSION
__all__ = ['run', 'Result', 'config', '__version__']
