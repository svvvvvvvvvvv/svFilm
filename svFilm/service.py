# -*- coding: utf-8 -*-
r"""svFilm 常驻 HTTP 服务 —— 引擎对外的第二条口子（给工作台 svStudio 调）。

## 为什么必须常驻
实测（700 长边，2026-09-14）：

| 做法 | 换一次卷 |
|---|---|
| 每次调一下、起一个 Python 进程 | **5.03 s**（且解码缓存留不下来）|
| **常驻 + 缓存解码** | **1.45 s** |

差 3.5 倍 —— `io.load_raw` 一个人占全链 **57%**。**常驻是硬要求，不是优化选项。**

## 设计要点
- **零依赖**（只用标准库 `http.server`）—— 为了"能搬到别人机器上"。
- ★ **一条 `/render` 就够**：`GET /render?id=3&stock=portra400&side=700` → 图。
  前端直接 `<img src="...">`，**改参数 = 改 URL**，不用写任何取字节的代码。
- **只绑 127.0.0.1**（不对外）。
- 缓存 = `io.load()` 的产物（Sample），LRU 上限可调。

## 跑
```bash
$PY -m svFilm.service                 # 默认 127.0.0.1:8765
$PY -m svFilm.service --port 8800 --cache 40
```

## 接口
| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | `{ok, cached, version}` |
| GET | `/stocks` | 卷列表（name/label/desc） |
| GET | `/bases` | 基准成色列表 |
| GET | `/load?paths=a,b&side=700` | **同步**载入并缓存（慢，1.9 s/张；前端分批调） |
| GET | `/render?id=3&stock=&base=&side=&fmt=jpg` | 出图（直接返回图片字节） |
| GET | `/stats?id=3` | 只出数字，不出图（调参时看指标用） |
| GET | `/list` | 当前缓存里有什么 |
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
import urllib.parse
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from . import config as C
from . import io, pipeline, stocks

VERSION = 'svstudio-1'
DEFAULT_PORT = 8765
DEFAULT_SIDE = 700          # ★ 预览用 700（1080 屏根本显示不到 1000 px）
DEFAULT_CACHE = 40          # 最多缓存多少张

_cache = OrderedDict()      # id -> dict(sample=..., path=..., side=..., t=...)
_cache_lock = threading.Lock()
_next_id = [1]


def _cache_put(path, side, sample):
    with _cache_lock:
        i = _next_id[0]
        _next_id[0] += 1
        _cache[i] = dict(sample=sample, path=path, side=side, t=time.time())
        while len(_cache) > _CACHE_MAX[0]:
            _cache.popitem(last=False)
    return i


def _cache_get(i):
    with _cache_lock:
        return dict(_cache.get(i) or {})


def _load_one(path, side):
    """★ 慢的那一步（解码 + 入口 + 锚点）—— 只在这里做一次。"""
    t = time.perf_counter()
    s = io.load(path, side, src=None)
    return s, (time.perf_counter() - t) * 1000.0


def _render_bytes(i, stock, base, side, fmt, quality, params=None):
    """出图。side 与缓存不一致时用缓存的（不重新解码）—— 前端要别的尺寸得重新 /load。"""
    row = _cache_get(i)
    if not row:
        return None, {'error': 'id 不在缓存里，先 /load'}
    s = row['sample']
    with _Overrides(_parse_params(params)):
        r = pipeline.run_from(s, stock=stock or None, base=base or None)
    disp = np.clip(r.disp, 0.0, 1.0)
    arr = (disp * 255.0 + 0.5).astype(np.uint8)
    if fmt in ('jpg', 'jpeg'):
        from PIL import Image
        import io as _io
        buf = _io.BytesIO()
        Image.fromarray(arr).save(buf, format='JPEG', quality=int(quality))
        return buf.getvalue(), {'ms': round(r.report.get('ms', 0)),
                                'mime': 'image/jpeg'}
    from PIL import Image
    import io as _io
    buf = _io.BytesIO()
    Image.fromarray(arr).save(buf, format='PNG')
    return buf.getvalue(), {'ms': round(r.report.get('ms', 0)), 'mime': 'image/png'}


def _stats_of(i, stock, base, params=None):
    row = _cache_get(i)
    if not row:
        return {'error': 'id 不在缓存里'}
    with _Overrides(_parse_params(params)):
        r = pipeline.run_from(row['sample'], stock=stock or None, base=base or None)
    from . import color
    lab = color.to_lab(np.clip(r.disp, 0, 1))
    L = lab[..., 0]
    Cc = np.sqrt(lab[..., 1] ** 2 + lab[..., 2] ** 2)
    return dict(ms=round(r.report.get('ms', 0)),
                stock=r.report.get('stock'), base=r.report.get('base'),
                L50=round(float(np.median(L)), 1),
                L90=round(float(np.percentile(L, 90)), 1),
                b=round(float(np.median(lab[..., 2])), 2),
                c50=round(float(np.median(Cc)), 2))


class _H(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *a):        # 别刷屏
        pass

    def _json(self, obj, code=200):
        b = json.dumps(obj, ensure_ascii=False).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(b)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(b)

    def _img(self, b, mime):
        self.send_response(200)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(len(b)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        q = {k: v[0] for k, v in urllib.parse.parse_qs(u.query).items()}
        try:
            if u.path == '/health':
                with _cache_lock:
                    n = len(_cache)
                return self._json(dict(ok=True, version=VERSION, cached=n,
                                       side=DEFAULT_SIDE))
            if u.path == '/stocks':
                out = []
                for n in ('neutral', 'portra400', 'pro400h', 'fuji_c200',
                          'ektar100', 'cinestill800t', 'air'):
                    s = stocks.get(n) or {}
                    out.append(dict(name=n, label=s.get('label') or n,
                                    desc=s.get('desc') or ''))
                return self._json(out)
            if u.path == '/bases':
                out = []
                for n, d in (C.BASE_TABLE or {}).items():
                    out.append(dict(name=n, label=d.get('label') or n,
                                    desc=d.get('desc') or ''))
                return self._json(out)
            if u.path in ('/', '/index.html') and _WEB[0]:
                # ★ 可选：把工作台的静态页 serve 出来（路径由 `--web` 给，**不写死** ⇒ 边界不破）
                fp = os.path.join(_WEB[0], 'index.html')
                if os.path.exists(fp):
                    with open(fp, 'rb') as f:
                        b = f.read()
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/html; charset=utf-8')
                    self.send_header('Content-Length', str(len(b)))
                    self.end_headers()
                    self.wfile.write(b)
                    return
            if u.path == '/scan':
                # 只列**用户指定的**目录（不自己去翻盘）—— 按 .RAF/.JPG 收，自然序排
                d = q.get('dir') or ''
                if not d or not os.path.isdir(d):
                    return self._json({'error': '目录不存在: %s' % d}, 400)
                exts = tuple('.' + e.strip().lower()
                             for e in (q.get('ext') or 'raf,jpg,jpeg').split(',') if e.strip())
                lim = int(q.get('limit') or 400)
                ps = []
                for n in sorted(os.listdir(d)):
                    if n.lower().endswith(exts):
                        ps.append(os.path.join(d, n).replace('\\', '/'))
                return self._json(dict(dir=d, n=len(ps), files=ps[:lim]))
            if u.path == '/params':
                out = []
                for p in PARAMS:
                    d = dict(p)
                    d['value'] = getattr(C, p['k'], None)
                    out.append(d)
                return self._json(out)
            if u.path == '/list':
                with _cache_lock:
                    out = [dict(id=i, path=v['path'], side=v['side'])
                           for i, v in _cache.items()]
                return self._json(out)
            if u.path == '/load':
                ps = [x.strip() for x in (q.get('paths') or '').split(',') if x.strip()]
                side = int(q.get('side') or DEFAULT_SIDE)
                if not ps:
                    return self._json({'error': '缺 paths'}, 400)
                out = []
                for p in ps:
                    if not os.path.exists(p):
                        out.append(dict(path=p, error='文件不存在'))
                        continue
                    s, ms = _load_one(p, side)
                    out.append(dict(id=_cache_put(p, side, s), path=p, ms=round(ms)))
                return self._json(out)
            if u.path == '/render':
                b, info = _render_bytes(int(q.get('id') or 0), q.get('stock'),
                                        q.get('base'), q.get('side'),
                                        (q.get('fmt') or 'jpg').lower(),
                                        q.get('q') or 92, q.get('params'))
                if b is None:
                    return self._json(info, 404)
                return self._img(b, info['mime'])
            if u.path == '/stats':
                return self._json(_stats_of(int(q.get('id') or 0), q.get('stock'),
                                            q.get('base'), q.get('params')))
            return self._json({'error': 'no such path', 'path': u.path}, 404)
        except Exception as e:                                    # noqa: BLE001
            return self._json({'error': '%s: %s' % (type(e).__name__, str(e)[:200])}, 500)


# ---- ★ 通用参数口子：`params=KEY:VAL,KEY:VAL` 临时覆盖 config -------------------
# 为什么这么做：**不用为每一个旋钮写代码**。前端只要知道「哪个参数、什么范围」，
# 就能自动生成滑杆；引擎这边一个口子全接住。
_param_lock = threading.Lock()

# 允许被外部覆盖的前缀（**白名单**，防止前端乱改引擎契约里的东西）
PARAM_PREFIX = ('TONE_', 'GRAIN_', 'BLOOM_', 'HALATION_', 'DENOISE_', 'WHITE_MICRO',
                'FACE_', 'DENSITY_', 'CROSSTALK_', 'LAYER_', 'ENTRY_', 'SHADOW_',
                'COLOR_', 'SKIN_', 'ANCHOR_', 'CAP_', 'SHARP_')

# 不收白名单里的这些（结构性/开关类，乱改会破契约）
PARAM_BLOCK = ('ENTRY_CURVE', 'ENTRY_SHOULDER_KIND', 'LUT_PATH', 'BASE', 'STOCK')


def _parse_params(txt):
    """`KEY:VAL,KEY:VAL` → dict。不合法/不在白名单的**静默丢掉**（不让前端报错卡住）。"""
    out = {}
    for seg in (txt or '').split(','):
        if ':' not in seg:
            continue
        k, v = seg.split(':', 1)
        k, v = k.strip(), v.strip()
        if not k or k in PARAM_BLOCK:
            continue
        if not any(k.startswith(p) for p in PARAM_PREFIX):
            continue
        if not hasattr(C, k):
            continue
        cur = getattr(C, k)
        if isinstance(cur, (int, float)) and not isinstance(cur, bool):
            try:
                out[k] = float(v)
            except ValueError:
                pass
    return out


class _Overrides:
    """临时覆盖 config（**用完还原**）。常驻服务是多线程的 ⇒ 加锁。"""

    def __init__(self, kv):
        self.kv = kv or {}

    def __enter__(self):
        self.old = {}
        _param_lock.acquire()
        for k, v in self.kv.items():
            self.old[k] = getattr(C, k)
            cur = self.old[k]
            setattr(C, k, int(round(v)) if isinstance(cur, int) else float(v))
        return self

    def __exit__(self, *a):
        for k, v in self.old.items():
            setattr(C, k, v)
        _param_lock.release()
        return False


# ---- 前端要的「可调参数清单」（带范围）--------------------------------------
# 只列**真值得给人拧的**那几个 —— 别把 config 里 200 个常量全倒出来。
PARAMS = [
    dict(k='TONE_LIFT',       name='中高调抬起',   lo=0,  hi=16,  step=0.5, d='整张变亮（也会带出高光肩部）'),
    dict(k='TONE_TOE',        name='趾部压深',     lo=0,  hi=2.0, step=0.05, d='暗部压深的量'),
    dict(k='TONE_SHOULDER',   name='高光肩部',     lo=0,  hi=6,   step=0.5, d='大 = 高光收得多；小 = 开顶'),
    dict(k='GRAIN_AMOUNT',    name='颗粒',        lo=0,  hi=0.10, step=0.002, d='颗粒强度'),
    dict(k='BLOOM_AMOUNT',    name='黑柔',        lo=0,  hi=0.20, step=0.005, d='高光溢出（黑柔）强度'),
    dict(k='HALATION_AMOUNT', name='红橙晕圈',     lo=0,  hi=0.30, step=0.01, d='高光边缘的红橙光晕'),
    dict(k='WHITE_MICRO',     name='白区层次',     lo=0,  hi=1.5, step=0.1, d='白衣服/白墙的微反差'),
]
_PARAM_KEYS = tuple(p['k'] for p in PARAMS)


_CACHE_MAX = [DEFAULT_CACHE]
_WEB = [None]        # --web <dir>：要不要顺手 serve 一个静态前端（可选）


def serve(port=DEFAULT_PORT, host='127.0.0.1', cache=DEFAULT_CACHE, web=None):
    _CACHE_MAX[0] = int(cache)
    _WEB[0] = web
    srv = ThreadingHTTPServer((host, int(port)), _H)
    srv.daemon_threads = True
    print('svFilm 服务已起： http://%s:%d   （缓存上限 %d 张，预览长边 %d）'
          % (host, port, _CACHE_MAX[0], DEFAULT_SIDE))
    if web:
        print('  工作台页面： http://%s:%d/   （静态目录 %s）' % (host, port, web))
    else:
        print('  试一下： curl "http://%s:%d/health"' % (host, port))
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print('\n停了。')


def main(argv=None):
    ap = argparse.ArgumentParser('svFilm.service', description='svFilm 常驻 HTTP 服务')
    ap.add_argument('--port', type=int, default=DEFAULT_PORT)
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--cache', type=int, default=DEFAULT_CACHE)
    ap.add_argument('--web', default=None,
                    help='可选：顺手 serve 一个静态前端目录（例如 ../svStudio/web）')
    a = ap.parse_args(argv)
    serve(a.port, a.host, a.cache, a.web)


if __name__ == '__main__':
    main()
