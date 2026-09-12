# -*- coding: utf-8 -*-
"""把整个 svFilm git 仓库（含 .git 历史）打成一个 tar.gz，用来同步/备份到资料库。

为什么是 .tar.gz 不是 .zip：资料库把 .zip / .html 当"可编辑网页"导入，传代码仓库会走错通道。
.tar.gz 是普通文件，走网盘，能原样下回来。

用法：
  python _debug/export_repo.py                 # 打到 ../_export/
  python _debug/export_repo.py --out <目录>

打完会打印下一步要走的命令（上传要联网鉴权，只能由助理在会话里执行）。
"""
import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GIT = r'C:\Users\psw99\.workbuddy\binaries\PortableGit\versions\1.2.0\cmd\git.exe'
SKILL = ('D:/Program Files/wb/WorkBuddy/resources/app.asar.unpacked/resources/'
         'plugins/workbuddy-builtin/skills/library')


def last_commit():
    try:
        out = subprocess.run([GIT, 'log', '-1', '--pretty=%h %s'], cwd=ROOT,
                             capture_output=True, text=True, encoding='utf-8')
        return (out.stdout or '').strip()
    except Exception as e:                                    # noqa: BLE001
        return '（取不到 git log: %s）' % e


def dirty():
    try:
        out = subprocess.run([GIT, 'status', '--porcelain'], cwd=ROOT,
                             capture_output=True, text=True, encoding='utf-8')
        return [l for l in (out.stdout or '').splitlines() if l.strip()]
    except Exception:                                          # noqa: BLE001
        return []


def export(out_dir, name=None):
    d = [l for l in dirty()]
    if d:
        print('⚠ 工作区有未提交的改动，建议先 commit 再导出：')
        for l in d[:20]:
            print('   ' + l)
    ver = 'v?'
    cfg = os.path.join(ROOT, 'svFilm', 'config.py')
    if os.path.exists(cfg):
        for line in open(cfg, encoding='utf-8'):
            if line.startswith('VERSION'):
                ver = line.split('=', 1)[1].strip().strip('\'"')
                break
    tag = '%s_%s' % (ver, time.strftime('%Y-%m-%d'))
    fn = name or ('svFilm_git-repo_%s.tar.gz' % tag)
    os.makedirs(out_dir, exist_ok=True)
    dst = os.path.join(out_dir, fn)
    if os.path.exists(dst):
        os.remove(dst)
    base = os.path.basename(ROOT)
    subprocess.run(['tar', '-czf', dst, '--exclude=__pycache__', '--exclude=*.pyc',
                    '-C', os.path.dirname(ROOT), base], check=True)
    size = os.path.getsize(dst)
    print('归档: %s  (%.1f KB)' % (dst, size / 1024.0))
    print('版本: %s   最后提交: %s' % (ver, last_commit()))
    print('下一步（在会话里让助理执行，需要联网鉴权）：')
    print('  printf \'%%s\' "<token>" | python3 "%s/drive/upload_drive_file.py" '
          '--token-stdin "%s" --file-name "%s"' % (SKILL, dst, fn))
    return dst


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(os.path.dirname(ROOT), '_export'))
    ap.add_argument('--name', default=None)
    a = ap.parse_args()
    export(a.out, a.name)
    sys.exit(0)
