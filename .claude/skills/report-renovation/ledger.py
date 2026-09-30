#!/usr/bin/env python3
"""把翻新完的篇目在执笔记录里改为「现行」,只暂存「HEAD 版本 + 这几行」。

用法(在仓库根目录):
  python3 .claude/skills/report-renovation/ledger.py --lib reports --date 2026-09-30 DeepSeek/NSA Moonshot/MoBA
  加 --dry-run 只打印将要做的改动,不写文件、不动暂存区。

做的事:
  1. 从 HEAD 读执笔记录,把每个篇目行的「待重审重写 | 旧执笔 | 旧日期 | 把握」改成「现行 | 新执笔 | 日期 | 高」,
     并按旧执笔方把汇总表的篇数挪到新执笔方,现行 +n、待重审重写 −n
  2. 把改好的 HEAD 版本写进暂存区(git update-index),工作区里别的线未提交的行不会被暂存
  3. 对工作区文件做同样的改动:先把内容读进变量、算好,再以写模式打开写回(不先截断)
"""
import argparse
import re
import subprocess
import sys


def transform(text, rows, writer, date):
    moved = {}
    for key in rows:
        pat = re.compile(
            r'^\| ' + re.escape(key) + r' \| 待重审重写 \| ([^|]+?) \| [^|]+ \| [^|]+ \| — \|$', re.M)
        found = pat.findall(text)
        if len(found) != 1:
            sys.exit(f'找不到或不唯一的待重审行:{key}(命中 {len(found)} 处)')
        old = found[0].strip()
        moved[old] = moved.get(old, 0) + 1
        text = pat.sub(f'| {key} | 现行 | {writer} | {date} | 高 | — |', text)
    deltas = {name: -n for name, n in moved.items()}
    deltas[writer] = deltas.get(writer, 0) + len(rows)
    deltas['现行'] = len(rows)
    deltas['待重审重写'] = -len(rows)
    for name, d in deltas.items():
        if d == 0:
            continue
        pat = re.compile(r'^\| ' + re.escape(name) + r' \| (\d+) \|$', re.M)
        m = pat.findall(text)
        if len(m) != 1:
            sys.exit(f'汇总表里找不到或不唯一:{name}')
        text = pat.sub(f'| {name} | {int(m[0]) + d} |', text)
    return text, deltas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lib', choices=['reports', 'readings'], required=True)
    ap.add_argument('--date', required=True)
    ap.add_argument('--writer', default='Claude')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('rows', nargs='+', help='公司/材料 或 方向/材料')
    a = ap.parse_args()
    path = f'{a.lib}/_执笔记录.md'

    head = subprocess.run(['git', 'show', f'HEAD:{path}'], capture_output=True, text=True, check=True).stdout
    staged, deltas = transform(head, a.rows, a.writer, a.date)
    with open(path, encoding='utf-8') as f:
        work = f.read()
    new_work, _ = transform(work, a.rows, a.writer, a.date)

    print('汇总表增量:', {k: v for k, v in deltas.items() if v})
    if a.dry_run:
        print('dry-run:未写文件、未动暂存区')
        return
    blob = subprocess.run(['git', 'hash-object', '-w', '--stdin'], input=staged,
                          capture_output=True, text=True, check=True).stdout.strip()
    subprocess.run(['git', 'update-index', '--cacheinfo', '100644', blob, path], check=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(new_work)
    print(f'已暂存 {path}(HEAD + {len(a.rows)} 行),工作区同步改好;用 git diff --cached --numstat 核对行数')


if __name__ == '__main__':
    main()
