#!/usr/bin/env bash
# 把 global/ 里的全局守则与全局 skill 链接到 Claude Code 与 Codex 的用户级位置(docs/12-范式库维护.md 第五节)
#
# 可重复运行;换机器后跑一次。已是正确链接的不动;位置上是别的链接就改指向;
# 位置上是普通文件或目录就先改名备份(<原名>.bak-<时间>)再链接,不删任何东西。
set -euo pipefail

SRC="$(cd "$(dirname "$0")/../global" && pwd -P)"
LINK_ROOT="${ORRERY_LINK_ROOT:-$HOME}"
MODE=install
case "${1:-}" in
  --check) MODE=check ;;
  '') ;;
  *) echo '用法: bash scripts/link-global.sh [--check]' >&2; exit 2 ;;
esac
if [ "$#" -gt 1 ]; then echo '参数过多' >&2; exit 2; fi
failures=0

link() { # link <目标> <链接位置>
  local target="$1" at="$2"
  if [ "$MODE" = check ]; then
    if [ -e "$target" ] && [ -L "$at" ] && [ "$(readlink "$at")" = "$target" ] && [ -e "$at" ]; then
      echo "已就位 $at"
    else
      echo "未就位 $at -> $target" >&2
      failures=$((failures + 1))
    fi
    return
  fi
  mkdir -p "$(dirname "$at")"
  if [ -L "$at" ]; then
    if [ "$(readlink "$at")" = "$target" ]; then echo "已就位 $at"; return; fi
    ln -sfn "$target" "$at"; echo "改指向 $at -> $target"; return
  fi
  if [ -e "$at" ]; then
    local bak; bak="$at.bak-$(date +%Y%m%d-%H%M%S)"
    mv "$at" "$bak"; echo "已备份 $at -> $bak"
  fi
  ln -s "$target" "$at"; echo "新建   $at -> $target"
}

link "$SRC/AGENTS.md" "$LINK_ROOT/.claude/CLAUDE.md"
link "$SRC/AGENTS.md" "$LINK_ROOT/.codex/AGENTS.md"
for dir in "$SRC"/skills/*/; do
  name="$(basename "$dir")"
  link "${dir%/}" "$LINK_ROOT/.claude/skills/$name"
  link "${dir%/}" "$LINK_ROOT/.agents/skills/$name"
done
if [ "$failures" -gt 0 ]; then exit 1; fi
