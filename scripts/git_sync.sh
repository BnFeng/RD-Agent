#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

message="${*:-chore: sync $(date -u +'%Y-%m-%d %H:%M:%S UTC')}"

git add -A

if git diff --cached --quiet; then
  echo "没有可提交的改动"
  exit 0
fi

git commit -m "${message}"
echo "提交完成。若当前分支已绑定 origin，对应改动会自动推送。"
