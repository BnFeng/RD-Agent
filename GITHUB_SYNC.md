# RD-Agent GitHub 同步说明

当前本地仓库已经完成 GitHub 绑定：

- 你的 Fork：`https://github.com/BnFeng/RD-Agent`
- 上游仓库：`https://github.com/microsoft/RD-Agent`

## 当前远端结构

```bash
origin   -> https://github.com/BnFeng/RD-Agent.git
upstream -> https://github.com/microsoft/RD-Agent.git
```

## 当前分支约定

- `main`
  - 仅用于跟踪 `upstream/main`
  - 不建议直接在这个分支开发

- `local-prediction-market`
  - 当前本地预测市场集成开发分支
  - 已绑定到：`origin/local-prediction-market`

## 日常提交流程

推荐直接使用同步脚本：

```bash
cd /root/Ai-polyTest/RD-Agent
./scripts/git_sync.sh "feat: 你的改动说明"
```

脚本会自动：

1. `git add -A`
2. 创建提交
3. 触发本地 `post-commit` 钩子
4. 自动推送到当前分支对应的 `origin`

## 查看当前状态

```bash
git remote -v
git branch -vv
git status
```

## 同步上游主线

如果后续需要把微软官方最新代码同步到本地：

```bash
cd /root/Ai-polyTest/RD-Agent
git checkout main
git fetch upstream
git reset --hard upstream/main
git checkout local-prediction-market
git merge main
```

如果合并冲突较多，建议不要直接硬合，先单独评估变更范围。
