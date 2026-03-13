# RD-Agent 本地 UI 使用说明

当前本机已经按官方方式部署 `RD-Agent` 页面：

- 官方命令：`rdagent ui --port 19899 --log-dir /root/Ai-polyTest/RD-Agent/log`
- 当前服务名：`rdagent-ui.service`
- 当前访问地址：`http://127.0.0.1:19899`

## GitHub 同步

当前本地改动已经切到你的 fork 工作流：

- Fork：`https://github.com/BnFeng/RD-Agent`
- 开发分支：`local-prediction-market`

日常同步建议使用：

```bash
cd /root/Ai-polyTest/RD-Agent
./scripts/git_sync.sh "feat: 你的改动说明"
```

更完整的远端与分支说明见：

- `GITHUB_SYNC.md`

## 管理命令

安装并启动：

```bash
cd /root/Ai-polyTest/RD-Agent
./scripts/install_ui_service.sh 19899 /root/Ai-polyTest/RD-Agent/log
```

查看状态：

```bash
./scripts/status_ui_service.sh
systemctl status rdagent-ui.service --no-pager
```

健康探测：

```bash
curl -fsS http://127.0.0.1:19899 | head
```

重启 / 停止：

```bash
./scripts/restart_ui_service.sh
./scripts/stop_ui_service.sh
```

## 说明

当前服务使用的仍然是官方入口 `rdagent ui`，只是通过 `systemd` 常驻化。

由于 RD-Agent 的 `ui` 命令会再启动 `streamlit` 子进程，所以服务模板里补充了 `PATH` 环境，确保 `streamlit` 可被找到。

如果你后续已经生成了 RD-Agent 的实验日志，可以直接在页面里手动指定日志路径；当前不强制给 `--log-dir`，是为了避免空日志目录导致页面初始化失败。
