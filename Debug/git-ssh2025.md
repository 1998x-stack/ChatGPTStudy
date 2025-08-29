# GitHub SSH 常见问题一锅端：端口 22 超时、走到 127.0.0.1:7890、Host key 报错、无法 push

## TL;DR（最快路径）

* **这是正常现象**：`Hi xxx! ... GitHub does not provide shell access` 并非报错，只是说明 SSH 成功但 GitHub 不提供交互式 shell。
* **22 被拦** 就用 **443**：把远程改为 `ssh://git@ssh.github.com:443/USER/REPO.git` 或 `github-443:USER/REPO.git`（见下文配置）。
* **被代理劫持到 127.0.0.1:7890？** 在 `~/.ssh/config` 为 GitHub 增加 `ProxyCommand none`，或使用 443 专用别名并禁代理。
* **Host key 报错**：用 `ssh-keyscan -p 443 ssh.github.com >> ~/.ssh/known_hosts` 预写入，再连。

---

## 1. 先判断现状：你的 SSH 其实已经可用了吗？

运行测试：

```bash
ssh -T git@github.com
```

如果看到：

```
Hi <YOUR_NAME>! You've successfully authenticated, but GitHub does not provide shell access.
```

说明 **认证成功** ✅，这不是错误。继续做 Git 操作即可（clone/pull/push）。

进一步验证 Git 协议（无需 clone）：

```bash
git ls-remote git@github.com:USER/REPO.git HEAD
```

能返回一个 commit hash 就 OK。

---

## 2. 22 端口被拦怎么办？——改走 443 端口（强烈推荐）

### 2.1 临时一次性克隆（无需改配置）

```bash
# 预写入 443 的主机钥（只需一次）
ssh-keygen -R "[ssh.github.com]:443" 2>/dev/null
ssh-keyscan -p 443 ssh.github.com >> ~/.ssh/known_hosts

# 一次性命令：强制走 443 并禁用代理
GIT_SSH_COMMAND='ssh -p 443 -o ProxyCommand=none -o IdentitiesOnly=yes' \
git clone ssh://git@ssh.github.com:443/USER/REPO.git
```

### 2.2 永久方案：在 \~/.ssh/config 加一个 443 专用别名

```sshconfig
# 常规 22（可选使用代理，按需）
Host github.com
  HostName github.com
  User git
  IdentityFile ~/.ssh/id_ed25519
  IdentitiesOnly yes
  AddKeysToAgent yes
  UseKeychain yes
  # 如果你之前写过走本地代理的行，建议先注释：
  # ProxyCommand nc -x 127.0.0.1:7890 %h %p

# 专用于 443 的别名（显式禁用代理）
Host github-443
  HostName ssh.github.com
  User git
  Port 443
  IdentityFile ~/.ssh/id_ed25519
  IdentitiesOnly yes
  AddKeysToAgent yes
  UseKeychain yes
  ProxyCommand none
```

使用：

```bash
# 首次接受主机钥（若未写入）
ssh -T git@github-443

# 克隆
git clone github-443:USER/REPO.git
# 也可用长 URL
# git clone ssh://git@ssh.github.com:443/USER/REPO.git
```

---

## 3. 为什么会出现 127.0.0.1:7890？如何查清“来源”并拦住

许多人看到 `Authenticated to github.com ([127.0.0.1]:7890)` 会以为是 SSH 配置，其实**常是系统或第三方工具的透明代理**（Clash/ClashX/Surge/sing-box/Proxifier 等）把 TCP 流量劫持了。

### 3.1 先看 SSH 的最终生效配置（是否自己写了 ProxyCommand）

```bash
ssh -G github.com | sed -n '1,200p' | egrep -i 'proxy|jump|identity|host|port'
```

* 若 **没有** `proxycommand`/`proxyjump`，说明 **SSH 干净**，7890 来自系统/第三方。
* 可检索配置文件确认：

```bash
grep -nE "Proxy(Command|Jump)|7890" ~/.ssh/config /etc/ssh/ssh_config 2>/dev/null
```

### 3.2 谁在监听 7890（确定是哪个代理程序）

```bash
lsof -iTCP:7890 -sTCP:LISTEN -nP
ps -ax | egrep -i 'clash|proxifier|surge|sing|xray|v2ray|privoxy|leaf' | grep -v egrep
```

### 3.3 系统代理/TUN 模式检查（macOS）

```bash
scutil --proxy
networksetup -listallnetworkservices
networksetup -getwebproxy "Wi-Fi"
networksetup -getsocksfirewallproxy "Wi-Fi"
ifconfig | egrep -i 'utun|tap|tun'   # 若有 utun 多半开了 TUN/Enhanced
```

### 3.4 解决建议

* 在代理 App（ClashX/Surge/Proxifier 等）里 **关闭 TUN/增强模式**，或把 **Terminal/ssh** 加入「直连/排除」规则。
* 在 SSH 层为 `github-443` 加 `ProxyCommand none`（如上配置），确保 **GitHub SSH 不走代理**。

---

## 4. Host key verification failed / 首次连 443 不认主机钥

**症状：**

```
The authenticity of host '[ssh.github.com]:443 (...)' can't be established.
Host key verification failed.
```

**解决：**

```bash
ssh-keygen -R "[ssh.github.com]:443" 2>/dev/null
ssh-keyscan -p 443 ssh.github.com >> ~/.ssh/known_hosts
chmod 700 ~/.ssh
chmod 600 ~/.ssh/known_hosts

# 再连
ssh -o "ProxyCommand=none" -p 443 git@ssh.github.com -T
```

首连时输入 `yes` 接受主机钥即可。

---

## 5. Push 手把手（含切换远程 URL）

### 5.1 设置/切换远程为 443 别名

```bash
git remote -v
git remote set-url origin github-443:USER/REPO.git
# 或
# git remote set-url origin ssh://git@ssh.github.com:443/USER/REPO.git
```

### 5.2 正常推送

```bash
git add .
git commit -m "your message"
git push origin main    # 或 master
# 新分支：
# git checkout -b feature-x
# git push -u origin feature-x
```

---

## 6. 常见错误对照表

| 报错/现象                                                                | 根因                     | 处理                                                                                                          |
| -------------------------------------------------------------------- | ---------------------- | ----------------------------------------------------------------------------------------------------------- |
| `Hi ... but GitHub does not provide shell access.` + `Exit status 1` | 正常提示（GitHub 不提供 shell） | 无需处理，Git 操作可用                                                                                               |
| `ssh: connect to host github.com port 22: Operation timed out`       | 网络拦截 22 端口             | 改走 443（见第 2 节）                                                                                              |
| `Permission denied (publickey)`                                      | 公钥未加入账号/用错私钥           | `cat ~/.ssh/id_ed25519.pub` 加到 GitHub → Settings → SSH keys；`IdentitiesOnly yes`；`ssh -T git@github-443` 验证 |
| `Repository not found` / `access denied`                             | URL 写错/无权限/大小写错误       | 用 `git ls-remote` 先测；确认 `git@github.com:USER/REPO.git` 或 443 版本；检查你是否有私库权限                                  |
| `Too many authentication failures`                                   | 太多密钥尝试                 | `ssh -o IdentitiesOnly=yes -i ~/.ssh/id_ed25519 -T git@github-443`，或在 config 里 `IdentitiesOnly yes`         |
| `Host key verification failed`                                       | 未信任主机钥                 | 用 `ssh-keyscan -p 443 ssh.github.com >> ~/.ssh/known_hosts` 预写入                                             |
| 仍看到 `127.0.0.1:7890`                                                 | 透明代理/TUN 劫持            | 关闭 TUN/把 ssh 设为直连；或命令行临时 `-o ProxyCommand=none`                                                             |

---

## 7. 诊断命令小抄（复制即用）

```bash
# 查看 SSH 最终生效配置（是否走代理）
ssh -G github.com | sed -n '1,200p' | egrep -i 'proxy|jump|identity|host|port'

# 查配置文件里是否写了 ProxyCommand/7890
grep -nE "Proxy(Command|Jump)|7890" ~/.ssh/config /etc/ssh/ssh_config 2>/dev/null

# 哪个程序在监听 7890（代理是谁）
lsof -iTCP:7890 -sTCP:LISTEN -nP

# 是否开了系统级代理/TUN
scutil --proxy
ifconfig | egrep -i 'utun|tap|tun'

# 远端连通性
nc -vz github.com 22
nc -vz ssh.github.com 443

# 详细调试 SSH（禁代理，走 443）
ssh -vvv -o ProxyCommand=none -p 443 git@ssh.github.com -T
```

---

## 8. 备用方案：HTTPS（对公开仓库或已配置凭据的私库）

```bash
git clone https://github.com/USER/REPO.git
# 如要全局把 https 自动替换为 ssh（非必须）
git config --global url."git@github.com:".insteadOf "https://github.com/"
```

---

### 结语

* **优先建议**：为 GitHub 配置 **443 专用别名**并显式 **禁用代理**；这样在公司网/校园网/机场代理/TUN 模式下，都能稳定使用 SSH。
* **碰到问题**，先跑「诊断命令小抄」，基本能立刻定位是 **端口被拦**、**代理劫持** 还是 **主机钥/权限** 问题。