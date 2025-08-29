# Web3 常用包

**User:** Anonymous (kanekoveblen824@outlook.com)  
**Created:** 8/29/2025 13:58  
**Updated:** 8/29/2025 14:00  
**Exported:** 8/29/2025 14:01  
**Link:** [https://chatgpt.com/c/68b1416a-d5f4-8329-a0c3-fa0ee2e79f32](https://chatgpt.com/c/68b1416a-d5f4-8329-a0c3-fa0ee2e79f32)  

## Prompt:
go 关于 web3 最常用的包！

## Response:
明白 👍 你想要一个 \*\*Web3 最常用的包（libraries / frameworks）\*\*清单。这里我帮你分成几类：

* * *

🛠 基础核心库
--------

这些是和区块链交互最常用的：

*   **[web3.js](https://github.com/web3/web3.js?utm_source=chatgpt.com)**：最经典的以太坊 JavaScript SDK，支持合约调用、交易发送、账户管理。
*   **ethers.js**：轻量、现代的以太坊 SDK，TypeScript 友好，社区现在更推荐用它替代 web3.js。
*   **web3.py**：Python 版本的 Web3 库，科研/后端脚本里很常用。
*   **viem**：新一代 TypeScript Web3 库，API 设计更现代，和 wagmi 配合常见。

* * *

💳 钱包交互相关
---------

*   **wagmi**：React hooks 库，专门和 EVM 钱包交互，配合 ethers.js/viem 使用。
*   **RainbowKit**：钱包连接 UI 组件，用户体验好，常和 wagmi 搭配。
*   **WalletConnect**：通用的钱包连接协议，支持手机钱包扫码连接。

* * *

🔧 智能合约开发 & 测试框架
----------------

*   **Hardhat**：主流智能合约开发框架，插件生态丰富。
*   **Truffle**：早期流行的开发框架，现在 Hardhat 更受欢迎。
*   **Foundry**：Rust 编写的超快合约开发工具链，测试/部署效率很高。
*   **Brownie**：Python 开发的合约框架，适合和 web3.py 搭配。

* * *

📊 数据索引 & API
-------------

*   **The Graph**：去中心化数据索引协议，用 subgraph 解析链上数据。
*   **Alchemy SDK**、**Infura**：区块链节点 API 提供商。
*   **Moralis**：全套 Web3 后端服务，提供 NFT、交易、用户登录等 API。

* * *

🖼 NFT / DeFi 常用
----------------

*   **OpenZeppelin Contracts**：标准的 ERC-20/ERC-721/ERC-1155 合约库，安全可靠。
*   **thirdweb**：提供开箱即用的 SDK + Dashboard，方便快速构建 NFT/DeFi 应用。
*   **[Uniswap SDK](https://github.com/Uniswap/sdk-core?utm_source=chatgpt.com)**：用于 DeFi 交易 / AMM 的 SDK。

* * *

⚡ Layer2 & 跨链
-------------

*   **ethers + zkSync SDK**：Layer2 交互。
*   **[cosmjs](https://github.com/cosmos/cosmjs?utm_source=chatgpt.com)**：Cosmos 链生态的 JS SDK。
*   **Polkadot.js**：Polkadot & Substrate 链的 SDK。

* * *

👉 总结一下：

*   **DApp 前端**：wagmi + viem/ethers.js + RainbowKit
*   **合约开发**：Hardhat / Foundry + OpenZeppelin
*   **数据层**：The Graph + Alchemy/Infura
*   **NFT/DeFi**：OpenZeppelin + thirdweb + Uniswap SDK

* * *

要不要我帮你整理成一份 **「Web3 常用包速查表（Cheat Sheet）」**，每个包写一句“使用场景 + 优缺点”，方便你快速对比？

## Prompt:
go

## Response:
好嘞！给你一份**Web3 常用包速查表（Cheat Sheet）**——按场景分组，每个条目都有「用途 | 何时用 | 优点 | 注意点 | 安装/一行上手」👇

🛠 基础交互（EVM）
============

| 包 | 用途 | 何时用 | 优点 | 注意点 | 安装 / 一行上手 |
| --- | --- | --- | --- | --- | --- |
| **ethers.js** | 调用合约、发交易、签名 | 前端/Node 通用首选 | TS 友好、文档好、体积小 | v6 API 与 v5 有差异 | `npm i ethers` · `new ethers.JsonRpcProvider(URL)` |
| **viem** | 现代化 TS Web3 客户端 | React + wagmi 推荐 | API 纯函数、类型极强 | 生态较新 | `npm i viem` · `createPublicClient({ transport: http(URL) })` |
| **web3.js** | 经典以太坊 SDK | 维护旧项目 | 生态老牌 | API 偏旧 | `npm i web3` · `new Web3(URL)` |
| **web3.py** | Python 调链 | 数据脚本/科研 | Python 生态强 | 需要 Python 环境 | `pip install web3` |

💳 钱包连接 / 前端集成
==============

| 包 | 用途 | 何时用 | 优点 | 注意点 | 安装 / 一行上手 |
| --- | --- | --- | --- | --- | --- |
| **wagmi** | React hooks 调链 | React DApp | 与 viem 深度集成 | 仅 React | `npm i wagmi viem` |
| **RainbowKit** | 钱包连接 UI | 快速做钱包弹窗 | 设计好看 | 依赖 wagmi | `npm i @rainbow-me/rainbowkit` |
| **WalletConnect** | 多钱包连接协议 | 手机钱包扫码 | 钱包覆盖广 | 需注册项目ID | `npm i @walletconnect/ethereum-provider` |

🔧 合约开发 / 测试 / 部署
=================

| 包 | 用途 | 何时用 | 优点 | 注意点 |
| --- | --- | --- | --- | --- |
| **Hardhat** | 编译、测试、脚本、插件 | JS/TS 团队标配 | 插件丰富（ethers、foundryup 等） | Node 项目依赖较多 |
| **Foundry** | 超快测试/模糊测试 | 追求速度/覆盖 | Rust 实现、cheatcodes 强 | 需安装 `foundryup` |
| **OpenZeppelin Contracts** | 标准合约库 | ERC20/721/1155 | 安全、审计友好 | 注意选对版本（v5 与 v4 区别） |
| **Brownie** | Python 合约框架 | Python 团队 | 与 web3.py 顺手 | 依赖较多 |

常用安装：

```bash
npm i --save-dev hardhat @nomicfoundation/hardhat-toolbox
npm i @openzeppelin/contracts
# Foundry: curl -L https://foundry.paradigm.xyz | bash && foundryup
```

📊 数据层 / 节点服务
=============

| 服务/包 | 用途 | 何时用 | 优点 | 注意点 |
| --- | --- | --- | --- | --- |
| **The Graph** | 子图索引链上数据 | 复杂查询 | GraphQL 强大 | 需写 subgraph 映射 |
| **Alchemy SDK** | 节点 + 增值 API | 快速稳定 | 文档好、NFT/Tx API | 需注册 |
| **Infura** | 节点服务 | 经典选择 | 多链支持 | 免费配额限制 |
| **Moralis** | 全家桶后端 | 要省时全栈 | 用户/NFT/通知 | 商业化依赖 |

🖼 NFT / DeFi
=============

| 包 | 用途 | 何时用 | 优点 | 注意点 |
| --- | --- | --- | --- | --- |
| **OpenZeppelin Contracts** | 标准 NFT/代币 | 发行/权限/可升级 | 模块化、社区成熟 | 遵循安全模式 |
| **thirdweb** | 快速搭建+仪表盘 | MVP/黑客松 | SDK + 合约预置 | 框架绑定 |
| **Uniswap SDK (v3/core)** | AMM 报价/路由 | 交易/聚合 | 行业标准 | 需理解池/费级 |

⚡ 多链 / 非 EVM
============

| 包 | 用途 | 生态 | 安装 |
| --- | --- | --- | --- |
| **Polkadot.js** | 连接 Substrate/Polkadot | 波卡 | `npm i @polkadot/api` |
| **cosmjs** | Cosmos SDK 链 | Cosmos | `npm i @cosmjs/stargate` |
| **ethers + zkSync SDK** | zkSync Era 交互 | L2 | 参考官方文档 |

🔐 安全 / 审计辅助
============

| 包 | 用途 | 备注 |
| --- | --- | --- |
| **slither** | 静态分析 | `pip install slither-analyzer` |
| **mythril** | 符号执行 | 漏洞检测 |
| **ethlint (Solhint)** | 合约风格/规则 | CI 里跑 |

🧰 辅助工具（前后端）
============

| 包 | 用途 | 备注 |
| --- | --- | --- |
| **TypeChain** | 从 ABI 生成 TS 类型 | `npm i typechain --save-dev` |
| **ABIs via `@openzeppelin/contracts`** | 直接复用标准 ABI | 版本匹配很重要 |
| **dotenv** | 私钥/URL 管理 | 别把私钥进仓库！ |

* * *

🚀「三行起步」模板
----------

**前端（React + wagmi + viem + RainbowKit）**

```bash
npm i react react-dom wagmi viem @rainbow-me/rainbowkit
```

```ts
// main.tsx
const { chains, config } = getDefaultConfig({ appName:'DApp', projectId:'WC_ID', chains:[mainnet, sepolia] });
createRoot(el).render(<WagmiConfig config={config}><RainbowKitProvider><App/></RainbowKitProvider></WagmiConfig>);
```

```ts
// App.tsx
const { address, isConnected } = useAccount();
const { connect } = useConnect(); const { disconnect } = useDisconnect();
```

**Hardhat + ethers 部署最小脚本**

```bash
npm i -D hardhat @nomicfoundation/hardhat-toolbox ethers
npx hardhat init
```

```ts
// scripts/deploy.ts
const factory = await ethers.getContractFactory("MyToken");
const token = await factory.deploy(); await token.waitForDeployment();
console.log("Deployed:", await token.getAddress());
```

**Python 快速读链（web3.py）**

```bash
pip install web3
```

```py
from web3 import Web3
w3 = Web3(Web3.HTTPProvider("https://eth-mainnet.g.alchemy.com/v2/KEY"))
print(w3.eth.block_number)
```

* * *

🧭 选型决策速览
---------

*   **前端交互**：`wagmi + viem + RainbowKit`（需要 React）；非 React 用 `ethers.js`/`viem` 直连。
*   **合约开发**：全 JS/TS 团队选 **Hardhat**；追求速度和测试体验选 **Foundry**。
*   **标准合约**：一律 **OpenZeppelin**（留意主次版本差异）。
*   **数据查询**：简单读链 → 节点（Alchemy/Infura）；复杂关系/过滤 → **The Graph**。
*   **快速 MVP**：**thirdweb** 全家桶直接开干。

* * *

如果你告诉我**你的技术栈（React/Vue/纯后端/Python）、目标（NFT、DeFi、GameFi、数据分析）、是否要多链支持**，我可以基于这张表帮你出一套**最小可用模板 + 依赖清单 + 目录结构**，直接开工。



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)