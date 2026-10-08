# ServeMind · 智应客服

## 用途

ServeMind 是面向多领域智能客服的多 Agent 项目，用于客户咨询、业务信息核验、知识库问答和人工协作。目前以电商场景为示例，支持商品、价格、库存、配送、售后及商家接管；其他领域可接入自己的角色、工具和知识库。

## 基本技术

| 部分 | 技术 |
| --- | --- |
| 客服界面 | Vue 3、Vite，Tauri 桌面壳 |
| 后端服务 | Python 3.12、FastAPI |
| Agent | DeepSeek、自研主／辅助 Agent 路由与并行执行、MCP 工具调用 |
| 知识检索 | 关键词检索；可选 Qwen Embedding/Reranker 与 pgvector |
| 数据与记忆 | PostgreSQL、Redis；SQLite 用于本地商业演示 |

## Agent 处理流程

1. 读取当前会话上下文，识别用户意图和多个诉求。
2. 按领域评分选择主 Agent 与辅助 Agent。
3. 各 Agent 并行处理，在权限范围内查询业务工具或知识库。
4. 汇总查询证据，生成并校验回答；信息不足时澄清或安全回退。
5. 需要人工处理时，整理已核验信息和待处理事项，交给对应商家继续沟通。

## 快速启动

以下步骤用于全新克隆的本地开发环境：商业数据使用 SQLite，会话记忆使用 PostgreSQL，知识检索使用关键词模式。填写模型密钥后可体验模型 Agent；留空时使用规则回退。

### 1. 准备环境与数据

需要 Python 3.12+、Node.js 22.12+，以及已启动的 PostgreSQL 16（安装 pgvector）和 Redis 7。PostgreSQL 账号需有创建表及启用 `vector` 扩展的权限。

```sh
git clone https://github.com/FreeFoots/ServeMind.git
cd ServeMind
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e 'backend[mcp]'
mkdir -p data/MSOM_Data_Driven_Challenge_2020 backend/runtime
```

将自行准备且获授权的 MSOM CSV 放入 `data/MSOM_Data_Driven_Challenge_2020/`。启动至少需要 `JD_order_data.csv` 和 `JD_delivery_data.csv`；商品目录体验还需 `JD_sku_data.csv`，其余数据文件按需补充。仓库不包含原始数据、模型权重或私密配置。

### 2. 配置后端

为新环境创建数据库；已有数据库可直接使用自己的连接地址：

```sh
createdb servemind
```

在本机创建 `backend/.env`，填入自己的连接地址和模型密钥：

```dotenv
SERVEMIND_ENV=development
SERVEMIND_DEMO_ACCOUNT_SWITCH=true
SERVEMIND_DATA_BACKEND=csv
SERVEMIND_COMMERCE_BACKEND=sqlite
SERVEMIND_DATABASE_URL=postgresql:///servemind
SERVEMIND_REDIS_URL=redis://127.0.0.1:6379/0
SERVEMIND_RAG_BACKEND=keyword
SERVEMIND_SEMANTIC_INTENT=false
SERVEMIND_SEMANTIC_MEMORY=false
SERVEMIND_AGENT_MODE=model
DEEPSEEK_API_KEY=
```

演示账户选择仅用于本机开发。`.env` 已被 Git 忽略，不要提交模型密钥或数据库凭证。

### 3. 启动后端

在仓库根目录、已激活虚拟环境的终端运行：

```sh
PYTHONPATH=backend/src python -m uvicorn servemind.api.main:app --host 127.0.0.1 --port 8317
```

后端会初始化所需的记忆表和本地演示账户。健康接口：<http://127.0.0.1:8317/v1/health>。

### 4. 启动前端

新开终端，从仓库根目录运行：

```sh
cd frontend
npm ci
npm run dev -- --host 127.0.0.1
```

打开 <http://127.0.0.1:5317>，选择买家或商家演示账户体验咨询与交接。前端默认连接本机 API，可通过 `VITE_SERVEMIND_API_BASE_URL` 修改地址。

完整向量检索需另行安装 `backend[rag]`、准备 Qwen 模型缓存并运行 `backend/scripts/index_knowledge.py`；商业 PostgreSQL 模式需先完成 `backend/scripts/migrate_commerce.py` 的数据迁移。
