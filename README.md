# BuildSentry（工安智审）

智能施工安全合规审核系统 V1.0

## 项目简介

BuildSentry 是一个基于大语言模型的智能施工安全合规审核系统。通过微调 Qwen3-8B 模型，结合 RAG 检索增强技术，自动审核施工方案文档，识别违规项并生成结构化的审核意见书。

### 核心功能

- **文档解析**：支持 PDF/DOCX 格式施工方案上传与文本提取
- **AI 审核**：基于微调大模型的逐段合规性审查
- **RAG 检索**：从规范知识库中检索相关条文作为审核依据
- **结果聚合**：自动生成审核结论（通过/条件通过/不通过）和质量评分
- **模拟推理**：GPU 不可用时自动降级为模拟审核模式
- **用户认证**：JWT 认证，支持多角色权限控制
- **PDF 报告**：一键导出结构化审核报告
- **人工复核**：支持对违规项进行人工确认/驳回

## 技术栈

- **后端**：FastAPI + SQLAlchemy + SQLite
- **前端**：Gradio
- **向量数据库**：ChromaDB
- **模型推理**：transformers + peft (LoRA)

## 环境要求

- Python 3.14+
- CUDA GPU（可选，用于真实模型推理）

## 快速开始

### 1. 克隆项目

```bash
git clone <repository-url>
cd BuildSentry
```

### 2. 安装依赖

```bash
# 使用 uv（推荐）
uv sync

# 或使用 pip
pip install -e .
```

### 3. 配置环境变量

复制 `.env.example` 为 `.env`（如需要），或直接编辑 `.env`：

```bash
# 基座模型名称
BASE_MODEL=Qwen/Qwen3-8B

# LoRA 适配器路径（可选）
LORA_PATH=/path/to/your/lora/adapter

# 设备选择：auto, cuda, cpu
DEVICE=auto

# CORS 允许的源（逗号分隔）
CORS_ORIGINS=http://localhost:7860,http://localhost:3000
```

### 4. 启动服务

```bash
python start.py
```

或分别启动：

```bash
# 启动后端 API
uvicorn app.main:app --host 0.0.0.0 --port 8000

# 启动前端 UI（新终端）
python ui/app.py
```

### 5. 访问系统

- **API 文档**：http://localhost:8000/docs
- **前端界面**：http://localhost:7860

## 项目结构

```
BuildSentry/
├── app/                        # 后端应用
│   ├── api/                    # API 路由层
│   │   ├── documents.py        # 文档上传/查询
│   │   ├── audit.py            # 审核任务管理
│   │   ├── kb.py               # 知识库检索
│   │   └── kb_admin.py         # 知识库管理
│   ├── core/                   # 核心配置
│   │   ├── config.py           # 配置管理
│   │   └── database.py         # 数据库连接
│   ├── models/                 # 数据模型
│   │   └── db_models.py        # ORM 模型
│   ├── repositories/           # 数据访问层
│   ├── schemas/                # Pydantic 模型
│   ├── services/               # 业务逻辑层
│   │   ├── audit_service.py    # 审核编排
│   │   ├── document_service.py # 文档解析
│   │   ├── inference_service.py # 模型推理
│   │   ├── model_loader.py     # 模型加载器
│   │   └── vector_service.py   # 向量检索
│   └── utils/                  # 工具函数
├── ui/                         # Gradio 前端
│   └── app.py
├── data/                       # 运行时数据（gitignored）
├── docs/                       # 项目文档
├── .env                        # 环境变量
├── pyproject.toml              # 项目配置
└── start.py                    # 启动脚本
```

## API 接口

### 文档管理

- `POST /api/v1/documents/upload` - 上传施工方案文档
- `GET /api/v1/documents/{doc_id}` - 获取文档详情

### 审核任务

- `POST /api/v1/audit/tasks` - 创建审核任务
- `GET /api/v1/audit/tasks/{task_id}` - 查询任务状态
- `GET /api/v1/audit/tasks/{task_id}/result` - 获取审核结果

### 知识库

- `GET /api/v1/kb/search` - 检索规范条文
- `POST /api/v1/kb/admin/upload-spec` - 上传规范文档

## 配置说明

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `BASE_MODEL` | `Qwen/Qwen3-8B` | 基座模型名称 |
| `LORA_PATH` | `None` | LoRA 适配器路径 |
| `DEVICE` | `auto` | 设备选择（auto/cuda/cpu） |
| `DATABASE_URL` | `sqlite:///data/buildsentry.db` | 数据库连接字符串 |
| `CHROMA_COLLECTION_NAME` | `clauses` | 向量数据库集合名 |
| `CORS_ORIGINS` | `http://localhost:7860` | CORS 允许的源 |
| `SIMULATION_DELAY` | `0.5` | 模拟推理延迟（秒） |

## 开发说明

### 模拟模式

当 GPU 不可用时，系统自动使用模拟审核模式：
- 基于关键词匹配生成模拟违规项
- 使用规范知识库中的真实条文
- 返回格式与真实模式一致

### 数据库

项目默认使用 SQLite，数据存储在 `data/buildsentry.db`。

如需使用 PostgreSQL，修改 `.env`：

```bash
DATABASE_URL=postgresql://user:password@localhost:5432/buildsentry
```

## 文档

详细设计文档位于 `docs/` 目录：

- [需求设计书](docs/需求设计书.md)
- [系统设计](docs/系统设计.md)
- [接口设计](docs/接口设计.md)
- [数据库设计](docs/数据库设计.md)

## 许可证

私有项目，未经授权禁止使用。
