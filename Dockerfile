# ============================================
# BuildSentry Dockerfile - 多阶段构建
# ============================================

# ---- Stage 1: 构建阶段 ----
FROM python:3.14-slim AS builder

WORKDIR /app

# 安装构建依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 安装 uv
RUN pip install --no-cache-dir uv

# 复制依赖文件
COPY pyproject.toml uv.lock ./

# 安装 Python 依赖
RUN uv sync --frozen --no-dev

# ---- Stage 2: 运行阶段 ----
FROM python:3.14-slim AS runtime

# 创建非 root 用户
RUN groupadd -r buildsentry && useradd -r -g buildsentry -d /app -s /sbin/nologin buildsentry

WORKDIR /app

# 安装运行时依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 从构建阶段复制依赖
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/uv.lock /app/uv.lock

# 复制应用代码
COPY . .

# 创建数据目录并设置权限
RUN mkdir -p data/reports data/chroma_data data/chroma_cache \
    && chown -R buildsentry:buildsentry /app

# 切换到非 root 用户
USER buildsentry

# 设置环境变量
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/system/health || exit 1

# 启动命令
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
