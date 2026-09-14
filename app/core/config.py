# app/core/config.py
"""
配置管理模块。
统一管理所有配置项，支持从环境变量读取。
"""

import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path

# 尝试加载 .env 文件
try:
    from dotenv import load_dotenv

    # 加载项目根目录的 .env 文件
    _env_path = Path(__file__).parent.parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    pass  # python-dotenv 未安装，跳过


@dataclass
class Settings:
    """应用配置类"""

    # === 项目路径 ===
    PROJECT_ROOT: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent)
    DATA_DIR: Path = field(default=None)

    # === 运行环境 ===
    ENVIRONMENT: str = "development"  # development, testing, production

    # === 安全配置 ===
    SECRET_KEY: str = ""  # JWT 密钥，必须从环境变量读取
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24小时
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    # 初始管理员密码。未设置时：开发环境用 admin123，生产环境随机生成并打印一次。
    DEFAULT_ADMIN_PASSWORD: str | None = None

    # === 速率限制 ===
    RATE_LIMIT_PER_MINUTE: int = 60
    RATE_LIMIT_PER_HOUR: int = 1000
    LOGIN_RATE_LIMIT_PER_MINUTE: int = 5

    # === 上传限制 ===
    # 施工方案 PDF 通常几 MB，50MB 足够覆盖扫描件；此前无任何限制，
    # 上传会被整个读进内存。
    MAX_UPLOAD_SIZE_MB: int = 50

    # === 模型配置 ===
    # Qwen3-8B：同一权重通过 enable_thinking 切换思考/非思考模式，
    # 本任务不需要思维链，推理侧会显式关闭（见 audit_contract.encode_for_model）。
    BASE_MODEL: str = "Qwen/Qwen3-8B"
    LORA_PATH: str | None = None
    DEVICE: str = "auto"  # auto, cuda, cpu
    # 模型加载到哪张 GPU：
    #   - "auto"（默认）：自动挑选，见下面两项
    #   - 整数（如 "5"）：固定使用该卡
    MODEL_GPU_INDEX: str = "auto"
    # auto 模式下排除的卡号（逗号分隔）。例如 "0,1" 表示不使用 0 号和 1 号卡。
    MODEL_GPU_EXCLUDE: str = ""
    # auto 模式下要求的最小空闲显存（MiB），低于此值的卡会被跳过。
    # 7B fp16 权重约 14.5GiB，加 CUDA 上下文与 KV cache 约需 16-18GiB，
    # 因此 18000 是实际下限。调高可更保守，但可能无卡可用而回退模拟模式。
    MODEL_GPU_MIN_FREE_MB: int = 18000

    # === 数据库配置 ===
    DATABASE_URL: str = ""
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20

    # === 向量数据库配置 ===
    CHROMA_COLLECTION_NAME: str = "clauses"
    CHROMA_DATA_DIR: Path = field(default=None)
    CHROMA_CACHE_DIR: Path = field(default=None)

    # === API配置 ===
    API_V1_PREFIX: str = "/api/v1"
    CORS_ORIGINS: list[str] = field(default_factory=lambda: ["http://localhost:7860"])
    API_TITLE: str = "BuildSentry API"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = "智能施工安全合规审核系统"

    # === 推理配置 ===
    SIMULATION_DELAY: float = 0.5
    AUDIT_KEYWORDS: list[str] = field(
        default_factory=lambda: [
            "基坑",
            "堆载",
            "放坡",
            "脚手架",
            "模板",
            "钢筋",
            "混凝土",
            "降水",
            "支护",
        ]
    )

    # === 日志配置 ===
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"  # json 或 text
    LOG_FILE: str | None = None

    # === 重试配置 ===
    MAX_RETRIES: int = 3
    RETRY_DELAY: float = 1.0
    RETRY_BACKOFF: float = 2.0

    def __post_init__(self):
        """初始化后处理，从环境变量读取配置"""
        # 设置项目根目录
        if self.DATA_DIR is None:
            self.DATA_DIR = self.PROJECT_ROOT / "data"
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)

        # 运行环境
        self.ENVIRONMENT = os.getenv("ENVIRONMENT", self.ENVIRONMENT)

        # 安全配置 - 必须从环境变量读取
        self.SECRET_KEY = os.getenv("SECRET_KEY", "")
        if not self.SECRET_KEY:
            # 开发环境自动生成，生产环境必须设置
            if self.is_production():
                raise ValueError("SECRET_KEY 环境变量必须在生产环境中设置")
            self.SECRET_KEY = secrets.token_hex(32)
            print("[WARNING] 使用自动生成的 SECRET_KEY，生产环境请设置环境变量")

        self.DEFAULT_ADMIN_PASSWORD = os.getenv("DEFAULT_ADMIN_PASSWORD") or None

        # 速率限制（此前未从环境变量读取，.env 中的配置会被忽略）
        self.RATE_LIMIT_PER_MINUTE = int(
            os.getenv("RATE_LIMIT_PER_MINUTE", self.RATE_LIMIT_PER_MINUTE)
        )
        self.RATE_LIMIT_PER_HOUR = int(os.getenv("RATE_LIMIT_PER_HOUR", self.RATE_LIMIT_PER_HOUR))
        self.LOGIN_RATE_LIMIT_PER_MINUTE = int(
            os.getenv("LOGIN_RATE_LIMIT_PER_MINUTE", self.LOGIN_RATE_LIMIT_PER_MINUTE)
        )

        # 上传限制
        self.MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", self.MAX_UPLOAD_SIZE_MB))

        # 从环境变量读取配置
        self.BASE_MODEL = os.getenv("BASE_MODEL", self.BASE_MODEL)
        self.LORA_PATH = os.getenv("LORA_PATH", self.LORA_PATH)
        self.DEVICE = os.getenv("DEVICE", self.DEVICE)
        self.MODEL_GPU_INDEX = os.getenv("MODEL_GPU_INDEX", self.MODEL_GPU_INDEX)
        self.MODEL_GPU_EXCLUDE = os.getenv("MODEL_GPU_EXCLUDE", self.MODEL_GPU_EXCLUDE)
        self.MODEL_GPU_MIN_FREE_MB = int(
            os.getenv("MODEL_GPU_MIN_FREE_MB", self.MODEL_GPU_MIN_FREE_MB)
        )

        # 数据库配置（此前未读取环境变量，README 中切换 PostgreSQL 的说明实际不生效）
        self.DATABASE_URL = os.getenv("DATABASE_URL") or self.DATABASE_URL
        if not self.DATABASE_URL:
            db_path = self.DATA_DIR / "buildsentry.db"
            self.DATABASE_URL = f"sqlite:///{db_path}"

        # 向量数据库目录
        if self.CHROMA_DATA_DIR is None:
            self.CHROMA_DATA_DIR = self.DATA_DIR / "chroma_data"
        if self.CHROMA_CACHE_DIR is None:
            self.CHROMA_CACHE_DIR = self.DATA_DIR / "chroma_cache"

        self.CHROMA_DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.CHROMA_CACHE_DIR.mkdir(parents=True, exist_ok=True)

        # CORS配置
        cors_env = os.getenv("CORS_ORIGINS")
        if cors_env:
            self.CORS_ORIGINS = [origin.strip() for origin in cors_env.split(",")]

        # 日志配置
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", self.LOG_LEVEL)
        self.LOG_FORMAT = os.getenv("LOG_FORMAT", self.LOG_FORMAT)
        self.LOG_FILE = os.getenv("LOG_FILE", self.LOG_FILE)

    def is_production(self) -> bool:
        """是否为生产环境"""
        return self.ENVIRONMENT.lower() == "production"

    def get_device(self) -> str:
        """获取实际设备"""
        if self.DEVICE == "auto":
            try:
                import torch

                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return self.DEVICE


# 全局配置实例
settings = Settings()
