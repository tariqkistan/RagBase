import os

os.environ.setdefault("CHUNK_SIZE", "500")
os.environ.setdefault("CHUNK_OVERLAP", "50")
os.environ.setdefault("TOP_K", "4")
os.environ.setdefault("OPENAI_API_KEY", "test-key-for-testing")
os.environ.setdefault("OPENAI_CHAT_MODEL", "gpt-4o")
os.environ.setdefault("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://test:test@localhost:5432/test")
