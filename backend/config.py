from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Documentation Assistant"
    ui_language: str = "en"
    llm_model: str = "qwen2.5:7b-instruct-q4_K_M"
    embed_model: str = "multilingual-e5-large"
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 3
    admin_user: str = "admin"
    admin_password: str = "changeme"
    ollama_url: str = "http://ollama:11434"
    chroma_url: str = "http://chromadb:8001"

    model_config = {"env_file": ".env", "case_sensitive": False}


settings = Settings()
