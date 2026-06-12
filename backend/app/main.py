from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

from app.api import chat, documents, health, search
from app.config import get_settings
from app.db.database import init_db


settings = get_settings()
app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(documents.router)
app.include_router(search.router)
app.include_router(chat.router)


def custom_openapi() -> dict:
    if app.openapi_schema:
        return app.openapi_schema

    schema = get_openapi(
        title=settings.app_name,
        version="0.1.0",
        routes=app.routes,
    )
    upload_schema = schema.get("components", {}).get("schemas", {}).get(
        "Body_upload_documents_api_documents_upload_post"
    )
    if upload_schema:
        files_schema = upload_schema.get("properties", {}).get("files")
        if files_schema and files_schema.get("items", {}).get("contentMediaType"):
            files_schema["items"] = {"type": "string", "format": "binary"}

    app.openapi_schema = schema
    return app.openapi_schema


app.openapi = custom_openapi


@app.on_event("startup")
def on_startup() -> None:
    settings.ensure_dirs()
    init_db()
