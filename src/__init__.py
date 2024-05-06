from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import music_app


def create_app():
    app = FastAPI(title="Music Generation API", version="0.1.0")
    app.include_router(music_app)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return app
