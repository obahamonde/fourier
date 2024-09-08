from fastapi import FastAPI

from src.handler import Job, handler

app = FastAPI(
    title="Music Generation API",
    description="API for generating music using OpenAI's Jukebox.",
    version="1.0.0",
)


@app.post("/v1/music/generations")
async def main(data: Job):
    """
    Generate music based on the input data.
    """
    return await handler(job=data)


@app.get("/")
def root():
    return {"message": "Welcome to the Music Generation API.", "status": "running."}
