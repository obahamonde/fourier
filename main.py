from fastapi import FastAPI
from src.handler import handler, JobResponse, Request

app = FastAPI(
	title="Music Generation API",
	description="API for generating music using OpenAI's Jukebox.",
	version="1.0.0"
)

@app.post("/generate")
async def generate(data: Request)->JobResponse:
	"""
	Generate music based on the input data.
	"""
	return await handler(data)