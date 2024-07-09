from fastapi import FastAPI
from src import handler
from src.schemas import Request, JobResponse

app = FastAPI(
	title="Music Generation API",
	description="API for generating music and vocals.",
	version="0.1.0"
)

@app.post("/generate", response_model=JobResponse)
async def generate(job: Request):
	"""
	Entry Point for the Music Generation API.
	"""
	return await handler(job)


