from fastapi import APIRouter

from ..models import Prompt, URLResponse, Vocals

app = APIRouter(tags=["vocals"], prefix="/api/vocals")
vocals = Vocals()


@app.post("/generate")
async def generate_vocals(data: Prompt):
    """
    Generate vocals based on the provided data.

    Args:
            data (Prompt): The prompt data for generating vocals.

    Returns:
            URLResponse: The response containing the generated vocals URL.
    """
    url = await vocals.run(data.text)
    return URLResponse(url=url)
