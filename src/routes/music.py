from fastapi import APIRouter

from ..models import Music, NamespacedPrompt, Prompt, URLResponse

app = APIRouter()
music = Music()

@app.post("/api/generate")
async def generate_music(data: Prompt):
    """
    Generate music based on the provided data.

    Args:
        data (Prompt): The prompt data for generating music.

    Returns:
        URLResponse: The response containing the generated music URL.
    """
    url = await music.generate(data.text)
    return URLResponse(url=url)

@app.post("/api/generate/next")
async def continue_generation(data: NamespacedPrompt):
    """
    Continue the generation of music based on the given prompt and namespace.

    Args:
        data (NamespacedPrompt): The input data containing the text prompt and namespace.

    Returns:
        URLResponse: The response containing the generated music URL.
    """
    url = await music.continue_generation(data.text, data.namespace)
    return URLResponse(url=url)

@app.post("/api/seed")
async def seed_generation():
    """
    Generate a seed for music generation.

    Returns:
        URLResponse: The URL of the generated seed.
    """
    url = await music.seed_generation()
    return URLResponse(url=url)