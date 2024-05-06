import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
# from src.models.vocals import Vocals

# v = Vocals()

ai = OpenAI(base_url=os.getenv("OPENAI_BASE_URL"), api_key=os.getenv("OPENAI_API_KEY"))


# async def main():
#     print(await v.generate_vocals("Hello, world!"))


from typing import TypeVar

T = TypeVar("T")
TONE = " ♪ "


def add_tone_separator(text: str) -> str:
    """
    Add a tone separator to the text.

    Args:
                                                                                                                                    text (str): The text to add the tone separator to.

    Returns:
                                                                                                                                    str: The text with the tone separator added.
    """
    return "\n".join([f"{TONE}{line}{TONE}" for line in text.splitlines()])


def chat(text: str) -> str | None:
    """
    Chat with the AI.

    Args:
                                                                                                                                    text (str): The text to chat with the AI.

    Returns:
                                                                                                                                    str: The response from the AI.
    """
    # Generate the response from the AI.
    return (
        ai.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {
                    "role": "system",
                    "content": "You are a singer that generates lyrics based on a given text prompt",
                },
                {"role": "user", "content": text},
            ],
        )
        .choices[0]
        .message.content
    )


def strip_adyacent_spaced_tones(text: str) -> str:
    """
    Strip adyacent tone separators from the text.

    Args:
                                                                                                                                    text (str): The text to strip the tone separators from.

    Returns:
                                                                                                                                    str: The text with the tone separators stripped.
    """


lyrics = chat("Generate a song about love")
assert lyrics is not None
print(lyrics)
print(add_tone_separator(lyrics))
