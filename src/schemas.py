from typing import Literal, Optional

from pydantic import BaseModel, Field, computed_field


class Job(BaseModel):
    """
    Represents a job object.

    Attributes:
        prompt (Optional[str]): The prompt for the job.
        audio (Optional[list[float]]): The audio data for the job.
        language (Optional[Literal["en", "es"]]): The language of the job.
        response_format (Optional[Literal["mp3", "wav", "ogg", "flac"]]): The response format of the job.
    """

    prompt: Optional[str] = Field(default=None)
    audio: Optional[list[float]] = Field(default=None)
    language: Optional[Literal["en", "es"]] = Field(default="en")
    response_format: Optional[Literal["mp3", "wav", "ogg", "flac"]] = Field(
        default="wav"
    )

    @computed_field(return_type=Literal["zero", "text", "next"])
    @property
    def type(self) -> Literal["zero", "text", "next"]:
        """
        Returns the type of the job.

        Returns:
            Literal["zero", "text", "next"]: The type of the job.
        """
        if self.prompt:
            return "text"
        if self.audio:
            return "next"
        return "zero"


class JobResponse(BaseModel):
    """
    Represents the response for a job.

    Attributes:
        url (str): The URL of the generated audio file.
        audio (list[float]): The raw audio signal.
        refined_prompt (Optional[str]): The refined prompt, if available.
    """

    url: str = Field(
        ..., title="URL", description="The URL of the generated audio file"
    )
    audio: list[float] = Field(..., title="Audio", description="The raw audio signal")
    refined_prompt: Optional[str] = Field(default=None)
