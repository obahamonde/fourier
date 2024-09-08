"""
Module for generating music using a pre-trained model.
"""

import tempfile
from functools import cached_property
from typing import Literal, Union

import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()
import httpx
import scipy  # type: ignore
import spacy
import torch
from audiocraft.models.musicgen import MusicGen
from spacy.language import Language

from ..schemas import Job
from .storage import ObjectStorage

MODELS: dict[Literal["en", "es", "music"], Union[Language, MusicGen]] = {
    "en": spacy.load("en_core_web_sm"),
    "es": spacy.load("es_core_news_sm"),
    "music": MusicGen.get_pretrained("facebook/musicgen-melody-large", device=torch.device("cuda")),  # type: ignore
}


class Music(BaseModel):
    """
    API for generating music using a pre-trained model.

    Args:
                    max_length (int): The maximum length of the audio in samples.
                    key (str): The key of the audio file in the object storage.

    Attributes:
                    music (cached_property): The pre-trained music generation model.
                    storage (cached_property): The object storage for saving and retrieving audio files.

    Methods:
                    _trim_audio_tensor: Trims the audio tensor if its length exceeds the maximum length.
                    _save_wav_tensor: Saves the audio tensor as a WAV file and stores it in the object storage.
                    _fetch_audio: Fetches audio data from a given URL and converts it to a tensor.
                    generate: Generates music based on a given text prompt.
                    continue_generation: Generates a continuation of the music based on a given text prompt and the previous audio.
                    rag_generation: Generates music with a melody and chroma based on a given text prompt.
                    seed_generation: Generates unconditional music with a fixed number of samples.
    """

    max_length: int = Field(
        default=1600000,
        title="Max Length",
        description="The maximum length of the audio in samples",
    )

    @cached_property
    def music(self) -> MusicGen:
        mdo = MODELS["music"]  # type: ignore
        assert isinstance(mdo, MusicGen)
        mdo.set_generation_params(
            use_sampling=True, temperature=1, duration=15, extend_stride=12
        )
        return mdo

    @cached_property
    def storage(self):
        return ObjectStorage()

    def _trim_audio_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.shape[-1] > self.max_length:
            return tensor[..., : self.max_length]
        return tensor
    
    def _get_descriptions(self, *, prompt: str, language: Literal["en", "es"] = "en"):
        lang = MODELS[language]
        assert isinstance(lang, Language)
        string=""
        for sent in lang(prompt).sents:
            string+=sent.text
            yield string

        return string
    async def _save_wav_tensor(self, *, tensor: torch.Tensor) -> str:
        tensor = tensor.to("cpu").float()
        scaled_tensor = (tensor * 32767).clamp(min=-32768, max=32767).short()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            scipy.io.wavfile.write(tmp_file.name, 16000, scaled_tensor.numpy())
            tmp_file.seek(0)
            file_content = tmp_file.read()
            key = tmp_file.name.split("/")[-1]
            return await self.storage.put(key=key, data=file_content)

    async def _fetch_audio(self, *, url: str):
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            audio_data = np.frombuffer(response.content, dtype=np.int16)  # type: ignore
            tensor = torch.tensor(audio_data, dtype=torch.float32, device="cpu")
            return self._trim_audio_tensor(tensor)

    async def generate(self, *, prompt: str, language: Literal["en", "es"] = "en"):
        lang = MODELS[language]
        assert isinstance(lang, Language)
        descriptions = [d for d in self._get_descriptions(prompt=prompt, language=language)]
        tensor = self.music.generate(
            descriptions=descriptions, progress=True, return_tokens=False
        )
        assert isinstance(tensor, torch.Tensor)
        return await self._save_wav_tensor(tensor=tensor)

    async def generate_continuation(self, *, audio: Union[str, list[float]]):
        if isinstance(audio, str):
            tensor = await self._fetch_audio(url=audio)
        else:
            tensor = torch.tensor(audio, dtype=torch.float32, device="cpu")
        tensor_out = self.music.generate_continuation(
            prompt=tensor, prompt_sample_rate=self.music.sample_rate, progress=True
        )
        assert isinstance(tensor_out, torch.Tensor)
        return await self._save_wav_tensor(tensor=tensor_out)

 
    async def generate_unconditional(self):
        tensor = self.music.generate_unconditional(progress=True, num_samples=1)
        assert isinstance(tensor, torch.Tensor)
        return await self._save_wav_tensor(tensor=tensor)

    async def run(self, *, job: Job):
        if job.type == "zero":
            return await self.generate_unconditional()
        if job.type == "text":
            assert job.prompt is not None, "Prompt must be provided."
            assert job.language is not None, "Language must be provided."
            return await self.generate(prompt=job.prompt, language=job.language)
        if job.type == "next":
            assert job.audio is not None, "Audio must be provided."
            return await self.generate_continuation(audio=job.audio)
        else:
            raise ValueError(f"Invalid job type: {job.type}")
