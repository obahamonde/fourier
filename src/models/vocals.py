"""
Module for generating lyrics using a pre-trained Bark model.
"""

import tempfile
from functools import cached_property
from typing import Any

import httpx
import numpy as np
import scipy  # type: ignore
import torch
from bark import SAMPLE_RATE, generate_audio, preload_models  # type: ignore
from pydantic import BaseModel

from src.models.storage import ObjectStorage

preload_models()


class Vocals(BaseModel):
    """
    API for generating lyrics using a pre-trained Bark model.

    Methods:
                    generate_lyrics: Generates lyrics based on a given text prompt.
    """

    @cached_property
    def storage(self):
        return ObjectStorage()

    async def _fetch_audio(self, url: str):
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            audio_data = np.frombuffer(response.content, dtype=np.int16)  # type: ignore
            return torch.tensor(data=audio_data, dtype=torch.float32, device="cuda")

    async def _save_wav_tensor(self, tensor: torch.Tensor) -> str:
        tensor = tensor.to("cpu").float()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            scipy.io.wavfile.write(tmp_file.name, SAMPLE_RATE, tensor.numpy())
            tmp_file.seek(0)
            file_content = tmp_file.read()
            key = tmp_file.name.split("/")[-1]
            return await self.storage.put(key=key, data=file_content)

    async def generate_vocals(self, text: str):
        nd_array: np.ndarray[float, Any] = generate_audio(text="\n".join(self._parse(text)))  # type: ignore
        tensor = torch.tensor(data=nd_array, dtype=torch.float32, device="cuda")
        return await self._save_wav_tensor(tensor)

    def _parse(self, text: str):
        for line in text.splitlines():
            if line.strip() != "":
                yield " ♪ " + line.strip() + " ♪ "
            else:
                continue
