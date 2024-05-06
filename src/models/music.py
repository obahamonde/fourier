"""
Module for generating music using a pre-trained model.
"""

import tempfile
from functools import cached_property

import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()
import httpx
import scipy  # type: ignore
import torch
from audiocraft.models.musicgen import MusicGen

from .storage import ObjectStorage

music_gen: MusicGen = MusicGen.get_pretrained("facebook/musicgen-small")


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
        default=16000, description="The maximum length of the audio in samples."
    )

    @cached_property
    def music(self):
        return music_gen

    @cached_property
    def storage(self):
        return ObjectStorage()

    def _trim_audio_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.shape[-1] > self.max_length:
            return tensor[..., : self.max_length]
        return tensor

    async def _save_wav_tensor(self, tensor: torch.Tensor) -> str:
        tensor = tensor.to("cpu").float()
        scaled_tensor = (tensor * 32767).clamp(min=-32768, max=32767).short()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            scipy.io.wavfile.write(tmp_file.name, 16000, scaled_tensor.numpy())
            tmp_file.seek(0)
            file_content = tmp_file.read()
            key = tmp_file.name.split("/")[-1]
            await self.storage.put(key=key, data=file_content)
            return await self.storage.get(key=key)

    async def _fetch_audio(self, url: str):
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            audio_data = np.frombuffer(response.content, dtype=np.int16)  # type: ignore
            tensor = torch.tensor(data=audio_data, dtype=torch.float32, device="cpu")
            return self._trim_audio_tensor(tensor)

    async def generate(self, text: str):
        """
        Generates music based on a given text prompt.

        Args:
                text (str): The text prompt for generating music.

        Returns:
                str: The key of the generated audio file in the object storage.
        """
        tensor = self.music.generate(descriptions=[text], progress=True)
        assert isinstance(tensor, torch.Tensor)
        return await self._save_wav_tensor(tensor)

    async def continue_generation(self, text: str, namespace: str):
        """
        Generates a continuation of the music based on a given text prompt and the previous audio.

        Args:
                text (str): The text prompt for generating the continuation.

        Returns:
                str: The key of the generated audio file in the object storage.
        """
        url = await self.storage.get(key=namespace)
        tensor = await self._fetch_audio(url)
        tensor_out = self.music.generate_continuation(
            prompt=tensor, prompt_sample_rate=16000, descriptions=[text], progress=True
        )
        assert isinstance(tensor_out, torch.Tensor)
        return await self._save_wav_tensor(tensor_out)

    async def rag_generation(self, text: str):
        """
        Generates music with a melody and chroma based on a given text prompt.

        Args:
                text (str): The text prompt for generating music.

        Returns:
                str: The key of the generated audio file in the object storage.
        """
        melody = self.music.generate(descriptions=[text], progress=True)
        assert isinstance(melody, torch.Tensor)
        tensor = self.music.generate_with_chroma(
            descriptions=[text],
            progress=True,
            melody_sample_rate=16000,
            melody_wavs=melody,
        )
        assert isinstance(tensor, torch.Tensor)
        return await self._save_wav_tensor(tensor)

    async def seed_generation(self):
        """
        Generates unconditional music with a fixed number of samples.

        Returns:
                str: The key of the generated audio file in the object storage.
        """
        tensor = self.music.generate_unconditional(progress=True, num_samples=1)
        assert isinstance(tensor, torch.Tensor)
        return await self._save_wav_tensor(tensor)
