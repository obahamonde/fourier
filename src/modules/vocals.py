""" Module for generating lyrics using a pre-trained Bark model. """

import re
import tempfile
from functools import cached_property
from typing import Any, Generator

import httpx
import numpy as np
import scipy  # type: ignore
import torch
from bark import SAMPLE_RATE, generate_audio, preload_models  # type: ignore

from .chatbot import Agent
from .storage import ObjectStorage

preload_models()


class Vocals:
    """API for generating lyrics using a pre-trained Bark model.
    Methods:
            generate_lyrics: Generates lyrics based on a given text prompt.
            generate_full_song: Generates a full song by concatenating vocals from multiple chunks of text.
    """

    @cached_property
    def storage(self):
        return ObjectStorage()

    @cached_property
    def agent(self):
        return Agent(messages=[])

    def _fetch_audio(self, url: str):
        with httpx.Client() as client:
            response = client.get(url)
            audio_data = np.frombuffer(response.content, dtype=np.int16)  # type: ignore
            return torch.tensor(data=audio_data, dtype=torch.float32, device="cuda")

    def _save_wav_tensor(self, tensor: torch.Tensor) -> str:
        tensor = tensor.to("cpu").float()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            scipy.io.wavfile.write(tmp_file.name, SAMPLE_RATE, tensor.numpy())
            tmp_file.seek(0)
            file_content = tmp_file.read()
            key = tmp_file.name.split("/")[-1]
            return self.storage.put(key=key, data=file_content)

    def generate_vocals(self, text: str):
        nd_array: np.ndarray[float, Any] = generate_audio(text="\n".join(self._parse(text)))  # type: ignore
        tensor = torch.tensor(data=nd_array, dtype=torch.float32, device="cuda")
        return self._save_wav_tensor(tensor)

    def generate_full_song(self, text: str):
        chunks = list(self._parse(text))
        vocals: list[torch.Tensor] = []
        for chunk in chunks:
            nd_array: np.ndarray[float, Any] = generate_audio(text=chunk)  # type: ignore
            tensor = torch.tensor(data=nd_array, dtype=torch.float32, device="cuda")
            vocals.append(tensor)
        full_song = torch.cat(vocals)
        return self._save_wav_tensor(full_song)

    def _parse(self, text: str) -> Generator[str, None, None]:
        pattern = r"(?m)^♪.*(?:\n♪.*)*"
        chunks = re.findall(pattern, text)

        excluded_patterns = [
            r"^♪\s*♪$",
            r"^♪\s*\*\*.*\*\*\s*♪$",
        ]

        for chunk in chunks:
            if not any(re.match(pattern, chunk) for pattern in excluded_patterns):
                yield chunk

    def run(self, text: str):
        lyrics = self.agent.generate_lyrics(text)
        return self.generate_vocals(lyrics)