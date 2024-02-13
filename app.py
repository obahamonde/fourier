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
from TTS.tts.models.bark import (Bark, BarkAudioConfig,  # type: ignore
                                 BertTokenizer)

from src.models.storage import ObjectStorage

preload_models()


class Lyrics(BaseModel):
	"""
	API for generating lyrics using a pre-trained Bark model.

	Methods:
			generate_lyrics: Generates lyrics based on a given text prompt.
	"""

	@cached_property
	def bark(self):
		return Bark(
			config=self.config,
			tokenizer=BertTokenizer.from_pretrained("bert-base-uncased"), # type: ignore
		)

	@cached_property
	def config(self):
		return BarkAudioConfig(sample_rate=SAMPLE_RATE, output_sample_rate=SAMPLE_RATE)

	@cached_property
	def storage(self):
		return ObjectStorage()

	def _trim_audio_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
		if tensor.shape[-1] > SAMPLE_RATE:
			return tensor[..., :SAMPLE_RATE]
		return tensor

	async def _fetch_audio(self, url: str):
		async with httpx.AsyncClient() as client:
			response = await client.get(url)
			audio_data = np.frombuffer(response.content, dtype=np.int16)  # type: ignore
			tensor = torch.tensor(data=audio_data, dtype=torch.float32, device="cpu")
			return self._trim_audio_tensor(tensor)

	async def _save_wav_tensor(self, tensor: torch.Tensor) -> str:
		tensor = tensor.to("cpu").float()
		with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
			scipy.io.wavfile.write(tmp_file.name, SAMPLE_RATE, tensor.numpy())
			tmp_file.seek(0)
			file_content = tmp_file.read()
			key = tmp_file.name.split("/")[-1]
			return await self.storage.put(key=key, data=file_content)

	async def generate_vocals(self,text:str,key:str):
		nd_array:np.ndarray[float,Any] = generate_audio(text="\n".join(self._parse(text))) # type: ignore
		tensor = torch.tensor(data=nd_array, dtype=torch.float32, device="cpu")	
		return await self._save_wav_tensor(self._trim_audio_tensor(tensor))

	def _parse(self, text: str):
		for line in text.split("\n"):
			if line.strip() != "":
				yield "♪ " + line.strip() + " ♪"
			else:
				continue


synthesizer = Lyrics()

import asyncio


async def main():
	text = "I'm a little teapot\nShort and stout\nHere is my handle\nHere is my spout"
	key = "test"
	print(await synthesizer.generate_vocals(text,key))
