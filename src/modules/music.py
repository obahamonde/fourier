"""
Module for generating music using a pre-trained model.
"""
from typing import Literal, Union
import tempfile
from functools import cached_property

import numpy as np
import torchaudio  # type: ignore
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()
import spacy
from spacy.language import Language
import httpx
import scipy  # type: ignore
import torch
from audiocraft.models.musicgen import MusicGen
from pytube import YouTube # type: ignore

from .storage import ObjectStorage


MODELS:dict[Literal['en','es','music'],Union[Language,MusicGen]] = {
    'en':spacy.load('en_core_web_sm'),
    'es':spacy.load('es_core_news_sm'),
    'music':MusicGen.get_pretrained("facebook/musicgen-melody-large",device=torch.device('cuda')) # type: ignore
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
        default=160000, title="Max Length", description="The maximum length of the audio in samples"
    )

    @cached_property
    def music(self)->MusicGen:
        mdo = MODELS['music'] # type: ignore
        assert isinstance(mdo, MusicGen)
        mdo.set_generation_params(use_sampling=True,temperature=1,duration=15,extend_stride=24)
        return mdo
    @cached_property
    def storage(self):
        return ObjectStorage()

    def _trim_audio_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.shape[-1] > self.max_length:
            return tensor[..., : self.max_length]
        return tensor

    def _fetch_audio_from_youtube(self, url: str) -> torch.Tensor:
        yt = YouTube(url)
        audio_stream = yt.streams.filter(only_audio=True).first() # type: ignore
        temp_audio_file = audio_stream.download() # type: ignore

        waveform, sample_rate = torchaudio.load(temp_audio_file) # type: ignore
        if sample_rate != self.music.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=self.music.sample_rate # type: ignore
            )
            waveform = resampler(waveform)

        return self._trim_audio_tensor(waveform) # type: ignore

    def _save_wav_tensor(self,*, tensor: torch.Tensor) -> str:
        tensor = tensor.to("cpu").float()
        scaled_tensor = (tensor * 32767).clamp(min=-32768, max=32767).short()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            scipy.io.wavfile.write(tmp_file.name, 16000, scaled_tensor.numpy())
            tmp_file.seek(0)
            file_content = tmp_file.read()
            key = tmp_file.name.split("/")[-1]
            self.storage.put(key=key, data=file_content) # type: ignore
            return self.storage.get(key=key) # type: ignore

    def _fetch_audio(self,*, url: str):
        if "youtube.com" in url:
            return self._fetch_audio_from_youtube(url)
        with httpx.Client() as client:
            response = client.get(url)
            audio_data = np.frombuffer(response.content, dtype=np.int16)  # type: ignore
            tensor = torch.tensor(audio_data, dtype=torch.float32, device="cpu")
            return self._trim_audio_tensor(tensor)

    def generate(self, * , descriptions: Union[str, list[str]],language:Literal['en','es']='en'):
        if isinstance(descriptions, str):
            lang = MODELS[language]
            assert isinstance(lang, Language)
            descriptions = [token.text for token in lang(descriptions).sents]
        tensor = self.music.generate(descriptions=descriptions, progress=True)
        assert isinstance(tensor, torch.Tensor)
        return self._save_wav_tensor(tensor=tensor)

    def generate_continuation(self, *, prompt: Union[str, list[float]]):
        if isinstance(prompt, str):
            if "youtube.com" in prompt:
                tensor = self._fetch_audio_from_youtube(prompt)
            else:
                tensor = self._fetch_audio(url=prompt)
        else:
            tensor = torch.tensor(prompt, dtype=torch.float32, device="cpu")
        tensor_out = self.music.generate_continuation(
            prompt=tensor, prompt_sample_rate=self.music.sample_rate, progress=True
        )
        assert isinstance(tensor_out, torch.Tensor)
        return self._save_wav_tensor(tensor=tensor_out)

    def generate_unconditional(self):
        tensor = self.music.generate_unconditional(progress=True, num_samples=1)
        assert isinstance(tensor, torch.Tensor)
        return self._save_wav_tensor(tensor=tensor)
