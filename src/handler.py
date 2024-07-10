from .modules import Music
from .schemas import JobResponse, Request
from .utils import asyncify

music = Music()

@asyncify
def handler(job: Request):
	"""
	Entry Point for the Music Generation API.
	"""
	input = job['input']
	if input['kind'] == 'continuation':
		assert input['prompt'] is not None, "Prompt must be provided."
		return JobResponse(url=music.generate_continuation(prompt=input['prompt']))
	elif input['kind'] == 'conditional':
		assert input['descriptions'] is not None, "Descriptions must be provided."
		return JobResponse(url=music.generate(descriptions=input['descriptions']))
	return JobResponse(url=music.generate_unconditional())