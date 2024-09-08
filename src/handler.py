from .modules import Music
from .schemas import Job
from .utils import handle

music = Music()


@handle
async def handler(job: Job):
    """
    Entry Point for the Music Generation API.
    """
    return await music.run(job=job)
