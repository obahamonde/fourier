from .modules import Music
from .schemas import Job
from .utils import handle, ttl_cache


@ttl_cache
def init():
    """
    Initializes and returns an instance of the Music class.

    Returns:
        Music: An instance of the Music class.
    """
    return Music()


music = init()


@handle
async def handler(job: Job):
    """
    Handles the execution of a job using the provided music instance.

    Args:
        job (Job): The job to be executed.
        music (Music, optional): The music instance to use for executing the job. Defaults to the instance provided by the `init` dependency.

    Returns:
        The result of the music instance's `run` method with the provided job.
    """
    return await music.run(job=job)
