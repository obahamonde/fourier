from typing import Generator, ParamSpec, Sequence, TypeVar, Callable, Awaitable

T = TypeVar("T")
P = ParamSpec("P")

def chunker(seq: Sequence[T], size: int) -> Generator[Sequence[T], None, None]:
    return (seq[pos : pos + size] for pos in range(0, len(seq), size))


def asyncify(func:Callable[P,T]) -> Callable[P,Awaitable[T]]:
    async def wrapper(*args:P.args, **kwargs:P.kwargs) -> T:
        return func(*args, **kwargs)
    return wrapper

