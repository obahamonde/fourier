from typing import Literal, Optional, Union

from pydantic import BaseModel, Field

from typing_extensions import TypedDict, Required

class Job(TypedDict):
    kind: Literal["unconditional", "conditional", "continuation","vocals"]
    descriptions: Optional[Union[str, list[str]]]
    prompt: Optional[Union[str, list[float]]]
    text:Optional[str]
    
class Request(TypedDict):
    input:Required[Job]
    
class JobResponse(BaseModel):
    url: str = Field(
        ..., title="URL", description="The URL of the generated audio file"
    )
