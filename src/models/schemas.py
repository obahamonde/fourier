from pydantic import BaseModel, Field


class Prompt(BaseModel):
    text: str = Field(..., title="Prompt text", description="The text of the prompt")
    
class NamespacedPrompt(Prompt):
	namespace: str = Field(..., title="Namespace", description="The namespace for the music generation")
 
class URLResponse(BaseModel):
	url: str = Field(..., title="URL", description="The URL of the generated audio file")