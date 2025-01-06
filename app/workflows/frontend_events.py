from uuid import uuid4

from llama_index.core.workflow import Event
from pydantic import UUID4, Field


class StringEvent(Event):
    uuid: UUID4 = Field(default_factory=uuid4)
    result: str
    label: str
