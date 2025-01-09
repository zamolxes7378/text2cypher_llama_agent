from uuid import uuid4

from llama_index.core.workflow import Event
from pydantic import UUID4, Field


class SseEvent(Event):
    label: str
    message: str
