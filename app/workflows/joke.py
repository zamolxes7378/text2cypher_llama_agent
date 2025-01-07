import os
from uuid import uuid4

from llama_index.core.workflow import (
    Context,
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)
from llama_index.llms.openai import OpenAI
from pydantic import UUID4, Field


class JokeEvent(Event):
    uuid: UUID4 = Field(default_factory=uuid4)
    result: str


class CritiqueEvent(Event):
    uuid: UUID4 = Field(default_factory=uuid4)
    result: str


class JokeWorkflow(Workflow):
    llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    @step
    async def generate_joke(self, ctx: Context, ev: StartEvent) -> JokeEvent:
        input = ev.input
        if not input:
            raise ValueError(f"generate_joke is missing input.")

        prompt = f"Write your best joke about {input}."
        response = await self.llm.acomplete(prompt)

        # Emit the joke event
        joke_event = JokeEvent(result=str(response))
        ctx.write_event_to_stream(joke_event)

        # Return for the next step
        return joke_event

    @step
    async def critique_joke(self, ctx: Context, ev: JokeEvent) -> StopEvent:
        joke = ev.result
        if not joke:
            raise ValueError(f"critique_joke is missing joke.")

        prompt = (
            f"Give a 1 sentence analysis and critique of the following joke: {joke}"
        )

        gen = await self.llm.astream_complete(prompt)
        critique_event = CritiqueEvent(result="")
        async for response in gen:
            critique_event.result = response.delta
            ctx.write_event_to_stream(critique_event)

        stop_event = StopEvent(result="Workflow completed.")

        # Return the final result
        return stop_event
