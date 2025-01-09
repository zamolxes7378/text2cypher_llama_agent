import os
from uuid import uuid4

from app.workflows.frontend_events import SseEvent
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


class JokeWorkflow(Workflow):
    llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    @step
    async def generate_joke(self, ctx: Context, ev: StartEvent) -> SseEvent:
        input = ev.input
        if not input:
            raise ValueError(f"generate_joke is missing input.")

        prompt = f"Write your best joke about {input}."
        response = await self.llm.acomplete(prompt)

        # Emit the joke event
        joke_event = SseEvent(
            label="Joke",
            message=str(response),
        )
        ctx.write_event_to_stream(joke_event)

        # Return for the next step
        return joke_event

    @step
    async def critique_joke(self, ctx: Context, ev: SseEvent) -> StopEvent:
        joke = ev.message
        if not joke:
            raise ValueError(f"critique_joke is missing joke.")

        prompt = (
            f"Give a 1 sentence analysis and critique of the following joke: {joke}"
        )

        gen = await self.llm.astream_complete(prompt)
        async for response in gen:
            ctx.write_event_to_stream(
                SseEvent(
                    label="Critique",
                    message=response.delta,
                )
            )

        stop_event = StopEvent(result="Workflow completed.")

        # Return the final result
        return stop_event
