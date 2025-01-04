import os
from llama_index.llms.openai import OpenAI
from llama_index.core.workflow import (
    Event,
    Context,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)

class JokeEvent(Event):
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

        prompt = f"Give a 5 word analysis and critique of the following joke: {joke}"
        response = await self.llm.acomplete(prompt)

        # Emit the critique event - not needed if this is last step
        stop_event = StopEvent(result=str(response))
        ctx.write_event_to_stream(stop_event)

        # Return the final result
        return stop_event
