import asyncio

from llama_index.core.workflow import (
    Context,
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)

from app.workflows.frontend_events import StringEvent
from app.workflows.naive_text2cypher_steps import (
    correct_cypher_step,
    generate_cypher_step,
    naive_final_answer_prompt,
)
from app.workflows.utils import graph_store, llm


class SummarizeEvent(Event):
    question: str
    cypher: str
    context: str


class ExecuteCypherEvent(Event):
    question: str
    cypher: str


class CorrectCypherEvent(Event):
    question: str
    cypher: str
    error: str


class NaiveText2CypherRetryFlow(Workflow):
    max_retries = 1

    @step
    async def generate_cypher(self, ctx: Context, ev: StartEvent) -> ExecuteCypherEvent:
        # Init global vars
        await ctx.set("retries", 0)

        question = ev.input

        cypher_query = await generate_cypher_step(question)

        ctx.write_event_to_stream(
            StringEvent(
                result=f"Generated Cypher: {cypher_query}", label="Cypher generation"
            )
        )

        # Return for the next step
        return ExecuteCypherEvent(question=question, cypher=cypher_query)

    @step
    async def execute_query(
        self, ctx: Context, ev: ExecuteCypherEvent
    ) -> SummarizeEvent | CorrectCypherEvent:
        # Get global var
        retries = await ctx.get("retries")
        try:
            database_output = str(graph_store.structured_query(ev.cypher))
        except Exception as e:
            database_output = str(e)
            # Retry
            if retries < self.max_retries:
                await ctx.set("retries", retries + 1)
                return CorrectCypherEvent(
                    question=ev.question, cypher=ev.cypher, error=database_output
                )

        return SummarizeEvent(
            question=ev.question, cypher=ev.cypher, context=database_output
        )

    @step
    async def correct_cypher_step(
        self, ctx: Context, ev: CorrectCypherEvent
    ) -> ExecuteCypherEvent:
        print("#" * 30)
        results = await correct_cypher_step(ev.question, ev.cypher, ev.error)
        return ExecuteCypherEvent(question=ev.question, cypher=results)

    @step
    async def summarize_answer(self, ctx: Context, ev: SummarizeEvent) -> StopEvent:
        gen = await llm.astream_chat(
            naive_final_answer_prompt.format_messages(
                context=ev.context, question=ev.question, cypher_query=ev.cypher
            )
        )
        final_event = StringEvent(result="", label="Final answer")
        final_answer = ""
        async for response in gen:
            final_event.result = response.delta
            final_answer += response.delta
            ctx.write_event_to_stream(final_event)
            await asyncio.sleep(0.05)

        stop_event = StopEvent(result=f"{ev.cypher}<split>{final_answer}")

        # Return the final result
        return stop_event
