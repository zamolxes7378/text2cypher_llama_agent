from llama_index.core.workflow import (
    Context,
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)

from app.workflows.frontend_events import SseEvent
from app.workflows.naive_text2cypher_steps import (
    generate_cypher_step,
    get_naive_final_answer_prompt,
)
from app.workflows.utils import default_llm, graph_store


class SummarizeEvent(Event):
    question: str
    cypher: str
    context: str


class ExecuteCypherEvent(Event):
    question: str
    cypher: str


class NaiveText2CypherFlow(Workflow):
    def __init__(self, llm=None, *args, **kwargs):
        super().__init__(*args, **kwargs)  # Call the parent init
        self.llm = llm or default_llm  # Add child-specific logic

    @step
    async def generate_cypher(self, ctx: Context, ev: StartEvent) -> ExecuteCypherEvent:
        question = ev.input

        cypher_query = await generate_cypher_step(self.llm, question)

        ctx.write_event_to_stream(
            SseEvent(
                label="Cypher generation",
                message=f"Generated Cypher: {cypher_query}",
            )
        )

        # Return for the next step
        return ExecuteCypherEvent(question=question, cypher=cypher_query)

    @step
    async def execute_query(
        self, ctx: Context, ev: ExecuteCypherEvent
    ) -> SummarizeEvent:
        try:
            database_output = str(graph_store.structured_query(ev.cypher))
        except Exception as e:
            database_output = str(e)
        return SummarizeEvent(
            question=ev.question, cypher=ev.cypher, context=database_output
        )

    @step
    async def summarize_answer(self, ctx: Context, ev: SummarizeEvent) -> StopEvent:
        naive_final_answer_prompt = get_naive_final_answer_prompt()
        gen = await self.llm.astream_chat(
            naive_final_answer_prompt.format_messages(
                context=ev.context, question=ev.question, cypher_query=ev.cypher
            )
        )
        final_answer = ""
        async for response in gen:
            final_answer += response.delta
            ctx.write_event_to_stream(
                SseEvent(
                    label="Final answer",
                    message=response.delta,
                )
            )

        stop_event = StopEvent(
            result={
                "cypher": ev.cypher,
                "question": ev.question,
                "answer": final_answer,
            }
        )

        # Return the final result
        return stop_event
