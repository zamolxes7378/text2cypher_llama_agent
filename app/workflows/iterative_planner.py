import asyncio
from typing import List

from llama_index.core.workflow import (
    Context,
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)

from app.workflows.frontend_events import StringEvent
from app.workflows.iterative_planner_steps import *
from app.workflows.utils import default_llm, graph_store

MAX_INFORMATION_CHECKS = 3


class InitialPlan(Event):
    question: str


class GenerateCypher(Event):
    subquery: str


class ValidateCypher(Event):
    subquery: str
    generated_cypher: str


class CorrectCypher(Event):
    cypher: str
    subquery: str
    errors: List[str]


class ExecuteCypher(Event):
    validated_cypher: str
    subquery: str


class InformationCheck(Event):
    cypher: str
    subquery: str
    database_output: list


class FinalAnswer(Event):
    context: str


class IterativePlanningFlow(Workflow):
    def __init__(self, llm=None, *args, **kwargs):
        super().__init__(*args, **kwargs)  # Call the parent init
        self.llm = llm or default_llm  # Add child-specific logic

    @step
    async def start(self, ctx: Context, ev: StartEvent) -> InitialPlan | FinalAnswer:
        original_question = ev.input
        # Init global vars
        await ctx.set(
            "original_question", original_question
        )  # So we don't need to pass the question all the time
        await ctx.set(
            "subqueries_cypher_history", {}
        )  # History of which queries were executed
        # LLM call
        guardrails_output = await guardrails_step(self.llm, original_question)
        if guardrails_output.get("next_event") == "generate_final_answer":
            context = "The question is not about movies or cast, so I cannot answer the question"
            final_answer = FinalAnswer(context=context)
            return final_answer

        return InitialPlan(question=original_question)

    @step
    async def initial_plan(self, ctx: Context, ev: InitialPlan) -> GenerateCypher:
        original_question = ev.question
        # store in global context
        initial_plan_output = await initial_plan_step(self.llm, original_question)
        subqueries = initial_plan_output["arguments"].get("plan")

        ctx.write_event_to_stream(
            StringEvent(result=f"Plan:{subqueries}", label="Planning")
        )
        await ctx.set(
            "information_checks", 0
        )  # Current number of information check steps
        await ctx.set("dynamic_notebook", "")  # Current knowledge

        # we use this in ctx.collect() to know how many events to wait for
        await ctx.set("count_of_subqueries", len(subqueries[0]))
        await ctx.set("plan", subqueries)

        # Send events in the current step of the plan
        for subquery in subqueries[0]:
            ctx.send_event(GenerateCypher(subquery=subquery))

    @step(num_workers=4)
    async def generate_cypher_step(
        self, ctx: Context, ev: GenerateCypher
    ) -> ValidateCypher:
        # print("Running generate_cypher ", ev.subquery)
        generated_cypher = await generate_cypher_step(self.llm, ev.subquery)
        return ValidateCypher(subquery=ev.subquery, generated_cypher=generated_cypher)

    @step(num_workers=4)
    async def validate_cypher_step(
        self, ctx: Context, ev: ValidateCypher
    ) -> FinalAnswer | ExecuteCypher | CorrectCypher:
        # print("Running validate_cypher ", ev)
        results = await validate_cypher_step(self.llm, ev.subquery, ev.generated_cypher)
        if results["next_action"] == "end":  # DB value mapping
            return FinalAnswer(context=str(results["mapping_errors"]))
        if results["next_action"] == "execute_cypher":
            return ExecuteCypher(
                subquery=ev.subquery, validated_cypher=ev.generated_cypher
            )
        if results["next_action"] == "correct_cypher":
            return CorrectCypher(
                subquery=ev.subquery,
                cypher=ev.generated_cypher,
                errors=results["cypher_errors"],
            )

    @step(num_workers=4)
    async def correct_cypher_step(
        self, ctx: Context, ev: CorrectCypher
    ) -> ValidateCypher:
        results = await correct_cypher_step(self.llm, ev.subquery, ev.cypher, ev.errors)
        return ValidateCypher(subquery=ev.subquery, generated_cypher=results)

    @step
    async def execute_cypher_step(
        self, ctx: Context, ev: ExecuteCypher
    ) -> InformationCheck:
        ctx.write_event_to_stream(
            StringEvent(
                result=f"Executing Cypher query: {ev.validated_cypher}",
                label="Cypher Execution",
            )
        )
        try:
            database_output = graph_store.structured_query(ev.validated_cypher)[
                :100
            ]  # Hard limit of 100 results
        except Exception as e:  # Dividing by zero, etc... or timeout
            database_output = str(e)

        return InformationCheck(
            subquery=ev.subquery,
            cypher=ev.validated_cypher,
            database_output=database_output,
        )

    @step
    async def information_check_step(
        self, ctx: Context, ev: InformationCheck
    ) -> GenerateCypher | FinalAnswer:
        # retrieve from context
        number_of_subqueries = await ctx.get("count_of_subqueries")

        # wait until we receive all events
        result = ctx.collect_events(ev, [InformationCheck] * number_of_subqueries)
        if result is None:
            return None
        # Add executed cypher statements to global state
        subqueries_cypher_history = await ctx.get("subqueries_cypher_history")
        new_subqueries_cypher = {
            item.subquery: {
                "cypher": item.cypher,
                "database_output": item.database_output,
            }
            for item in result
        }
        await ctx.set(
            "subqueries_cypher_history",
            {**subqueries_cypher_history, **new_subqueries_cypher},
        )

        original_question = await ctx.get("original_question")
        dynamic_notebook = await ctx.get("dynamic_notebook")
        plan = await ctx.get("plan")

        # Do the information check

        data = await information_check_step(
            self.llm, result, original_question, dynamic_notebook, plan
        )
        # Get count of information checks done
        information_checks = await ctx.get("information_checks")
        # Go fetch additional information if needed
        if data.get("modified_plan") and information_checks < MAX_INFORMATION_CHECKS:
            ctx.write_event_to_stream(
                StringEvent(
                    result=f"Modified plan: {data.get("modified_plan")}",
                    label="Modified plan",
                )
            )
            await ctx.set(
                "count_of_subqueries", len(data["modified_plan"][0])
            )  # this is used for ctx.collect()
            await ctx.set("dynamic_notebook", data["dynamic_notebook"])
            await ctx.set("plan", data.get("modified_plan")[1:])
            await ctx.set("information_checks", information_checks + 1)
            for subquery in data["modified_plan"][0]:
                ctx.send_event(GenerateCypher(subquery=subquery))
        else:
            return FinalAnswer(context=data["dynamic_notebook"])

    @step
    async def final_answer(self, ctx: Context, ev: FinalAnswer) -> StopEvent:
        original_question = await ctx.get("original_question")
        subqueries_cypher_history = await ctx.get("subqueries_cypher_history")
        # wait until we receive all events
        gen = await self.llm.astream_chat(
            final_answer_prompt.format_messages(
                context=ev.context, question=original_question
            )
        )
        final_event = StringEvent(result="", label="Final answer")
        async for response in gen:
            final_event.result = response.delta
            ctx.write_event_to_stream(final_event)
            await asyncio.sleep(0.05)
        return StopEvent(result=subqueries_cypher_history)
