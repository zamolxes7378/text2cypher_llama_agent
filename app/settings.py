from app.workflows.iterative_planner import IterativePlanningFlow
from app.workflows.joke import JokeWorkflow
from app.workflows.naive_text2cypher import NaiveText2CypherFlow

WORKFLOW_MAP = {
    "iterative planning": IterativePlanningFlow,
    "naive text2cypher": NaiveText2CypherFlow,
    "joke": JokeWorkflow,
}
