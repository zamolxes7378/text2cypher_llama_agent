from app.workflows.joke import JokeWorkflow

# from app.workflows.iterative_planner import IterativePlanningFlow
from app.workflows.naive_text2cypher import NaiveText2CypherFlow

# from app.workflows.naive_text2cypher_retry import NaiveText2CypherRetryFlow

WORKFLOW_MAP = {
    # "iterative planning": IterativePlanningFlow,
    "naive_text2cypher": NaiveText2CypherFlow,
    # "naive text2cypher with 1 retry": NaiveText2CypherRetryFlow,
    "joke": JokeWorkflow,
}
