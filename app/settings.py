from app.workflows.joke import JokeWorkflow
from app.workflows.iterative_planner import IterativePlanningFlow

WORKFLOW_MAP = {
    'iterative_planning': IterativePlanningFlow,
    'joke': JokeWorkflow,
}
