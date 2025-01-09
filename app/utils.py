import asyncio
import json
from typing import Any, Type
from fastapi import Request
from llama_index.core.workflow import Workflow
from jinja2 import pass_context

from app.settings import WORKFLOW_MAP


# function to force HTTPS
@pass_context
def urlx_for(
    context: dict,
    name: str,
    **path_params: Any,
) -> str:
    request: Request = context["request"]
    http_url = request.url_for(name, **path_params)
    if scheme := request.headers.get("x-forwarded-proto"):
        return http_url.replace(scheme=scheme)
    return http_url


# run workflow that is present in app/workflows folder
async def run_workflow(workflow: str, context: dict):
    workflow_class: Type[Workflow] = WORKFLOW_MAP.get(workflow)
    if not workflow_class:
        raise ValueError(f"Workflow '{workflow}' is not recognized.")

    workflow_instance = workflow_class(timeout=60)

    handler = workflow_instance.run(**context)

    async for event in handler.stream_events():
        if type(event).__name__ != "StopEvent":
            event_data = json.dumps(
                {
                    "event_type": type(event).__name__,
                    "label": event.label,
                    "message": event.message,
                }
            )
            yield f"data: {event_data}\n\n"

    result = await handler

    yield f"data: {json.dumps({'result': result})}\n\n"
