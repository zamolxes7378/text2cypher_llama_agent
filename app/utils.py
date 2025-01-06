import asyncio
import functools
import json
import os

import markdown2
from fastapi import Request
from openai import OpenAI

from app.settings import WORKFLOW_MAP


def is_partial_request(request: Request):
    if not request.headers.get("hx-request", None):
        raise HTTPException(
            status_code=status.HTTP_307_TEMPORARY_REDIRECT,
            headers={"Location": "/"},
        )


async def run_workflow(workflow: str, context: dict):
    workflow_class: Type[Workflow] = WORKFLOW_MAP.get(workflow)
    if not workflow_class:
        raise ValueError(f"Workflow '{workflow}' is not recognized.")

    workflow_instance = workflow_class(timeout=60)

    handler = workflow_instance.run(**context)

    async for ev in handler.stream_events():
        if type(ev).__name__ != "StopEvent":
            event_data = json.dumps(
                {
                    "uuid": str(ev.uuid),
                    "event_type": type(ev).__name__,
                    "result": ev.result,
                }
            )
            yield f"data: {event_data}\n\n"

    result = await handler

    yield f"data: {json.dumps({'uuid': None, 'result': result})}\n\n"
