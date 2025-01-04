import os
import json
import markdown2
import functools
import asyncio
from openai import OpenAI
from fastapi import Request
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

    workflow_instance = workflow_class()

    handler = workflow_instance.run(**context)

    async for ev in handler.stream_events():
        if type(ev).__name__ != "StopEvent":
            yield f"data: {json.dumps({'event_type': type(ev).__name__, 'message': ev.result})}\n\n"

    result = await handler

    yield f"data: {json.dumps({'message': result})}\n\n"
