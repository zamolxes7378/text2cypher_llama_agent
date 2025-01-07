import json
from typing import Any
from fastapi import Request


from app.settings import WORKFLOW_MAP


# function to force HTTPS
def https_url_for(request: Request, name: str, **path_params: Any) -> str:
    http_url = request.url_for(name, **path_params)
    return http_url.replace("http", "https", 1)


# run workflow that is present in app/workflows folder
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
