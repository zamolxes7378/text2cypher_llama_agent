import json
from typing import Union

from dotenv import load_dotenv
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from app.settings import WORKFLOW_MAP
from app.utils import run_workflow, urlx_for

load_dotenv()

templates = Jinja2Templates(directory="templates")
templates.env.globals["url_for"] = urlx_for

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    workflows = list(WORKFLOW_MAP.keys())

    return templates.TemplateResponse(
        request=request, name="pages/index.html", context={"workflows": workflows}
    )


class InferPayload(BaseModel):
    workflow: str
    context: str


@app.post("/infer/")
async def infer(payload: InferPayload):
    workflow = payload.workflow
    context_input = payload.context

    try:
        context = json.loads(context_input)
    except json.JSONDecodeError:
        context = {"input": context_input}

    print("HERE context", context)

    return StreamingResponse(
        run_workflow(workflow=workflow, context=context),
        media_type="text/event-stream",
    )
