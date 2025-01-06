import json

from dotenv import load_dotenv
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.settings import WORKFLOW_MAP
from app.utils import run_workflow

load_dotenv()

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    workflows = list(WORKFLOW_MAP.keys())

    return templates.TemplateResponse(
        request=request, name="pages/index.html", context={"workflows": workflows}
    )


@app.post("/infer/")
async def infer(request: Request):
    form_data = await request.form()
    workflow = form_data["workflow"]
    context_input = form_data["context"]

    try:
        context = json.loads(context_input)
    except json.JSONDecodeError:
        context = {"input": context_input}

    return StreamingResponse(
        run_workflow(workflow=workflow, context=context),
        media_type="text/event-stream",
    )
