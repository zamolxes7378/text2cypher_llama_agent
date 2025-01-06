from llama_index.core import ChatPromptTemplate

from app.workflows.utils import graph_store, llm

schema = graph_store.get_schema_str(exclude_types=["Actor", "Director"])


correct_cypher_system = """You are a Cypher expert reviewing a statement written by a junior developer. 
You need to correct the Cypher statement based on the provided errors. No pre-amble."
Do not wrap the response in any backticks or anything else. Respond with a Cypher statement only!"""

correct_cypher_user = """Check for invalid syntax or semantics and return a corrected Cypher statement.

Schema:
{schema}

Note: Do not include any explanations or apologies in your responses.
Do not wrap the response in any backticks or anything else.
Respond with a Cypher statement only!

Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.

The question is:
{question}

The Cypher statement is:
{cypher}

The errors are:
{errors}

Corrected Cypher statement: """

# Correct cypher
correct_cypher_msgs = [
    (
        "system",
        correct_cypher_system,
    ),
    ("user", correct_cypher_user),
]

correct_cypher_prompt = ChatPromptTemplate.from_messages(correct_cypher_msgs)


async def correct_cypher_step(subquery, cypher, errors):
    resp = await llm.achat(
        correct_cypher_prompt.format_messages(
            question=subquery, schema=schema, errors=errors
        )
    )
    return resp.message.content
