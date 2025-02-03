from llama_index.core import ChatPromptTemplate

GENERATE_SYSTEM_TEMPLATE = """Given an input question, convert it to a Cypher query. No pre-amble.
Do not wrap the response in any backticks or anything else. Respond with a Cypher statement only!"""

GENERATE_USER_TEMPLATE = """You are a Neo4j expert. Given an input question, create a syntactically correct Cypher query to run. Additionally, determine whether the output should be visualized, and if so, specify the visualization type (e.g., graph, table, chart). 

Here is the schema information:
{schema}

User input: {question}

Response format:
<cypher>Cypher statement</cypher>
<visualization><"none" | "graph" | "table" | "chart"></visualization>"""


async def generate_cypher_step(llm, graph_store, subquery, fewshot_examples):
    # Remove multilabeled nodes
    schema = graph_store.get_schema_str(exclude_types=["Actor", "Director"])
    generate_cypher_msgs = [
        ("system", GENERATE_SYSTEM_TEMPLATE),
        ("user", GENERATE_USER_TEMPLATE),
    ]
    text2cypher_prompt = ChatPromptTemplate.from_messages(generate_cypher_msgs)

    response = await llm.achat(
        text2cypher_prompt.format_messages(
            question=subquery, schema=schema, fewshot_examples=fewshot_examples
        )
    )

    return response.message.content
