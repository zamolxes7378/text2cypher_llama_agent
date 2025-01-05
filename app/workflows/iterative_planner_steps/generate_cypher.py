from llama_index.core.schema import TextNode
from llama_index.core import VectorStoreIndex
from llama_index.core import ChatPromptTemplate

from app.workflows.utils import llm, graph_store, embed_model

examples = [
    {
        "question": "How many artists are there?",
        "query": "MATCH (a:Person)-[:ACTED_IN]->(:Movie) RETURN count(DISTINCT a)",
    },
    {
        "question": "Which actors played in the movie Casino?",
        "query": "MATCH (m:Movie {title: 'Casino'})<-[:ACTED_IN]-(a) RETURN a.name",
    },
    {
        "question": "How many movies has Tom Hanks acted in?",
        "query": "MATCH (a:Person {name: 'Tom Hanks'})-[:ACTED_IN]->(m:Movie) RETURN count(m)",
    },
    {
        "question": "List all the genres of the movie Schindler's List",
        "query": "MATCH (m:Movie {title: 'Schindler's List'})-[:IN_GENRE]->(g:Genre) RETURN g.name",
    },
    {
        "question": "Which actors have worked in movies from both the comedy and action genres?",
        "query": "MATCH (a:Person)-[:ACTED_IN]->(:Movie)-[:IN_GENRE]->(g1:Genre), (a)-[:ACTED_IN]->(:Movie)-[:IN_GENRE]->(g2:Genre) WHERE g1.name = 'Comedy' AND g2.name = 'Action' RETURN DISTINCT a.name",
    },
    {
        "question": "Which directors have made movies with at least three different actors named 'John'?",
        "query": "MATCH (d:Person)-[:DIRECTED]->(m:Movie)<-[:ACTED_IN]-(a:Person) WHERE a.name STARTS WITH 'John' WITH d, COUNT(DISTINCT a) AS JohnsCount WHERE JohnsCount >= 3 RETURN d.name",
    },
    {
        "question": "Identify movies where directors also played a role in the film.",
        "query": "MATCH (p:Person)-[:DIRECTED]->(m:Movie), (p)-[:ACTED_IN]->(m) RETURN m.title, p.name",
    },
    {
        "question": "Find the actor with the highest number of movies in the database.",
        "query": "MATCH (a:Actor)-[:ACTED_IN]->(m:Movie) RETURN a.name, COUNT(m) AS movieCount ORDER BY movieCount DESC LIMIT 1",
    },
]

few_shot_nodes = []
for line in examples:
    few_shot_nodes.append(TextNode(text=f"{{'query':{line['query']}, 'question': {line['question']}))"))

few_shot_index = VectorStoreIndex(few_shot_nodes, embed_model=embed_model)
few_shot_retriever = few_shot_index.as_retriever(similarity_top_k=5)


def get_fewshots(question):
    return [el.text for el in few_shot_retriever.retrieve(question)]

generate_system = """Given an input question, convert it to a Cypher query. No pre-amble.
Do not wrap the response in any backticks or anything else. Respond with a Cypher statement only!"""

generate_user = """You are a Neo4j expert. Given an input question, create a syntactically correct Cypher query to run.
Do not wrap the response in any backticks or anything else. Respond with a Cypher statement only!
Here is the schema information
{schema}

Below are a number of examples of questions and their corresponding Cypher queries.

{fewshot_examples}

User input: {question}
Cypher query:"""

generate_cypher_msgs = [
    (
        "system",
        generate_system,
    ),
    ("user", generate_user),
]

text2cypher_prompt = ChatPromptTemplate.from_messages(generate_cypher_msgs)

schema = graph_store.get_schema_str(exclude_types=["Actor", "Director"])

async def generate_cypher_step(subquery):
    fewshot_examples = get_fewshots(subquery)
    resp = await llm.achat(text2cypher_prompt.format_messages(question=subquery, schema=schema, fewshot_examples=fewshot_examples))
    return resp.message.content