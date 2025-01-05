from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI


import os

os.environ["OPENAI_API_KEY"] = "sk-"

graph_store = Neo4jPropertyGraphStore(
    username="recommendations",
    password="recommendations",
    database="recommendations",
    url="neo4j+s://demo.neo4jlabs.com:7687",
    enhanced_schema=True,
    create_indexes=False,
    timeout=10
)
"""
llm = Gemini(
    model="models/gemini-1.5-flash",
    api_key=""
)
"""
llm = OpenAI(model="gpt-4o-2024-11-20", temperature=0)

embed_model = OpenAIEmbedding(model="text-embedding-3-small")

from llama_index.graph_stores.neo4j import CypherQueryCorrector, Schema

# Cypher query corrector is experimental
corrector_schema = [
    Schema(el["start"], el["type"], el["end"])
    for el in graph_store.get_schema().get("relationships")
]
cypher_query_corrector = CypherQueryCorrector(corrector_schema)