# text2cypher_agent

This is a collection of text2cypher agents implements with LlamaIndex Workflows.

It is hardcoded access for public neo4j database `recommendations`:

```
URI: neo4j+s://demo.neo4jlabs.com
username: recommendations
password: recommendations
database: recommendations
```

## Run in development

### Set environment

Create `.env` file and add env variables as shown in `.env.example`

### Install uv and dependencies

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```
uv sync
```

```
uv run fastapi dev
```
