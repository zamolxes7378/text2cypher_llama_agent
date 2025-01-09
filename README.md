# Text2Cypher LLama Agents

A collection of LlamaIndex Workflows-powered agents that convert natural language to Cypher queries designed to retrieve information from a Neo4j database to answer the question.

## 🎯 Features

- Multiple text2Cypher agents
- Built-in benchmarking suite
- Interactive web UI for testing
- Powered by LlamaIndex Workflows

## 🚀 Getting Started with web UI

### Prerequisites

1. Create `.env` file based on `.env.example`
```
cp .env.example .env
```
2. Edit `.env` and include your `OPENAI_API_KEY`.

### Installation

1. Install `uv` package manager:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
2. Install dependencies:
```bash
uv sync
```

### Run Development Server

Start the FastAPI server:
```bash
uv run fastapi dev
```

Open the `localhost:8000`

## 📊 Benchmarking

The `benchmark` directory contains:
- Test datasets
- Evaluation notebooks using Ragas

The benchmark can be evaluated against the `recommendations` database.

```
URI: neo4j+s://demo.neo4jlabs.com
username: recommendations
password: recommendations
database: recommendations
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
