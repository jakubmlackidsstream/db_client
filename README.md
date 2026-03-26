# DB QA Agent

A conversational agent for querying a SQLite database using natural language. Built with LangGraph and served via a Gradio chat interface. The agent translates user questions into SQL, executes them, and when a query fails it automatically repairs the SQL and retries — without user intervention. It supports follow-up questions, domain term definitions, and schema introspection, and persists the database schema and known terms across sessions in a local `knowledge.json` file.

The agent is provider-agnostic: OpenAI, Anthropic, and Google Gemini are all supported and can be swapped with a single line change.

## Environment setup

The project uses [uv](https://docs.astral.sh/uv/) for dependency management.

### Prerequisites

- Python 3.12
- [uv](https://docs.astral.sh/uv/getting-started/installation/) installed

### Install dependencies

```bash
uv sync
```

### Configure API keys

Create a `.env` file in the project root and add the key for your chosen provider:

```bash
# OpenAI (default)
OPENAI_API_KEY=sk-...

# Anthropic
ANTHROPIC_API_KEY=sk-ant-...

# Google Gemini
GOOGLE_API_KEY=...
```

### Run

Launch JupyterLab and open `langgraph.ipynb`:

```bash
uv run jupyter lab
```

Run all cells — the last cell starts the Gradio server at `http://127.0.0.1:7860`.

To switch LLM provider, change the `PROVIDER` variable in the notebook:

```python
PROVIDER = Provider.OPENAI       # default
# PROVIDER = Provider.ANTHROPIC
# PROVIDER = Provider.GOOGLE
```
