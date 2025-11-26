# nodes/schema_explainer.py

import json
from typing import Dict, Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage, AnyMessage

from state import GraphState  # adjust import to your path


def make_schema_explainer_node(llm: BaseChatModel):
    """
    Factory that creates a schema_explainer node.

    It answers questions about the database structure:
    - what tables exist
    - what columns a table has
    - primary keys, etc.

    Reads:
      - user_query
      - db_schema        (structured from schema_introspector)
      - db_schema_summary

    Writes:
      - final_answer
    """

    def schema_explainer(state: GraphState) -> Dict[str, Any]:
        user_query = (state.user_query or "").strip()

        # If we somehow don't have schema yet, fail gracefully
        if state.db_schema is None:
            msg = (
                "I don't have any database schema information loaded yet, "
                "so I can't answer schema-related questions."
            )
            return {"final_answer": msg}

        schema_summary = state.db_schema_summary or "No textual summary available."
        schema_struct = state.db_schema  # this is the dict from schema_introspector

        system_prompt = (
            "You are an assistant that explains the structure of a SQLite database.\n\n"
            "You will be given:\n"
            "- A natural language question about the database schema.\n"
            "- A human-readable summary of the schema.\n"
            "- A structured JSON representation of the schema produced by introspection.\n\n"
            "Your task is to answer questions about:\n"
            "- Which tables exist.\n"
            "- What columns a table has (and their types).\n"
            "- Primary keys or NOT NULL flags when relevant.\n\n"
            "Guidelines:\n"
            "- Base your answers ONLY on the provided schema; do not make up tables or columns.\n"
            "- If the user asks about a table or column that does not exist, say so explicitly.\n"
            "- Keep answers concise but clear; use bullet lists or small tables when useful."
        )

        # We include both the textual summary and the JSON so the model has detail + structure
        user_prompt = (
            f"User's schema question:\n{user_query}\n\n"
            f"Schema summary:\n{schema_summary}\n\n"
            "Structured schema (JSON):\n"
            f"{json.dumps(schema_struct, indent=2)}"
        )

        messages: list[AnyMessage] = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        resp = llm.invoke(messages)
        text = resp.content if hasattr(resp, "content") else str(resp)

        return {"final_answer": text}

    return schema_explainer
