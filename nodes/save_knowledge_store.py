import json
from pathlib import Path
from typing import Dict, Any

from state import GraphState


def make_save_knowledge_store_node(knowledge_path: str = "knowledge.json"):
    """
    Factory that creates a save_knowledge_store node.

    Merges the current state into the persisted knowledge file:
      - known_terms   (always)
      - db_schema / db_schema_summary / db_schema_last_updated
        (when a schema is present in state)

    Register this factory twice in the graph with different names:
      save_schema  – called after schema_normalizer
      save_terms   – called after store_clarification
    Both use the same file and same logic; only their graph positions differ.
    """
    path = Path(knowledge_path)

    def save_knowledge_store(state: GraphState) -> Dict[str, Any]:
        existing: Dict[str, Any] = {}
        if path.exists():
            with open(path) as f:
                existing = json.load(f)

        existing["known_terms"] = state.known_terms or {}

        schema = state.db_schema
        has_tables = (
            isinstance(schema, dict) and bool(schema.get("tables"))
        )
        if has_tables:
            existing["db_schema"] = schema
            existing["db_schema_summary"] = state.db_schema_summary
            existing["db_schema_last_updated"] = (
                state.db_schema_last_updated
            )

        with open(path, "w") as f:
            json.dump(existing, f, indent=2, default=str)

        return {}

    return save_knowledge_store
