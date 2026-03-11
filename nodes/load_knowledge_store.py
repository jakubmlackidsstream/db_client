import json
from pathlib import Path
from typing import Dict, Any

from state import GraphState


def make_load_knowledge_store_node(knowledge_path: str = "knowledge.json"):
    """
    Factory that creates a load_knowledge_store node.

    On every session start it reads the persisted knowledge file and
    hydrates state with:
      - known_terms  (always merged in)
      - db_schema / db_schema_summary / db_schema_last_updated
        (only when state has no schema yet)

    This makes the schema cache survive process restarts and carries
    user-defined terminology across conversations.
    """
    path = Path(knowledge_path)

    def load_knowledge_store(state: GraphState) -> Dict[str, Any]:
        if not path.exists():
            return {}

        with open(path) as f:
            data = json.load(f)

        update: Dict[str, Any] = {}

        if data.get("known_terms"):
            update["known_terms"] = data["known_terms"]

        # Only restore schema if state is empty and stored schema has tables
        stored_schema = data.get("db_schema")
        has_tables = (
            isinstance(stored_schema, dict)
            and bool(stored_schema.get("tables"))
        )
        if state.db_schema is None and has_tables:
            update["db_schema"] = stored_schema
            update["db_schema_summary"] = data.get("db_schema_summary")
            update["db_schema_last_updated"] = data.get(
                "db_schema_last_updated"
            )

        return update

    return load_knowledge_store
