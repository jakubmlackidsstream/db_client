from typing import Dict, Any
from state import GraphState  # adjust import to your actual module


def check_schema_cache(state: GraphState) -> Dict[str, Any]:
    """
    Decide whether the database schema is already available in state.

    Sets `db_schema_ready` to:
      - True  if we have a non-None db_schema
      - False otherwise

    This node does not call any tools. It only inspects the current state and
    returns a partial update.
    """
    schema_ready = state.db_schema is not None

    return {
        "db_schema_ready": schema_ready,
    }
