from typing import Dict, Any, List
from state import GraphState  # adjust import to your module


def schema_normalizer(state: GraphState) -> Dict[str, Any]:
    """
    Normalize the raw DB schema into a form that's easy for the LLM to consume.

    Inputs:
      - db_schema_raw: {
            "tables": [
                {
                    "name": str,
                    "columns": [
                        {
                            "name": str,
                            "type": str,
                            "notnull": bool,
                            "default_value": Any,
                            "pk": bool,
                        },
                        ...
                    ],
                },
                ...
            ]
        }

    Outputs (partial state):
      - db_schema: same structured schema (for programmatic use)
      - db_schema_summary: textual summary (for LLM prompts)
      - db_schema_ready: True
    """
    raw = state.db_schema_raw or {}

    tables = raw.get("tables", [])
    if not isinstance(tables, list):
        tables = []

    # Build a compact text summary for the LLM
    summary_lines: List[str] = []
    summary_lines.append("Database schema overview:")

    for table in tables:
        table_name = table.get("name", "<unknown_table>")
        columns = table.get("columns", [])
        col_parts: List[str] = []

        for col in columns:
            col_name = col.get("name", "<col>")
            col_type = col.get("type", "TEXT") or "TEXT"
            pk = col.get("pk", False)
            notnull = col.get("notnull", False)

            col_desc = f"{col_name} {col_type}"
            flags: List[str] = []
            if pk:
                flags.append("PK")
            if notnull and not pk:
                flags.append("NOT NULL")
            if flags:
                col_desc += " (" + ", ".join(flags) + ")"

            col_parts.append(col_desc)

        joined_cols = ", ".join(col_parts) if col_parts else "<no columns?>"
        summary_lines.append(f"- Table {table_name}: {joined_cols}")

    db_schema_summary = "\n".join(summary_lines)

    return {
        "db_schema": raw,                # Already structured nicely from introspector
        "db_schema_summary": db_schema_summary,
        "db_schema_ready": True,
    }
