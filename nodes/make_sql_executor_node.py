import sqlite3
from pathlib import Path
from typing import Dict, Any, List, Optional

from state import GraphState  # adjust import to your module


def make_sql_executor_node(db_path: str):
    """
    Factory that creates a SQL executor node bound to a specific SQLite DB.

    db_path: path to your SQLite file, e.g. "demo.db"
    """

    db_path = str(Path(db_path).resolve())

    def sql_executor_tool(state: GraphState) -> Dict[str, Any]:
        """
        Execute the generated SQL query against the SQLite database.

        Reads:
          - sql_query

        Writes:
          - query_result: list[dict] with column -> value
          - query_metadata: { "row_count": int, "columns": list[str] }
          - execution_error: str | None
        """
        sql_query: Optional[str] = state.sql_query

        # Safety fallback: no query to run
        if not sql_query or not sql_query.strip():
            return {
                "query_result": [],
                "query_metadata": {"row_count": 0, "columns": []},
                "execution_error": "No SQL query provided to execute.",
            }

        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()

            cur.execute(sql_query)
            rows = cur.fetchall()

            # Convert sqlite3.Row objects to plain dicts
            result: List[Dict[str, Any]] = [dict(row) for row in rows]

            # Column names
            columns = [desc[0] for desc in cur.description] if cur.description else []

            conn.close()

            return {
                "query_result": result,
                "query_metadata": {
                    "row_count": len(result),
                    "columns": columns,
                },
                "execution_error": None,
            }

        except Exception as e:
            # On error, we return the message and clear result/metadata
            return {
                "query_result": [],
                "query_metadata": {"row_count": 0, "columns": []},
                "execution_error": str(e),
            }

    return sql_executor_tool
