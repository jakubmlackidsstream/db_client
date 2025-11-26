import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

from state import GraphState  # adjust to your actual path


def make_schema_introspector_node(db_path: str):
    """
    Factory that creates a schema_introspector node bound to a specific SQLite DB.

    db_path: path to your SQLite file, e.g. "demo.db"
    """

    db_path = str(Path(db_path).resolve())

    def schema_introspector(state: GraphState) -> Dict[str, Any]:
        """
        Introspect the SQLite database and return a raw schema structure.

        Writes:
          - db_schema_raw: {
                "tables": [
                    {
                        "name": "table_name",
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
          - db_schema_last_updated: ISO timestamp string

        Does NOT set db_schema or db_schema_ready; a later node
        (schema_normalizer.py) will do that.
        """
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        # 1) Get table names (ignore SQLite internal tables)
        cur.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type='table'
              AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """
        )
        table_rows = cur.fetchall()
        table_names: List[str] = [row["name"] for row in table_rows]

        tables_schema: List[Dict[str, Any]] = []

        # 2) For each table, get column info via PRAGMA
        for table_name in table_names:
            cur.execute(f"PRAGMA table_info('{table_name}')")
            col_rows = cur.fetchall()

            columns: List[Dict[str, Any]] = []
            for col in col_rows:
                # PRAGMA table_info returns:
                # cid, name, type, notnull, dflt_value, pk
                columns.append(
                    {
                        "name": col["name"],
                        "type": col["type"],
                        "notnull": bool(col["notnull"]),
                        "default_value": col["dflt_value"],
                        "pk": bool(col["pk"]),
                    }
                )

            tables_schema.append(
                {
                    "name": table_name,
                    "columns": columns,
                }
            )

        conn.close()

        db_schema_raw: Dict[str, Any] = {
            "tables": tables_schema,
        }

        return {
            "db_schema_raw": db_schema_raw,
            "db_schema_last_updated": datetime.utcnow().isoformat(),
        }

    return schema_introspector
