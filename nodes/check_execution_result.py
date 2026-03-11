from typing import Dict, Any

from state import GraphState


def check_execution_result(state: GraphState) -> Dict[str, Any]:
    """
    Inspect the SQL executor output and set result_status accordingly.

    Reads:
      - execution_error

    Writes:
      - result_status: "success" | "error"
    """
    if state.execution_error:
        return {"result_status": "error"}
    return {"result_status": "success"}
