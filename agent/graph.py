from langgraph.graph import StateGraph, START, END

from agent.state import CustomerIQState
from agent.tools import TOOL_FUNCTIONS
from services.gemini_agent_service import start_agent, continue_agent


def extract_tool_calls(interaction):
    """Extract function calls requested by Gemini."""

    pending_tool_calls = []

    for step in interaction.steps:
        if step.type == "function_call":
            pending_tool_calls.append(
                {
                    "id": step.id,
                    "name": step.name,
                    "arguments": step.arguments,
                }
            )

    return pending_tool_calls


def call_gemini(state: CustomerIQState) -> CustomerIQState:

    try:
        interaction = start_agent(state["user_query"])

        pending_tool_calls = extract_tool_calls(interaction)

        if not pending_tool_calls:
            return {
                **state,
                "interaction_id": interaction.id,
                "final_response": interaction.output_text,
                "tools_used": [],
            }

        tools_used = [
            tool_call["name"]
            for tool_call in pending_tool_calls
        ]

        return {
            **state,
            "interaction_id": interaction.id,
            "pending_tool_calls": pending_tool_calls,
            "tools_used": tools_used,
        }

    except Exception as error:

        return {
            **state,
            "error": str(error),
            "final_response": (
                "I couldn't process the request right now. "
                "Please try again."
            ),
        }


def execute_tools(state: CustomerIQState) -> CustomerIQState:

    function_results = []

    for tool_call in state.get("pending_tool_calls", []):

        tool_name = tool_call["name"]
        arguments = tool_call["arguments"]

        tool_function = TOOL_FUNCTIONS.get(tool_name)

        if tool_function is None:

            result = {
                "error": f"Unknown tool: {tool_name}"
            }

        else:

            try:
                result = tool_function(**arguments)

            except Exception as error:

                result = {
                    "error": (
                        f"Tool '{tool_name}' failed: "
                        f"{str(error)}"
                    )
                }

        function_results.append(
            {
                "type": "function_result",
                "name": tool_name,
                "call_id": tool_call["id"],
                "result": [
                    {
                        "type": "text",
                        "text": str(result),
                    }
                ],
            }
        )

    return {
        **state,
        "pending_tool_calls": [],
        "function_results": function_results,
    }


def continue_gemini(state: CustomerIQState) -> CustomerIQState:

    try:

        interaction = continue_agent(
            interaction_id=state["interaction_id"],
            function_results=state["function_results"],
        )

        pending_tool_calls = extract_tool_calls(interaction)

        if pending_tool_calls:

            existing_tools = state.get("tools_used", [])

            new_tools = [
                tool_call["name"]
                for tool_call in pending_tool_calls
            ]

            return {
                **state,
                "interaction_id": interaction.id,
                "pending_tool_calls": pending_tool_calls,
                "tools_used": existing_tools + new_tools,
            }

        return {
            **state,
            "interaction_id": interaction.id,
            "final_response": interaction.output_text,
        }

    except Exception as error:

        return {
            **state,
            "error": str(error),
            "final_response": (
                "I couldn't complete the analysis right now. "
                "Please try again."
            ),
        }


def should_continue(state: CustomerIQState):

    if state.get("pending_tool_calls"):
        return "execute_tools"

    return END


def build_graph():

    graph = StateGraph(CustomerIQState)

    graph.add_node("call_gemini", call_gemini)
    graph.add_node("execute_tools", execute_tools)
    graph.add_node("continue_gemini", continue_gemini)

    graph.add_edge(
        START,
        "call_gemini"
    )

    graph.add_conditional_edges(
        "call_gemini",
        should_continue,
        {
            "execute_tools": "execute_tools",
            END: END,
        },
    )

    graph.add_edge(
        "execute_tools",
        "continue_gemini"
    )

    graph.add_conditional_edges(
        "continue_gemini",
        should_continue,
        {
            "execute_tools": "execute_tools",
            END: END,
        },
    )

    return graph.compile()


customeriq_graph = build_graph()