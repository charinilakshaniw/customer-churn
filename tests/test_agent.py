from agent.graph import customeriq_graph


result = customeriq_graph.invoke(
    {
        "user_query": (
            "Analyse customer C0001 "
            "and recommend what we should do."
        )
    }
)


print()
print("=" * 70)
print("CUSTOMERIQ AGENT")
print("=" * 70)

print(
    result.get(
        "final_response",
        "No response generated."
    )
)

print("=" * 70)