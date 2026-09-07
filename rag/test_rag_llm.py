from rag.rag_service import retrieve_business_knowledge
from services.llm_service import generate_answer


question = "What should we do with a high risk customer?"


results = retrieve_business_knowledge(
    question,
    k=3
)


context = "\n\n".join(
    result["content"]
    for result in results
)


answer = generate_answer(
    question,
    context
)


print("\nQUESTION:")
print(question)

print("\nRETRIEVED KNOWLEDGE:")
print(context)

print("\nAI ANSWER:")
print(answer)