import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


load_dotenv()


def get_llm():

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set. "
            "Add it to your .env file."
        )

    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )


def generate_answer(
    question: str,
    context: str
):

    llm = get_llm()

    prompt = f"""
You are CustomerIQ, an AI customer intelligence assistant.

Answer the user's question using ONLY the business
knowledge provided below.

If the information is not available in the business
knowledge, say that you do not have enough information.

Do not invent company policies or recommendations.

Business knowledge:
-------------------
{context}
-------------------

User question:
{question}

Provide a concise, professional business answer.
"""

    response = llm.invoke(prompt)

    return response.content