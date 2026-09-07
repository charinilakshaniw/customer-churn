from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VECTORSTORE_DIR = (
    PROJECT_ROOT
    / "rag"
    / "vectorstore"
)


def get_vectorstore():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma(
        persist_directory=str(VECTORSTORE_DIR),
        embedding_function=embeddings,
        collection_name="customeriq_business_knowledge"
    )

    return vectorstore


def retrieve_business_knowledge(
    query: str,
    k: int = 3
):

    vectorstore = get_vectorstore()

    documents = vectorstore.similarity_search(
        query,
        k=k
    )

    results = []

    for document in documents:

        results.append(
            {
                "content": document.page_content,
                "source": document.metadata.get(
                    "source",
                    "unknown"
                )
            }
        )

    return results