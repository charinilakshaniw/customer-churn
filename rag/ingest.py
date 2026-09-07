from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DOCS_DIR = PROJECT_ROOT / "rag" / "docs"
VECTORSTORE_DIR = PROJECT_ROOT / "rag" / "vectorstore"


def load_documents():

    documents = []

    for file_path in DOCS_DIR.glob("*.md"):

        text = file_path.read_text(
            encoding="utf-8"
        )

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": file_path.name
                }
            )
        )

    return documents


def build_vectorstore():

    print("Loading business documents...")

    documents = load_documents()

    print(
        f"Loaded {len(documents)} documents."
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(
        documents
    )

    print(
        f"Created {len(chunks)} chunks."
    )

    print("Loading embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Creating vector store...")

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(
            VECTORSTORE_DIR
        ),
        collection_name=(
            "customeriq_business_knowledge"
        )
    )

    print(
        "Vector store created successfully."
    )


if __name__ == "__main__":

    build_vectorstore()