import os
from typing import List
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings


def load_environment() -> None:
    """Load environment variables."""
    load_dotenv()

    if not os.getenv("GROQ_API_KEY"):
        raise EnvironmentError("GROQ_API_KEY is not set")

    if not os.getenv("HF_TOKEN"):
        raise EnvironmentError("HF_TOKEN is not set")

    os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")


def create_documents() -> List[Document]:
    """Create sample documents."""
    return [
        Document(
            page_content="Dogs are great companions, known for their loyalty and friendliness.",
            metadata={"source": "mammal-pets-docs"},
        ),
        Document(
            page_content="Cats are independent pets that often enjoy their own space.",
            metadata={"source": "mammal-pets-docs"},
        ),
        Document(
            page_content="Parrots are intelligent birds capable of mimicking human speech.",
            metadata={"source": "bird-pets-docs"},
        ),
        Document(
            page_content="Goldfish are popular pets for beginners, requiring relatively simple care.",
            metadata={"source": "fish-pets-docs"},
        ),
        Document(
            page_content="Rabbits are social animals that need plenty of space to hop around.",
            metadata={"source": "mammal-pets-docs"},
        ),
    ]


def main() -> None:
    load_environment()

    # Initialize LLM
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )

    # Create documents
    documents = create_documents()

    # Embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    # Vector store
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
    )

    # Retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 2},
    )

    # Prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "human",
                """
Answer the question using only the provided context.

Question:
{question}

Context:
{context}
""",
            )
        ]
    )

    # RAG chain
    rag_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
    )

    # Invoke chain
    response = rag_chain.invoke("Tell me about cats")

    print("\n--- RAG RESPONSE ---")
    print(response.content)


if __name__ == "__main__":
    main()
