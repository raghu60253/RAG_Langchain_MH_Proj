import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field, field_validator

BASE_DIR = Path(__file__).resolve().parent
COLLECTION_NAME = "mh_document_huggingface_embeddings"

load_dotenv(BASE_DIR / ".env")


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else BASE_DIR / path


def create_rag_chain() -> Any:
    api_key = os.getenv("OLLAMA_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Set OLLAMA_API_KEY in the environment or .env file.")

    pdf_path = project_path(os.getenv("PDF_PATH", "AST-2.pdf"))
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    chroma_dir = project_path(os.getenv("CHROMA_DB_DIR", "chroma_db"))
    embedding_model = HuggingFaceEmbeddings(
        model_name=os.getenv(
            "HF_EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        ),
    )
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=str(chroma_dir),
    )

    if vector_store._collection.count() == 0:
        pages = PyPDFLoader(str(pdf_path)).load()
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        ).split_documents(pages)
        if not chunks:
            raise ValueError(f"No readable text found in PDF: {pdf_path}")
        vector_store.add_documents(chunks)

    llm = ChatOpenAI(
        model=os.getenv("OLLAMA_CHAT_MODEL", "gemma4:31b"),
        base_url=os.getenv("OLLAMA_BASE_URL", "https://ollama.com/v1"),
        api_key=api_key,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are a question-answering assistant.
                Answer using only the supplied context.
                Do not use outside knowledge or invent answers.
                If the answer is not in the context, say
                "Answer is not in the context".
                Keep the answer concise.
                Context:
                {context}
                """,
            ),
            ("human", "{input}"),
        ]
    )
    document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    return create_retrieval_chain(
        retriever=vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3},
        ),
        combine_docs_chain=document_chain,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.rag_chain = create_rag_chain()
    yield


app = FastAPI(title="PDF RAG API", version="1.0.0", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Question must not be blank.")
        return value


class SourceDocument(BaseModel):
    page: int | None
    content: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceDocument]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(body: AskRequest, request: Request) -> AskResponse:
    result = request.app.state.rag_chain.invoke({"input": body.question})
    sources = [
        SourceDocument(
            page=document.metadata.get("page", 0) + 1,
            content=document.page_content,
        )
        for document in result["context"]
    ]
    return AskResponse(answer=result["answer"], sources=sources)
