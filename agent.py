import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
# from openai import OpenAI
# from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_core.vectorstores import InMemoryVectorStore
# from langchain_core.messages import AIMessageChunk
# from langchain.tools import tool
# from langchain.agents import create_agent
# from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_chroma import chroma

load_dotenv()
ollama_api_key = os.getenv("OLLAMA_API_KEY")
if not ollama_api_key:
    raise RuntimeError("Set OLLAMA_API_KEY in the environment or .env file.")

llm = ChatOpenAI(
    model=os.getenv("OLLAMA_CHAT_MODEL", "gemma4:31b"),
    base_url="https://ollama.com/v1",
    api_key=ollama_api_key,
)

pdf_path = "AST-2.pdf"
loader = PyPDFLoader(pdf_path);
pages = loader.load()
print("Number of pages in the PDF:", len(pages))
print(pages[0].page_content[:500])  # Print the first 500 characters of the first page

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(pages)
print("Number of chunks created:", len(chunks))
embedding_model = HuggingFaceEmbeddings(
    model_name=os.getenv(
        "HF_EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    ),
)
#Sample_text = "what is the GDP of india?"
#sample_embedding = embedding_model.embed_query(Sample_text)
#print(len(sample_embedding))
#print(sample_embedding[:1000])  # Print the first 10 dimensions of the embedding

vector_store = Chroma.from_documents(
    chunks,
    embedding_model,
    collection_name="mh_document_huggingface_embeddings",
    persist_directory="./chroma_db",
)
print("chunks stored in vector store:", vector_store._collection.count())

#search_query = "What is the GDP of India?"
#retrieved_docs = vector_store.similarity_search(search_query, k=3)
#print("Retrieved documents:",len(retrieved_docs))

#create retriever
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
print(retriever)

#create RAG prompt
rag_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", 
         """
        you are a question answering assistant.
        Answer the question using only from the supplied context.
        Rules:
        1. Do not use outside knowledge.
        2. Do not invent answers.
        3. if the answer is not in the context, say "Answer is not in the context".
        4. Keep the answer concise and to the point.
        Context:
        {context}
         """
         ),
        ("human", "{input}"),
    ]
)

# create Document chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
document_chain = create_stuff_documents_chain(llm=llm,prompt= rag_prompt)

from langchain_classic.chains import create_retrieval_chain
rag_chain = create_retrieval_chain(
    retriever=retriever,
    combine_docs_chain=document_chain,
)
Question = "Explain about manifest.in?"
result=rag_chain.invoke({"input": Question})

#print("Answer:", result['answer'])
#print("Source Documents:", result['context'])

# Previous OpenAI-compatible client example:
# load_dotenv()
# client = OpenAI(
#     base_url="http://127.0.0.1:8888/v1",
#     api_key=os.getenv("OPENAI_API_KEY")
# )
# response = client.chat.completions.create(
#     model="gpt-4o-mini",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "Hello! Can you help me with a question?"}],
#         stream=True,
# )
# for chunk in response:
#     print(chunk.choices[0].delta, end="", flush=True)