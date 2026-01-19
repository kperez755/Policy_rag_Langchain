import os
import chromadb
import uuid
from chromadb.config import Settings
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

load_dotenv()


os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
CHROMA_TENANT = os.getenv("CHROMA_TENANT")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME")


chroma_client = chromadb.CloudClient(
    tenant=CHROMA_TENANT,
    database=CHROMA_DATABASE,
    api_key=CHROMA_API_KEY
)

collection = chroma_client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)

SOURCE_DIR = "files"
CHUNKED_DIR = os.path.join(SOURCE_DIR, "chunked")

def read_file_text(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        reader = PdfReader(file_path)
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        return "\n".join(pages).strip()
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read().strip()

def ingest_data():
    if not os.path.exists(CHUNKED_DIR):
        os.makedirs(CHUNKED_DIR)

    if not os.path.exists(SOURCE_DIR):
        return

    source_files = [
        f for f in os.listdir(SOURCE_DIR)
        if os.path.isfile(os.path.join(SOURCE_DIR, f))
    ]

    splitter = RecursiveCharacterTextSplitter(
        separators=[". ", "? ", "! ", "\n", " ", ""],
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False
    )

    for file_name in source_files:
        file_path = os.path.join(SOURCE_DIR, file_name)
        text = read_file_text(file_path)
        if not text:
            continue

        chunks = splitter.split_text(text)
        base = os.path.splitext(file_name)[0]

        for i, chunk in enumerate(chunks):
            name = f"ch{i+1}-{base}-len{len(chunk)}.txt"
            with open(os.path.join(CHUNKED_DIR, name), "w", encoding="utf-8") as f:
                f.write(chunk)

    documents = []
    metadatas = []
    ids = []

    for file_name in os.listdir(CHUNKED_DIR):
        if not file_name.endswith(".txt"):
            continue

        file_path = os.path.join(CHUNKED_DIR, file_name)
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        name_no_ext = os.path.splitext(file_name)[0]
        parts = name_no_ext.split("-")
        chunk_part = int(parts[0].replace("ch", ""))
        size = int(parts[-1].replace("len", ""))
        original_filename = "-".join(parts[1:-1])

        documents.append(content)
        metadatas.append({
            "source": file_name,
            "file_name": original_filename,
            "chunk_part": chunk_part,
            "size": size
        })
        ids.append(str(uuid.uuid4()))

    if documents:
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

@tool
def retrieve_documents(query, n_results=5, threshold=1.5, filter_by=None):

    """
    Retrieve relevant documents from ChromaDB using semantic similarity.

    Args:
        query: Natural language search query.
        n_results: Maximum number of candidate results to fetch.
        threshold: Distance threshold for filtering results (lower is more similar).
        filter_by: Optional metadata filter dictionary.

    Returns:
        A list of dictionaries containing text, source, and distance.
    """
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=filter_by,
        include=["documents", "metadatas", "distances"]
    )

    docs = []
    if results["documents"] and results["documents"][0]:
        for text, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            if dist <= threshold:
                docs.append({
                    "text": text,
                    "source": meta.get("source"),
                    "distance": dist
                })
    return docs

llm = ChatOpenAI(model="gpt-5", temperature=0.1)

def llm_answer(prompt_text: str) -> str:
    return llm.invoke(prompt_text).content

def generate_answer(query: str) -> str:
    docs = retrieve_documents(query, n_results=5, threshold=1.2)
    if not docs:
        return "I could not find any relevant information to answer your question."

    context = "\n\n".join(
        f"[Source: {d['source']}]\n{d['text']}" for d in docs
    )

    prompt = f"""
Human: You are a concise and direct assistant.

Context:
{context}

Question: {query}

Assistant:
""".strip()

    return llm_answer(prompt)
