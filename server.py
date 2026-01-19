import os
import uvicorn
from fastapi import FastAPI, HTTPException, Header
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Any, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import MessagesState  

# Import retrieval + ingestion
from WithSemantic import retrieve_documents, ingest_data
from dotenv import load_dotenv

load_dotenv()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Model
llm = ChatOpenAI(model="gpt-4o", temperature=0.2)

# Tools
tools = [retrieve_documents]
llm_with_tools = llm.bind_tools(tools)

# Nodes
def reasoner(state: MessagesState):
    ai = llm_with_tools.invoke(state["messages"])
    return {"messages": [ai]}  # will append (not overwrite) because of MessagesState


# Graph Construction
builder = StateGraph(MessagesState)
builder.add_node("reasoner", reasoner)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "reasoner")
builder.add_conditional_edges("reasoner", tools_condition)
builder.add_edge("tools", "reasoner")

# Memory
memory = InMemorySaver()
react_graph = builder.compile(checkpointer=memory)


app = FastAPI(title="Semantic RAG Chatbot")

@app.on_event("startup")
async def startup_ingest():
    ingest_data()

# CORS SETS ORIGIN TO ALLOW ALL 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request Models
class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default_thread"

class ChatResponse(BaseModel):
    response: str


@app.get("/status")
async def status():
    """
    Quick health/status check. Useful to verify paths + ingestion environment.
    """
    return {
        "ok": True,
        "cwd": os.getcwd(),
        "files_dir_abs": os.path.abspath("files"),
        "chunked_dir_abs": os.path.abspath(os.path.join("files", "chunked")),
    }


@app.post("/ingest")
async def ingest_endpoint(
    x_ingest_token: Optional[str] = Header(default=None, alias="X-Ingest-Token"),
):
    """
    Triggers ingestion from ./files into Chroma.
    Protect this endpoint with a simple token header:
      X-Ingest-Token: <your token>

    If you set env var INGEST_TOKEN, the header is required.
    If INGEST_TOKEN is not set, ingestion is allowed without a token.
    """
    expected = os.environ.get("INGEST_TOKEN")

    if expected and x_ingest_token != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        ingest_data()
        return {"ok": True, "message": "Ingestion triggered. Check server logs for progress."}
    except Exception as e:
        print(f"Ingest error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        config = {"configurable": {"thread_id": request.thread_id}}

        # User message
        inputs = {"messages": [HumanMessage(content=request.message)]}

        # Run graph
        result = react_graph.invoke(inputs, config=config)

        # Extract last message content
        last_message = result["messages"][-1]
        return ChatResponse(response=last_message.content)

    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Mount static files for frontend
STATIC_DIR = "static"
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


if __name__ == "__main__":
    print("Starting server on http://localhost:8001")
    uvicorn.run(app, host="0.0.0.0", port=8001)
