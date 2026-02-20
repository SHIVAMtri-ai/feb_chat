from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import uuid
import os
from dotenv import load_dotenv

from fastapi.responses import HTMLResponse
from pathlib import Path
# ✅ Load env variables
load_dotenv()
HUGGING_FACE_API_KEY = os.getenv("HUGGING_FACE_API")

# -----------------------------
# LangChain HuggingFace Setup
# -----------------------------
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Hosted Meta LLaMA
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-1B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=HUGGING_FACE_API_KEY
)

model = ChatHuggingFace(llm=llm)

# -----------------------------
# FastAPI Setup
# -----------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Session Memory
# -----------------------------
sessions: Dict[str, List[Dict]] = {}
MAX_HISTORY_LENGTH = 20

# -----------------------------
# Request Schema
# -----------------------------
class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None

# -----------------------------
# Trim history
# -----------------------------
def trim_history(history: List[Dict]):
    if len(history) > MAX_HISTORY_LENGTH:
        return history[-MAX_HISTORY_LENGTH:]
    return history

# -----------------------------
# LLM Response using LangChain HuggingFace
# -----------------------------
def get_llm_response(messages: List[Dict]) -> str:
    """
    Convert conversation history to LangChain message format
    and get LLM response
    """
    chat_hist = []

    # Add system message only once
    if not messages or messages[0].get("role") != "system":
        chat_hist.append(SystemMessage(content="You are a helpful AI assistant."))

    for msg in messages:
        if msg["role"] == "user":
            chat_hist.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            chat_hist.append(AIMessage(content=msg["content"]))

    # Call LLaMA
    result = model.invoke(chat_hist)
    return result.content

# -----------------------------
# Chat Endpoint
# -----------------------------
@app.post("/chat")
async def chat(request: ChatRequest):
    session_id = request.session_id or str(uuid.uuid4())

    if session_id not in sessions:
        sessions[session_id] = []

    history = sessions[session_id]

    # Append user message
    history.append({"role": "user", "content": request.message})
    history = trim_history(history)

    # Get LLM response
    response_text = get_llm_response(history)

    # Append assistant response
    history.append({"role": "assistant", "content": response_text})
    sessions[session_id] = history

    return {
        "response": response_text,
        "history": history,
        "session_id": session_id
    }

# -----------------------------
# Reset Endpoint
# -----------------------------
@app.post("/reset")
async def reset(request: ChatRequest):
    if request.session_id and request.session_id in sessions:
        sessions[request.session_id] = []

    return {"message": "Conversation reset successfully."}


@app.get("/", response_class=HTMLResponse)
async def get_frontend():
    html_path = Path(__file__).parent.parent / "frontend" / "index.html"
    return html_path.read_text()
