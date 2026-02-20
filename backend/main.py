from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict
import uuid
import os
from dotenv import load_dotenv
from pathlib import Path

# Load env
load_dotenv()
HUGGING_FACE_API_KEY = os.getenv("HUGGING_FACE_API")

# -----------------------------
# LangChain HuggingFace Setup
# -----------------------------
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

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
# Serve Frontend Static Files
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
frontend_path = BASE_DIR / "frontend"

app.mount("/static", StaticFiles(directory=frontend_path), name="static")

@app.get("/")
async def serve_index():
    return FileResponse(frontend_path / "index.html")

# -----------------------------
# Session Memory
# -----------------------------
sessions: Dict[str, List[Dict]] = {}
MAX_HISTORY_LENGTH = 20

class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None

def trim_history(history: List[Dict]):
    return history[-MAX_HISTORY_LENGTH:]

def get_llm_response(messages: List[Dict]) -> str:
    chat_hist = []

    chat_hist.append(SystemMessage(content="""
You are a professional AI assistant.

Follow these rules strictly:
1. Give concise and clear answers.
2. Structure responses using bullet points or short paragraphs.
3. Provide logical reasoning when needed.
4. Avoid unnecessary explanations.
5. If the question is technical, explain step-by-step briefly.
6. If unsure, say you are not certain instead of guessing.
7. Keep tone professional and confident.

Always prioritize clarity, reasoning, and structured output.
"""))

    for msg in messages:
        if msg["role"] == "user":
            chat_hist.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            chat_hist.append(AIMessage(content=msg["content"]))

    result = model.invoke(chat_hist)
    return result.content

@app.post("/chat")
async def chat(request: ChatRequest):
    session_id = request.session_id or str(uuid.uuid4())

    if session_id not in sessions:
        sessions[session_id] = []

    history = sessions[session_id]

    history.append({"role": "user", "content": request.message})
    history = trim_history(history)

    response_text = get_llm_response(history)

    history.append({"role": "assistant", "content": response_text})
    sessions[session_id] = history

    return {
        "response": response_text,
        "history": history,
        "session_id": session_id
    }

@app.post("/reset")
async def reset(request: ChatRequest):
    if request.session_id and request.session_id in sessions:
        sessions[request.session_id] = []

    return {"message": "Conversation reset successfully."}
