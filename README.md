# Meta Memory Chatbot

## Project Description

This is an end-to-end memory-based conversational chatbot built using:

- FastAPI (Backend)
- Simple HTML frontend
- Meta LLM (LLaMA compatible)
- Session-based memory

The chatbot retains conversation history and sends the full context to the LLM on each request.

---

## Architecture

Frontend (HTML + JS)
        ↓
FastAPI Backend
        ↓
Session Memory Store (In-memory)
        ↓
Meta LLM (LLaMA / API / Local model)

---

## Setup Instructions

### 1. Backend

cd backend
pip install -r requirements.txt
uvicorn main:app --reload

Server runs at:
http://localhost:8000

### 2. Frontend

Open:
frontend/index.html

---

## How Memory Works

Each user session has:

[
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "..."}
]

Session ID is generated automatically and reused.

---

## How to Connect Meta LLM

Replace get_llm_response() function.

Example using HuggingFace:

from transformers import AutoTokenizer, AutoModelForCausalLM

Load LLaMA model
Generate response from prompt
Return output text

OR

Call hosted API (Together.ai / Replicate / HuggingFace endpoint).

---

## Future Improvements

- Redis-based memory storage
- Token-based trimming instead of message count
- Authentication system
- Streaming responses
- WebSocket support
- Docker-compose setup
- Persistent DB storage
- Vector memory (RAG)
- Multi-user scaling
- Rate limiting
