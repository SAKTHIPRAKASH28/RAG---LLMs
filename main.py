from fastapi import FastAPI, UploadFile, Form, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import uuid
import uvicorn
import os
from openai import OpenAI, RateLimitError
from file_processor import *
from model_manager import *

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

user_sessions = {}

AVAILABLE_MODELS = ["phi-3-small", "mistral", "meta-llama-3", "gpt-4o", "gemini", "ai21-jamba-1.5-mini"]

@app.post("/upload-file/")
async def upload_file(file: UploadFile = File(...)):
    file_bytes = await file.read()
    content_type = file.content_type

    try:
        if content_type == "application/pdf":
            text = read_pdf(file_bytes)
        elif content_type == "application/vnd.openxmlformats-officedocument.presentationml.presentation":
            text = read_pptx(file_bytes)
        elif content_type == "text/plain":
            text = read_txt(file_bytes)
        elif content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            text = read_docx(file_bytes)
        elif content_type.startswith("image/"):
            text = read_image(file_bytes)
        elif content_type.startswith("audio/"):
            text = read_audio(file_bytes)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")

        chunks = preprocess_text(text)
        index, chunks = create_index(chunks)

        session_id = str(uuid.uuid4())
        user_sessions[session_id] = {
            "index": index,
            "chunks": chunks
        }

        return {"message": "File processed and indexed.", "session_id": session_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {e}")

@app.post("/ask-question/")
async def ask_question(session_id: str = Form(...), question: str = Form(...), models: List[str] = Form(...)):

    if session_id not in user_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session_data = user_sessions[session_id]
    index = session_data["index"]
    chunks = session_data["chunks"]

    context = retrieve_context(question, index, chunks)
    
    # Filter responses based on selected models
    responses = generate_responses(question, context,models)
    filtered_responses = {model: response for model, response in responses.items() if model in models}

    return {"responses": filtered_responses}


@app.post("/close-session/")
async def close_session(session_id: str = Form(...)):
    if session_id not in user_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    del user_sessions[session_id]

    return {"message": "Session closed and resources cleaned up."}

@app.get("/available-models/")
async def get_available_models():
    return {"models": AVAILABLE_MODELS}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)