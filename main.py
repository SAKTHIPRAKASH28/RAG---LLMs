from fastapi import FastAPI, UploadFile, Form, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pdf_processor import read_pdf, preprocess_text, create_index, retrieve_context
from model_manager import generate_responses, close_clients
import uuid
import uvicorn
import os
app = FastAPI()

# Middleware to allow CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this based on your frontend domain
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dictionary to hold user sessions
user_sessions = {}

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Endpoint to upload the PDF and create an index for the user session.
    Returns a session ID to be used in subsequent requests.
    """
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="File must be a PDF")
    pdf_bytes = await file.read()

    try:
        # Process the binary data with the updated read_pdf function
        pdf_text = read_pdf(pdf_bytes)

        # Preprocess PDF text
        chunks = preprocess_text(pdf_text)
        index, chunks = create_index(chunks)

        # Generate unique session ID
        session_id = str(uuid.uuid4())

        # Store the processed data in user_sessions
        user_sessions[session_id] = {
            "index": index,
            "chunks": chunks
        }

        return {"message": "PDF processed and indexed.", "session_id": session_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {e}")

@app.post("/ask-question/")
async def ask_question(session_id: str = Form(...), question: str = Form(...)):
    """
    Endpoint to ask questions based on the uploaded PDF and session ID.
    """
    if session_id not in user_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session_data = user_sessions[session_id]
    index = session_data["index"]
    chunks = session_data["chunks"]

    # Retrieve context based on question
    context = retrieve_context(question, index, chunks)

    # Generate responses from the model
    responses = generate_responses(question, context)

    return {"responses": responses}


@app.post("/close-session/")
async def close_session(session_id: str = Form(...)):
    """
    Endpoint to close the session and cleanup resources.
    """
    if session_id not in user_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    # Close model clients if needed and delete session data
    close_clients()
    del user_sessions[session_id]

    return {"message": "Session closed and resources cleaned up."}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)