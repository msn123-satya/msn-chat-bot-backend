from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load API key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY is missing from .env file")

# Configure Google Gemini API
genai.configure(api_key=api_key)

# Create FastAPI app
app = FastAPI()

# Allow frontend requests (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (frontend)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Define request model
class Message(BaseModel):
    text: str

# 🧠 Create an in-memory dictionary to store chat history
chat_memory = {
    "history": []  # Single session memory; can be expanded to use session/user_id
}

@app.post("/chat/")
async def chat(msg: Message):
    try:
        # Append new user message to history
        chat_memory["history"].append(f"User: {msg.text}")

        # Build prompt from full conversation history
        full_prompt = "\n".join(chat_memory["history"]) + "\nBot:"

        # Generate AI response
        model = genai.GenerativeModel("gemini-2.0-flash")

        response = model.generate_content(full_prompt)

        # Save bot response to history
        chat_memory["history"].append(f"Bot: {response.text}")
        return {"reply": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
