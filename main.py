import os
import json
import fitz
import re
import datetime
from dotenv import load_dotenv
from typing import List
# --- CHANGE START ---
# Import Form and Depends for the new upload endpoint
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Depends
# --- CHANGE END ---
from pydantic import BaseModel
from pinecone import Pinecone, ServerlessSpec
from pymongo import MongoClient
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.docstore.document import Document
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timezone


#load env variables
load_dotenv()

#initialzie the server
app = FastAPI()

# Add health check endpoint for Render
@app.get("/")
def health_check():
    return {"status": "ok"}

#cors 
origins = [
    "http://127.0.0.1:5500", 
    "http://localhost:5500",
    "http://localhost:3000",  # for local React
    "https://thevedasinstitute-frontend.vercel.app",  # your deployed React
    "https://yourcustomdomain.com"  # if you add custom domain
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
#mongo setup 
try:
 MONGO_URI = os.getenv("MONGO_URI")
 client = MongoClient(MONGO_URI)
 db = client["Vedas-RAG"]
 chats_collection = db["chats"] 
except Exception as e:
 print(f"error in Mongo {e}")
 client = None 

#PINECONE SETUP
try:
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    INDEX_NAME = "vedas-rag"

    # Create index if not exists
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=3072,  
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    print(" Pinecone initialized")
except Exception as e:
    print(f"Pinecone init error: {e}")
    pc = None
#initialize llms
try:
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("OPENAI_API_VERSION"),
    )
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        openai_api_version=os.getenv("EMBEDDING_MODEL_VERSION"),
    )
    print("Successfully initialized Azure OpenAI clients.")
except Exception as e:
    print(f" Error initializing Azure OpenAI clients: {e}")
    llm = None
    embeddings = None

class ChatRequest(BaseModel):
    session_id: str
    query: str

def extract_text_from_pdf(file: UploadFile) -> List[Document]:
    try:
        doc = fitz.open(stream=file.file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_text(text)
        return [Document(page_content=chunk) for chunk in chunks]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF extraction failed: {e}")

def save_chat(session_id, user_msg, assistant_msg):
    try:
        if chats_collection is not None:
            now = datetime.now(timezone.utc)
            chats_collection.insert_many([
                {"session_id": session_id, "role": "user", "content": user_msg, "timestamp":now
},
                {"session_id": session_id, "role": "assistant", "content": assistant_msg, "timestamp": now}
            ])
    except Exception as e:
        print(f"Failed to save chat: {e}")

def fetch_history(session_id, limit=20):
    try:
        if chats_collection is not None:
            history = list(chats_collection.find({"session_id": session_id}).sort("timestamp", -1).limit(limit))
            history.reverse()
            return history
    except Exception as e:
        print(f" Failed to fetch history: {e}")
    return []

# --- CHANGE START: Modified upload_pdf endpoint ---
# It now accepts a 'session_id' from a form field along with the file.
@app.post("/upload_pdf/")
async def upload_pdf(session_id: str = Form(...), file: UploadFile = File(...)):
    if not all([pc, embeddings]):
        raise HTTPException(status_code=500, detail="Pinecone or Embeddings not initialized")
    try:
        docs = extract_text_from_pdf(file)
        # We use the provided session_id as the namespace to isolate the data.
        vectorstore = PineconeVectorStore.from_documents(
            docs, 
            embeddings, 
            index_name=INDEX_NAME, 
            namespace=session_id
        )
        return {"message": f"PDF processed for session {session_id}."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF upload failed: {e}")
# --- CHANGE END ---
    
@app.post("/chat/")
async def chat(req: ChatRequest):
    if not all([pc, embeddings, llm]):
        raise HTTPException(status_code=500, detail="Services not initialized")
    try:
        # Short-term memory (no changes needed here)
        history = fetch_history(req.session_id)
        history_text = "\n".join([f"{h['role']}: {h['content']}" for h in history])

        # Long-term memory from Pinecone
        try:
            # --- CHANGE START: Modified Pinecone retrieval ---
            # We tell Pinecone to ONLY search within the namespace of the current session.
            vectorstore = PineconeVectorStore.from_existing_index(
                INDEX_NAME, 
                embeddings, 
                namespace=req.session_id
            )
            # --- CHANGE END ---
            docs = vectorstore.similarity_search(req.query, k=5)
            context_text = "\n".join([d.page_content for d in docs])
        except Exception as e:
            context_text = ""
            print(f" Pinecone retrieval failed: {e}")

        # Build final prompt 
        prompt = f"""
You are a helpful assistant named Ved.Your goal is to answer the user's question accurately and concisely based on the following information:

Conversation history:
{history_text}

Relevant context from documents:
{context_text}

User question: {req.query}

⚡ Instructions:
- Answer concisely and clearly.
- Prioritize the relevant context from the documents.
- If the context does not have the answer, politely say you do not know.
- Do NOT make up information or hallucinate.
- Respond in plain text, no markdown or code blocks.

Answer:
"""
        try:
            response = llm.predict(prompt)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"LLM response failed: {e}")

        # Save conversation (no changes needed here)
        save_chat(req.session_id, req.query, response)

        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {e}")