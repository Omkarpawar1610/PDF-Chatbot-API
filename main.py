from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static folder (HTML frontend)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Globals
vectorstore = None
chat_history = []


class QueryRequest(BaseModel):
    question: str


@app.get("/")
async def serve_homepage():
    """Serve index.html when visiting root URL."""
    return FileResponse("static/index.html")


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload and process a PDF file, chunk it, and store embeddings in FAISS.
    """
    global vectorstore, chat_history
    chat_history = []

    # Save file temporarily
    os.makedirs("temp", exist_ok=True)
    file_path = f"temp/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Extract text from PDF
    pdf_reader = PdfReader(file_path)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    if not text.strip():
        return {"error": "No text found in the PDF"}

    # Chunk text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(text)

    # Create embeddings & store in FAISS
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_texts(chunks, embeddings)

    return {
        "message": "PDF processed successfully",
        "total_chunks": len(chunks),
        "file_name": file.filename
    }


@app.post("/query")
async def query_pdf(request: QueryRequest):
    """
    Ask a question about the uploaded PDF.
    """
    global vectorstore, chat_history
    if not vectorstore:
        return {"error": "No document uploaded"}

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(
        temperature=0,
        model_name="gpt-4o-mini",
        openai_api_key=OPENAI_API_KEY
    )
    qa = ConversationalRetrievalChain.from_llm(llm, retriever)

    result = qa({"question": request.question, "chat_history": chat_history})
    chat_history.append((request.question, result["answer"]))

    return {"answer": result["answer"], "chat_history": chat_history}
