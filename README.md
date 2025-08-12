Make few changes as follows:
Make a folder and place all the files in the folder. 
Make another folder as 'static' in the main folder and put the 'index.html' in it.
In .env copy/ paste your OPENAI_API_KEY.

PDF Chatbot with FastAPI & OpenAI
An interactive web application that lets you chat with your PDFs.
Upload any PDF document, ask questions in natural language, and get accurate, context-based answers powered by OpenAI embeddings and FastAPI.

 Features:
 Upload PDFs via the browser

Ask questions & get PDF-based answers

 Semantic search with vector embeddings

 Fast & interactive chat UI

 Built with FastAPI backend + HTML/JS frontend

 Tech Stack
Frontend: HTML, CSS, JavaScript

Backend: FastAPI (Python)

PDF Extraction: pdfplumber

AI: OpenAI API (embeddings + chat models)

Env Management: .env for API keys

Running the Server:
use below command in terminal:
uvicorn main:app --reload

Then open http://127.0.0.1:8000 in your browser.
