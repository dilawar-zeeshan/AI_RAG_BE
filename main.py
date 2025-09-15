from fastapi import FastAPI, UploadFile, File
import psycopg2
import google.generativeai as genai
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from pypdf import PdfReader
from fastapi.middleware.cors import CORSMiddleware
from langchain.text_splitter import RecursiveCharacterTextSplitter  # ✅ LangChain

# Load env variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Database connection
conn = psycopg2.connect(
    dbname="vector_db",
    user="postgres",
    password="11011101",   # change if needed
    host="localhost",
    port="5432"
)
cur = conn.cursor()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict later in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_embedding(text: str):
    """Generate embedding using Gemini"""
    model = "models/embedding-001"
    result = genai.embed_content(model=model, content=text)
    return result["embedding"]

class DocumentRequest(BaseModel):
    content: str

@app.post("/add_document/")
def add_document(request: DocumentRequest):
    embedding = get_embedding(request.content)
    embedding_str = "[" + ",".join([str(x) for x in embedding]) + "]"

    cur.execute(
        "INSERT INTO documents (content, embedding) VALUES (%s, %s)",
        (request.content, embedding_str)
    )
    conn.commit()
    return {"message": "Document added"}

@app.post("/add_document_file/")
async def add_document_file(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        return {"error": "Only PDF supported for now"}

    reader = PdfReader(file.file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    # ✅ Use LangChain RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,   # characters per chunk
        chunk_overlap=200, # overlap between chunks
        separators=["\n\n", "\n", ".", " ", ""]  # tries larger splits first
    )
    chunks = splitter.split_text(text)

    for chunk in chunks:
        embedding = get_embedding(chunk)
        embedding_str = "[" + ",".join([str(x) for x in embedding]) + "]"
        cur.execute(
            "INSERT INTO documents (content, embedding) VALUES (%s, %s)",
            (chunk, embedding_str)
        )
    conn.commit()

    return {"message": f"PDF processed into {len(chunks)} chunks and stored"}

@app.get("/search/")
def search_documents(query: str, top_k: int = 3):
    query_embedding = get_embedding(query)
    query_embedding_str = "[" + ",".join([str(x) for x in query_embedding]) + "]"

    cur.execute(
        """
        SELECT content
        FROM documents
        ORDER BY embedding <-> %s
        LIMIT %s;
        """,
        (query_embedding_str, top_k)
    )
    results = cur.fetchall()
    return {"results": [r[0] for r in results]}

class ChatRequest(BaseModel):
    query: str
    top_k: int = 3

@app.post("/chat/")
def chat(request: ChatRequest):
    query_embedding = get_embedding(request.query)
    query_embedding_str = "[" + ",".join([str(x) for x in query_embedding]) + "]"

    cur.execute(
        """
        SELECT content
        FROM documents
        ORDER BY embedding <-> %s
        LIMIT %s;
        """,
        (query_embedding_str, request.top_k)
    )
    results = cur.fetchall()
    docs = [r[0] for r in results]

    context = "\n".join(docs) if docs else "No relevant documents found."

    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(
        f"Answer the question using the context.\n\nContext:\n{context}\n\nQuestion: {request.query}"
    )

    return {
        "answer": response.text,
        "context_used": docs
    }
