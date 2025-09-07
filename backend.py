from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import uvicorn
import fitz  # PyMuPDF for PDF text extraction
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
import tempfile
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import OpenAI


app = FastAPI()

# Globals to store vector DB and QA chain
vectorstore = None
qa_chain = None

def extract_text_from_pdf(file_path: str) -> str:
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def create_vectorstore_and_qa_chain(doc_text: str):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500,
        chunk_overlap=100,
    )
    docs = splitter.create_documents([doc_text])
    embeddings = OpenAIEmbeddings()  # Requires OPENAI_API_KEY environment variable
    vector_db = FAISS.from_documents(docs, embeddings)
    retriever = vector_db.as_retriever(search_kwargs={"k": 4})
    llm = OpenAI(temperature=0)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return vector_db, qa

class QuestionRequest(BaseModel):
    question: str

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp.flush()
        text = extract_text_from_pdf(tmp.name)
    global vectorstore, qa_chain
    vectorstore, qa_chain = create_vectorstore_and_qa_chain(text)
    return {"message": f"PDF '{file.filename}' processed successfully.", "text_length": len(text)}

@app.post("/ask/")
async def ask_question(payload: QuestionRequest):
    global qa_chain
    if not qa_chain:
        return {"error": "No document uploaded. Please upload a PDF first."}
    answer = qa_chain.run(payload.question)
    return {"answer": answer}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
