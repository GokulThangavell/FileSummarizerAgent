from fastapi import FastAPI, UploadFile
from services.ollama_service import upload_doc, summarize, ask

app = FastAPI()

app.post("/upload-file")
def upload_file(file:UploadFile):
    return upload_doc(file)

app.get("/summarize")
def summarize(file_name:str):
    return summarize(file_name=file_name)

app.get("/ask")
def ask(file_name:str, question:str):
    return ask(file_name=file_name, question=question)