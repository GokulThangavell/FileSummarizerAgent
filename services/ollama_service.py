from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from qdrant_client.models import VectorParams
from fastapi import UploadFile
import PyPDF2
import nltk
from nltk.tokenize import sent_tokenize
import uuid
import ollama

qdrant_client = QdrantClient("localhost", "6333")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

collection_name = "summarize_collection"

if not qdrant_client.collection_exists(collection_name):
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance="Cosine")
    )

def extract_text_from_file(file):
    reader = PyPDF2.PdfReader(file)
    return " ".join(page.extract_text() for page in reader.pages)

def get_sentence_chunks(text):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if(len(current_chunk) + len(sentence) <= 1000):
            current_chunk +=" "+sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    if(current_chunk):
        chunks.append(current_chunk)
    
    return chunks


async def upload_doc(file:UploadFile):
    text = extract_text_from_file(file) if file.filename.endswith(".pdf") else (await file.read()).decode("utf-8")
    sentence_chunks = get_sentence_chunks(text)
    vectors = embedding_model.encode(sentence_chunks).tolist()
    points = [{
        "id":str(uuid.uuid4()),
        "vector":vec,
        "payload":{"file":file.filename, "chunk":chunk}
    }for vec, chunk in zip(vectors, sentence_chunks)]

    qdrant_client.upsert(
        collection_name=collection_name,
        points=points
    )
    return {"message":f"{file.filename} uploaded and indexed"}


async def summarize(file_name:str):
    results = qdrant_client.scroll(collection_name=collection_name
                                   , scroll_filter={"must":[{"key":"file", "match":{"value":file_name}}]})
    chunks = [p.payload["chunk"] for p in results[0]]
    combined_text = "\n".join(chunks)
    response = ollama.chat(model="mistral", message=[{"role":"user", "content":f"Summarize the following text:\n{combined_text}"}])
    return {"summary":response["message"]["content"]}

async def ask(file_name:str, question:str):
    question_vector = embedding_model.encode(question).tolist()
    results = qdrant_client.search(collection_name=collection_name
                                   , query_vector=question_vector
                                   , limit = 5
                                   , query_filter={"must":[{"key":"file", "match":{"value":file_name}}]})
    context ="\n".join([r.payload("chunk") for r in results])
    response = ollama.chat(model="mistral", message=[{"role":"user", "content":f"Answer the question based on this context:\n{context}\nQuestion:{question}"}])
    return{"answer":response["message"]["content"]}








