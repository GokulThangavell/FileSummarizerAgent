from fastapi import FastAPI, UploadFile
from services.ollama_service import upload_doc, summarize, ask

#import nltk

#It will download the Punkt resource the first time your app runs.
#On subsequent runs, NLTK checks if it’s already downloaded and will not download again.
#This ensures your app always has the resource available.

#nltk.download('punkt')  
#nltk.download('punkt_tab')  

#This keeps the resource inside your project folder.
#Useful for Docker containers or isolated environments.
#import nltk
#nltk.download('punkt', download_dir='./nltk_data')
#import nltk.data
#nltk.data.path.append('./nltk_data')

app = FastAPI()

@app.post("/upload-file")
async def upload_file_api(file:UploadFile):
    return await upload_doc(file)

@app.get("/summarize")
async def summarize_api(file_name:str):
    return await summarize(file_name=file_name)

@app.get("/ask")
async def ask_api(file_name:str, question:str):
    return await ask(file_name=file_name, question=question)