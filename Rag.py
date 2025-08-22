from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
import os
import json
import json_repair
from langchain_community.embeddings import HuggingFaceEmbeddings

import google.generativeai as genai

from dotenv import load_dotenv
load_dotenv()
if not os.getenv("HF_TOKEN"):
    raise ValueError("HF_TOKEN environment variable not set.")

genai.configure(api_key="AIzaSyBcnMJ3DAm5KDwG3lbmETAB1z4GTcd0k2k")

generator_gemini = genai.GenerativeModel("models/gemini-2.5-flash-lite")
documents = []
doc_dir = "docs"
for filename in os.listdir(doc_dir):
    filepath = os.path.join(doc_dir, filename)
    try:
        if filename.endswith(".txt"):
            with open(filepath, 'r', encoding='utf-8') as f:
                documents.append(Document(page_content=f.read(), metadata={"source": filepath}))
    except Exception as e:
        print(f"Error loading file {filepath}: {e}")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
#chunked_documents = text_splitter.split_documents(documents)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#vectorstore = Chroma.from_documents(documents=chunked_documents, embedding=embeddings)
persist_directory = "content/vectordb"
#vectorstore = Chroma.from_documents(documents=chunked_documents, embedding=embeddings, persist_directory=persist_directory)
#vectorstore.persist()

vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

## WithoutRAG
print("Without RAG:--------------------------------------------------------")
response = generator_gemini.generate_content("").text
print(response)
