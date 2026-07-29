from langchain_ollama import ChatOllama,OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable
import os
from dotenv import load_dotenv

load_dotenv()

#Query
query = "who was the captain during world cup 2003?"

@traceable(name="Load model and embedding")
def load_model_and_embedding():
# Load Model and Embeddings
    llm = ChatOllama(model = "qwen3:8b")
    embedding = OllamaEmbeddings(model='nomic-embed-text')
    return llm,embedding

@traceable(name="Load Documents")
def load_documents():
#Load documents
    document = PyPDFLoader(file_path='./data/star_players_indian_cricket_simple.pdf')
    pages = document.load()
    return pages

@traceable(name='Chunking')
def chunk_docs(pages):
    #Split Documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    docs = splitter.split_documents(pages)
    return docs



# Define your directory path string
dir_path = "./vector_db"

@traceable(name='vector db')
def check_for_vector_db(embedding):
    # Check if the directory exists
    if os.path.isdir(dir_path):
        print("The directory exists!")
        vector_db = FAISS.load_local(
        dir_path, 
        embedding, 
        allow_dangerous_deserialization=True)

    else:
        print("The directory does not exist.")
        #Create vector db
        
        pages = load_documents()
        docs = chunk_docs(pages)
        vector_db = FAISS.from_documents(docs,embedding)
        vector_db.save_local('./vector_db')
        print("FAISS index saved successfully.")

    return vector_db

@traceable(name="retrieve chunk")
def retriever_chunk(vector_db,query):
    ret = vector_db.as_retriever(search_kwags={'k':2})
    ret_chunk = ret.invoke(query)
    return ret_chunk

@traceable(name="Run full pipeline")
def run_full_pipeline(query):
    llm,embedding = load_model_and_embedding()
    vector_db = check_for_vector_db(embedding)
    ret_chunk = retriever_chunk(vector_db,query)

    prompt = f"""Answer the below user query from the information available in the context only. 
    If the context does not contain sufficient information, say I do not know the answer.
    Query:\n {query}
    \n Context:\n {ret_chunk}"""

    response = llm.invoke(prompt)
    return response.content
    

print(run_full_pipeline(query))






# print(docs[0])
# splitter.

# 

# response = llm.invoke("Who are you?")
# print(response.content)
