from langchain_ollama import ChatOllama,OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.retrievers import r
import os
from dotenv import load_dotenv

load_dotenv()

#Query
query = "who was the captain during world cup 2003?"

# Load Model and Embeddings
llm = ChatOllama(model = "qwen3:8b")
embedding = OllamaEmbeddings(model='nomic-embed-text')

#Load documents
document = PyPDFLoader(file_path='./data/star_players_indian_cricket_simple.pdf')
pages = document.load()

#Split Documents
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
docs = splitter.split_documents(pages)


# Define your directory path string
dir_path = "./vector_db"

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
    vector_db = FAISS.from_documents(docs,embedding)
    vector_db.save_local('./vector_db')
    print("FAISS index saved successfully.")

ret = vector_db.as_retriever(search_kwags={'k':2})
ret_chunk = ret.invoke(query)

# print(ret_chunk[0])

prompt = f"""Answer the below user query from the information available in the context only. 
If the context does not contain sufficient information, say I do not know the answer.
Query:\n {query}
\n Context:\n {ret_chunk}"""

response = llm.invoke(prompt)
print(response.content)

# print(docs[0])
# splitter.

# 

# response = llm.invoke("Who are you?")
# print(response.content)
