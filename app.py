from google.colab import drive
import requests
from PyPDF2 import PdfReader
import os
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain import HuggingFaceHub
from langchain.chains import ConversationChain

drive.mount('/content/drive')
path = '/content/drive/MyDrive/Colab Notebooks/bank_debit_rules.pdf'

try:
    loader = PyPDFLoader(path)
    pages = loader.load()
    print(f"Successfully loaded {len(pages)} pages.")
except FileNotFoundError:
    print("File not found. Please check the file path.")
except Exception as e:
    print(f"An error occurred: {e}")

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
docs = splitter.split_documents(pages)
embeddings = HuggingFaceEmbeddings()
doc_search = Chroma.from_documents(docs, embeddings)

template = """
Use the following context to answer the question:
------
Context:
{context}
------
Question:
{question}
Answer:
"""
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=template,
)

HUGGING_FACE_HUB_API_KEY = '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'  
os.environ['HUGGING_FACE_HUB_API_KEY'] = HUGGING_FACE_HUB_API_KEY
repo_id = 'google/t5-v1_1-large'  
try:
    llm = HuggingFaceHub(
        huggingfacehub_api_token=os.environ['HUGGING_FACE_HUB_API_KEY'],
        repo_id=repo_id,
        model_kwargs={'temperature': 1e-10, 'max_length': 128}  
    )
except requests.exceptions.RequestException as e:
    print(f"An error occurred while accessing the model: {e}")
    
try:
    retrieval_chain = RetrievalQA.from_chain_type(
        llm,
        chain_type='stuff',
        retriever=doc_search.as_retriever(),
        chain_type_kwargs={
            "prompt": prompt,
        }
    )
except Exception as e:
    print(f"An error occurred while creating the RetrievalQA chain: {e}")

query = input("Enter your query about the document: ")
try:
    answer = retrieval_chain.run(query)
    print("Answer:", answer.strip())
except Exception as e:
    print(f"An error occurred while running the query: {e}")
