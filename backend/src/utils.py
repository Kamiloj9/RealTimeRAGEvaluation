import os
import sys
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

def load_config():
    load_dotenv()
    if not os.getenv('OPENAI_API_KEY'):
        raise ValueError("OPENAI_API_KEY in not present in the enviroment")
    
    return os.getenv('OPENAI_API_KEY')