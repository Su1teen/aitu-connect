import os
import glob
from dotenv import load_dotenv  # Only needed for local development
from langchain.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
# Load environment variables from a .env file (for local development)
load_dotenv()

# Retrieve the OpenAI API key from the environment variables
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("The OPENAI_API_KEY environment variable is not set!")

def load_documents(directory: str):
    """
    Load PDF, DOCX, and TXT documents from a specified directory.
    """
    file_paths = glob.glob(os.path.join(directory, "*"))
    documents = []
    for path in file_paths:
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext == ".pdf":
                loader = PyPDFLoader(path)
            elif ext == ".docx":
                loader = UnstructuredWordDocumentLoader(path)
            elif ext == ".txt":
                loader = TextLoader(path)
            else:
                continue

            docs = loader.load()
            documents.extend(docs)
            print(f"Loaded {len(docs)} documents from {path}")
        except Exception as e:
            print(f"Error loading {path}: {e}")
    return documents

def build_vector_index(documents):
    """
    Split documents into chunks and create a FAISS vector store.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    print(f"Total chunks created: {len(docs)}")

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

def main():
    docs_folder = "Documents"  
    print("Loading documents...")
    documents = load_documents(docs_folder)
    if not documents:
        print("No documents found. Please ensure the folder contains PDF, DOCX, or TXT files.")
        return

    print("Building vector index...")
    vectorstore = build_vector_index(documents)

    # Save the FAISS index locally to avoid rebuilding it each time
    index_path = "faiss_index"
    vectorstore.save_local(index_path)
    print(f"Vector index saved locally at '{index_path}'.")

if __name__ == '__main__':
    main()
