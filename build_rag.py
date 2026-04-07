import os
import re
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from termcolor import colored
from config import get_embeddings, DATA_FOLDER, DB_FOLDER, FILES

def clean_text(text: str) -> str:
    """
    More robust text cleaning: 
    1. Remove multiple newlines
    2. Remove extra spaces
    3. Normalize whitespace
    """
    text = text.replace("\n", " ")
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def build_vector_dbs():
    """
    ETL Pipeline: Load PDF -> Clean -> Split -> Embed -> Store in ChromaDB
    """
    embeddings = get_embeddings()
    
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)
        print(colored(f"⚠️ Warning: {DATA_FOLDER} directory was empty. Creating it...", "yellow"))

    # Dynamic file discovery: If a file is in DATA_FOLDER but not in FILES, add it.
    # This makes the project much more extensible!
    all_files = FILES.copy()
    for f in os.listdir(DATA_FOLDER):
        if f.endswith(".pdf"):
            key = f.split(".")[0].lower()
            if key not in all_files:
                all_files[key] = f
                print(colored(f"✨ Found new document: {f} (Setting key to '{key}')", "green"))

    for key, filename in all_files.items():
        persist_dir = os.path.join(DB_FOLDER, key)
        file_path = os.path.join(DATA_FOLDER, filename)

        if os.path.exists(persist_dir):
            print(colored(f"✅ DB for '{key}' already exists at {persist_dir}. skipping...", "yellow"))
            continue

        if not os.path.exists(file_path):
            print(colored(f"❌ Missing source file: {filename}", "red"))
            continue

        print(colored(f"🔨 Building Vector Index for {key}...", "cyan"))
        
        # 1. Load PDF
        loader = PyMuPDFLoader(file_path)
        docs = loader.load()
        print(f"   - Loaded {len(docs)} pages.")

        # 2. Clean & Split
        for doc in docs:
            doc.page_content = clean_text(doc.page_content)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,     # Reduced slightly for better retrieval precision
            chunk_overlap=200,   
            separators=["\n\n", "\n", " ", ""] 
        )

        splits = splitter.split_documents(docs)
        print(f"   - Split into {len(splits)} chunks.")
        
        # 3. Embed & Store
        print("   - Embedding and storing... (This may take a while)")
        Chroma.from_documents(splits, embeddings, persist_directory=persist_dir)
        print(colored(f"🎉 Successfully built DB for {key}!", "green"))

if __name__ == "__main__":
    build_vector_dbs()