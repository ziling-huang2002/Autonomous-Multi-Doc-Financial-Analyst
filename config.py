import os
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from termcolor import colored
from langchain_groq import ChatGroq  # 確保導入這個

load_dotenv(override=True)

# ==============================================================================
# 1. Project Folders
# ==============================================================================
DATA_FOLDER = "data"
DB_FOLDER = "chroma_db"

# ==============================================================================
# 2. Dataset Configuration
# ==============================================================================
# Default file-to-key mapping. 
# Better yet, this could be moved to a config.json or scanned from folder.
FILES = {
    "apple": "FY24_Q4_Consolidated_Financial_Statements.pdf",
    "tesla": "tsla-20241231-gen.pdf"
}

# ==============================================================================
# 3. Embedding Model (Can Change)
# ==============================================================================
LOCAL_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" # BAAI/bge-small-en-v1.5

def get_embeddings():
    print(colored(f"🔄 Loading Local Embedding Model: {LOCAL_EMBEDDING_MODEL}...", "cyan"))
    return HuggingFaceEmbeddings(model_name=LOCAL_EMBEDDING_MODEL)

# ==============================================================================
# 4. LLM Model Factory (Supports Multiple Providers)
# ==============================================================================
def get_llm(temperature=0):
    """
    Returns a LangChain Chat Model based on environment variables.
    Supported Providers: google, openai, anthropic
    """
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    
    if provider == "groq":
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                print(colored("⚠️ Warning: GROQ_API_KEY not found in .env!", "red"))
            
            return ChatGroq(
                model="llama-3.3-70b-versatile", # Groq 目前最強的型號
                temperature=temperature,
                groq_api_key=api_key
            )

    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print(colored("⚠️ Warning: GOOGLE_API_KEY not found!", "red"))
        return ChatGoogleGenerativeAI(
            model=os.getenv("GOOGLE_MODEL", "gemini-2.5-flash"),
            temperature=temperature,
            google_api_key=api_key,
            convert_system_message_to_human=False,
            max_output_tokens=2048
        )
    
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print(colored("⚠️ Warning: OPENAI_API_KEY not found!", "red"))
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=temperature,
            api_key=api_key
        )
    
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print(colored("⚠️ Warning: ANTHROPIC_API_KEY not found!", "red"))
        return ChatAnthropic(
            model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest"),
            temperature=temperature,
            api_key=api_key
        )
    

    
    else:
        raise ValueError(f"❌ Unsupported LLM_PROVIDER: {provider}")