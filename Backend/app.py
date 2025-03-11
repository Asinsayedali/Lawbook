from flask import Flask, request, jsonify
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
from flask_cors import CORS
import traceback
import re
import google.generativeai as genai
load_dotenv()
app = Flask(__name__)
CORS(app)

# Configuration
DATA_FOLDER = 'data'
VECTOR_STORE_PATH = 'vector_store'  # Path to save the vector store
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)

# Initialize global components
embedder = HuggingFaceEmbeddings()
retriever_instance = None
llm = None  # Global LLM instance

# Custom prompt template for legal questions
LEGAL_PROMPT_TEMPLATE = """
You are a legal assistant. Use the following pieces of legal context to answer the question at the end.
If you don't know the answer or cannot find relevant information in the context, say so clearly
rather than making up information. Your answers should be accurate, informative, and based only
on the provided context.

Context:
{context}

Question: {question}

Answer:
"""

# Create the prompt object
legal_prompt = PromptTemplate(
    template=LEGAL_PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)

def initialize_gemini():
    """Initialize Google Gemini model with API key."""
    global llm
    
    # Only initialize if not already done
    if llm is not None:
        return llm
    
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set")
    
    # Configure the Gemini API
    genai.configure(api_key=api_key)
    
    # Initialize the Gemini model through LangChain
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0.2,
        google_api_key=api_key,
    )
    
    print("Gemini model initialized successfully")
    return llm

def check_vector_store_exists():
    """Check if the vector store files exist."""
    index_file = os.path.join(VECTOR_STORE_PATH, 'index.faiss')
    docstore_file = os.path.join(VECTOR_STORE_PATH, 'index.pkl')
    return os.path.exists(index_file) and os.path.exists(docstore_file)

def check_if_docs_updated():
    """Check if PDF documents have been added or modified since last index."""
    # Path to store the last modification timestamp
    timestamp_file = os.path.join(VECTOR_STORE_PATH, 'last_modified.txt')
    
    # Get the most recent modification time of any PDF in the data folder
    latest_mod_time = 0
    for filename in os.listdir(DATA_FOLDER):
        if filename.endswith('.pdf'):
            file_path = os.path.join(DATA_FOLDER, filename)
            mod_time = os.path.getmtime(file_path)
            latest_mod_time = max(latest_mod_time, mod_time)
    
    # If no PDFs found, return True to force reindexing
    if latest_mod_time == 0:
        return True
        
    # Check if timestamp file exists
    if not os.path.exists(timestamp_file):
        # Create the file with current timestamp
        with open(timestamp_file, 'w') as f:
            f.write(str(latest_mod_time))
        return True
    
    # Read the last modification time
    with open(timestamp_file, 'r') as f:
        stored_time = float(f.read().strip())
    
    # Update the timestamp file with new time if changed
    if latest_mod_time > stored_time:
        with open(timestamp_file, 'w') as f:
            f.write(str(latest_mod_time))
        return True
        
    return False

def load_law_documents():
    """Load and process all law-related PDFs from the data folder if not already loaded."""
    global retriever_instance
    
    # If retriever is already initialized, return early
    if retriever_instance is not None:
        print("Retriever already initialized in memory. Skipping reload.")
        return
    
    # Check if we have a saved vector store and no documents have changed
    if check_vector_store_exists() and not check_if_docs_updated():
        print("Loading existing vector store from disk...")
        try:
            vector = FAISS.load_local(VECTOR_STORE_PATH, embedder, allow_dangerous_deserialization=True)
            retriever_instance = vector.as_retriever(search_type="similarity", search_kwargs={"k": 5})
            print("Vector store loaded successfully from disk.")
            return
        except Exception as e:
            print(f"Error loading vector store: {str(e)}")
            print("Will recreate the vector store...")
    
    print("Loading and processing documents...")
    
    documents = []
    for filename in os.listdir(DATA_FOLDER):
        if filename.endswith('.pdf'):
            file_path = os.path.join(DATA_FOLDER, filename)
            print(f"Loading {file_path}...")
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            documents.extend(docs)

    if not documents:
        print("No documents found in the data folder.")
        return

    print(f"Processing {len(documents)} document pages...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(documents)
    print(f"Created {len(split_docs)} text chunks for indexing.")

    print("Creating vector embeddings... (this may take a while)")
    vector = FAISS.from_documents(split_docs, embedder)
    
    # Save the vector store to disk
    print("Saving vector store to disk...")
    vector.save_local(VECTOR_STORE_PATH)
    
    retriever_instance = vector.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    print("Document processing and indexing completed.")

def clean_llm_response(response):
    """Clean unwanted HTML-like tags and format the response."""
    cleaned = response
    cleaned = re.sub(r'<think>.*?</think>', '', cleaned, flags=re.DOTALL)
    cleaned = re.sub(r'<.*?>', '', cleaned)
    cleaned = re.sub(r'\*\*(.*?)\*\*', r'\1', cleaned)
    cleaned = re.sub(r'-\s+', '- ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    cleaned = re.sub(r'(\d+\.\s\S+?:)', r'\n\1', cleaned)
    return cleaned

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({"error": "Invalid request"}), 400

        question = data['question']

        if retriever_instance is None:
            return jsonify({"error": "Legal documents not loaded yet."}), 500

        # Initialize Gemini model if not already done
        try:
            if not llm:
                initialize_gemini()
        except ValueError as e:
            print(f"API Key Error: {str(e)}")
            return jsonify({"error": str(e)}), 500
        except Exception as e:
            print(f"Gemini Initialization Error: {str(e)}")
            traceback.print_exc()
            return jsonify({"error": f"Failed to initialize Gemini model: {str(e)}"}), 500

        # Retrieve relevant legal documents
        retrieved_docs = retriever_instance.get_relevant_documents(question)
        if not retrieved_docs:
            return jsonify({"answer": "I couldn't find any relevant information in the legal documents to answer your question."})
            
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # Format the question using the prompt template
        formatted_prompt = legal_prompt.format(context=context, question=question)

        try:
            # Pass the formatted prompt to the Gemini model
            response = llm.invoke(formatted_prompt)
            if hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)
                
            cleaned_response = clean_llm_response(response_text)
            return jsonify({"answer": cleaned_response})
        except Exception as e:
            print(f"Gemini API Call Error: {str(e)}")
            traceback.print_exc()
            return jsonify({"error": f"Error with Gemini API: {str(e)}"}), 500

    except Exception as e:
        print("Internal Server Error:", e)
        traceback.print_exc()
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Simple endpoint to verify the server is running."""
    return jsonify({"status": "ok", "documents_loaded": retriever_instance is not None})

@app.route('/refresh', methods=['POST'])
def refresh_index():
    """Force reindex of the documents."""
    global retriever_instance
    retriever_instance = None
    try:
        load_law_documents()
        return jsonify({"status": "success", "message": "Vector store refreshed successfully"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    # Load documents first
    load_law_documents()
    
    # Initialize Gemini at startup
    try:
        initialize_gemini()
        print("Server starting with Gemini initialized")
    except Exception as e:
        print(f"Warning: Could not initialize Gemini at startup: {str(e)}")
        print("Will attempt to initialize on first request")
    
    # Start the server
    app.run(debug=False)