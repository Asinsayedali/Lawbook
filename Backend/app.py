from flask import Flask, request, jsonify
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
from flask_cors import CORS
import traceback
import re

app = Flask(__name__)
CORS(app)

# Configuration
DATA_FOLDER = 'data'
os.makedirs(DATA_FOLDER, exist_ok=True)

# Initialize global components
embedder = HuggingFaceEmbeddings()
llm = Ollama(model="deepseek-r1:1.5b")
retriever_instance = None

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

def load_law_documents():
    """Load and process all law-related PDFs from the data folder if not already loaded."""
    global retriever_instance

    if retriever_instance is not None:
        print("Documents already loaded. Skipping reload.")
        return  # Prevent reloading if already initialized

    print("Loading documents...")

    documents = []
    for filename in os.listdir(DATA_FOLDER):
        if filename.endswith('.pdf'):
            loader = PyPDFLoader(os.path.join(DATA_FOLDER, filename))
            docs = loader.load()
            documents.extend(docs)

    if not documents:
        print("No documents found in the data folder.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(documents)

    vector = FAISS.from_documents(split_docs, embedder)
    retriever_instance = vector.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    print("Document loading completed.")

def clean_llm_response(response):
    """Clean unwanted HTML-like tags and format the response."""
    cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
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

        # Retrieve relevant legal documents
        retrieved_docs = retriever_instance.get_relevant_documents(question)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # Format the question using the prompt template
        formatted_prompt = legal_prompt.format(context=context, question=question)

        # Pass the formatted prompt to the LLM
        response = llm(formatted_prompt)
        response = clean_llm_response(response)

        return jsonify({"answer": response})

    except Exception as e:
        print("Internal Server Error:", e)
        traceback.print_exc()
        return jsonify({"error": "Internal Server Error"}), 500

if __name__ == '__main__':
    load_law_documents()  # Load documents only once when starting
    app.run(debug=True)
