from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
from flask import Flask, render_template, request

load_dotenv()

# Configuration
FAISS_PATH = "faiss"
chat_history = []
PROJECT_NAME = "DocChat"
PROJECT_VERSION = "1.0.0"

# Initialize models
print(f"\n{'='*60}")
print(f"🚀 {PROJECT_NAME} — RAG Chat Application v{PROJECT_VERSION}")
print(f"{'='*60}")
print("\n📦 Initializing models...")
print("   • Loading HuggingFace embeddings (first run may take a moment)...")

try:
    llm = Ollama(model='llama3.2')
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},  # Changed to 'cpu' for better compatibility
        encode_kwargs={'normalize_embeddings': True}
    )
    print("   ✓ Models loaded successfully!")
except Exception as e:
    print(f"   ✗ Error loading models: {e}")
    print("   Make sure Ollama is running: ollama serve")
    raise

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True


def get_document_loader():
    """Load PDF documents from static directory"""
    loader = DirectoryLoader('static', glob="**/*.pdf", show_progress=True, loader_cls=PyPDFLoader)
    docs = loader.load()
    print(f"   📄 Loaded {len(docs)} PDF documents")
    return docs


def get_text_chunks(documents):
    """Split documents into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    print(f"   ✂️  Split into {len(chunks)} chunks")
    return chunks


def get_embeddings():
    """Create or load FAISS vector store"""
    path = os.path.join(os.getcwd(), FAISS_PATH)
    
    if os.path.exists(path) and os.listdir(path):
        print(f"\n🔍 Loading existing index from '{path}'...")
        try:
            db = FAISS.load_local(
                path, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            print("   ✓ Index loaded successfully!")
            return db
        except Exception as e:
            print(f"   ✗ Error loading index: {e}")
            print("   Creating new index instead...\n")
    
    print(f"\n🏗️  Creating new index...")
    documents = get_document_loader()
    
    if not documents:
        raise ValueError("❌ No PDF documents found in 'static/' directory!")
    
    chunks = get_text_chunks(documents)
    print(f"   🧠 Creating embeddings for {len(chunks)} chunks...")
    print("      (This may take a few minutes on first run)\n")
    
    db = FAISS.from_documents(chunks, embeddings)
    
    print(f"   💾 Saving index to '{path}'...")
    db.save_local(path)
    print("   ✓ Index created and saved!")
    
    return db


def get_retriever():
    """Get retriever from vector store"""
    db = get_embeddings()
    retriever = db.as_retriever(search_kwargs={"k": 3})
    return retriever


def get_chain():
    """Create RetrievalQA chain with custom prompt"""
    retriever = get_retriever()
    
    # Custom prompt template
    template = """You are a helpful assistant answering questions based on provided documents.

Use the following pieces of context to answer the question at the end. 
If you don't know the answer from the context, just say that you don't know. Don't try to make up an answer.
Keep your answer concise, clear, and directly relevant to the question asked.

Context:
{context}

Question: {question}

Answer:"""
    
    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )
    
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    return chain


@app.route('/')
def index():
    """Home page"""
    return render_template('home.html')


@app.route('/chat', methods=['GET', 'POST'])
def chat():
    """Chat interface"""
    if request.method == 'GET':
        return render_template('chat.html', chat_history=chat_history)
    
    question = request.form.get('question', '').strip()
    
    if not question:
        return render_template('chat.html', chat_history=chat_history, error="Please enter a question")
    
    print(f"\n{'─'*60}")
    print(f"Q: {question}")
    print(f"{'─'*60}")
    
    try:
        # Get the chain
        chain = get_chain()
        
        # Invoke the chain
        print("⏳ Generating response...")
        llm_response = chain.invoke({"query": question})
        
        result = llm_response.get('result', 'No answer generated').strip()
        
        # Extract source information
        sources = []
        source_docs = llm_response.get('source_documents', [])
        
        if source_docs:
            for idx, source in enumerate(source_docs):
                try:
                    source_path = source.metadata.get('source', 'Unknown')
                    # Remove 'static/' prefix if present
                    if source_path.startswith('static'):
                        source_path = source_path[7:] if len(source_path) > 7 and source_path[6] == '/' else source_path[6:]
                    
                    page_num = source.metadata.get('page', '?')
                    sources.append(f"{source_path} (p. {page_num})")
                except Exception as e:
                    print(f"⚠️  Error processing source {idx}: {e}")
        
        # Add to chat history
        chat_history.append(f"Q: {question}")
        if sources:
            sources_text = " • ".join(sources)
            chat_history.append(f"A: {result}\n\n📚 Sources: {sources_text}")
        else:
            chat_history.append(f"A: {result}")
        
        print(f"✓ Answer: {result[:80]}..." if len(result) > 80 else f"✓ Answer: {result}")
        if sources:
            print(f"📚 Sources ({len(sources)}): {', '.join(sources)}\n")
        
        return render_template('chat.html', chat_history=chat_history)
        
    except Exception as e:
        error_msg = f"{str(e)}"
        print(f"❌ Error: {error_msg}\n")
        import traceback
        traceback.print_exc()
        
        return render_template('chat.html', chat_history=chat_history, error=error_msg)


@app.route('/clear', methods=['POST'])
def clear_history():
    """Clear chat history"""
    global chat_history
    chat_history = []
    print("\n🗑️  Chat history cleared\n")
    return render_template('chat.html', chat_history=chat_history)


if __name__ == "__main__":
    print("\n" + "="*60)
    print(f"✨ {PROJECT_NAME} is starting...")
    print("="*60)
    print("\n📋 Requirements:")
    print("  • Ollama running: ollama serve")
    print("  • Ollama model: ollama pull llama3.2")
    print("  • PDF files in: ./static/ directory")
    print("\n🌐 Starting Flask server on http://localhost:5000")
    print("\n" + "="*60 + "\n")
    
    app.run(debug=False, port=5000, host='127.0.0.1')