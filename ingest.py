import os
import glob
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# The requirements stated: No Langchain Agents. 
# We are only using langchain for the text splitter (chunking), not agents.
# We will use sentence_transformers manually to compute embeddings to be safe, 
# although using Langchain's HuggingFaceEmbeddings is usually fine.
import chromadb
from chromadb.utils import embedding_functions

def load_documents(directory="data"):
    print(f"Loading documents from {directory}...")
    documents = []
    
    # Handle PDFs
    pdf_files = glob.glob(os.path.join(directory, "*.pdf"))
    # Handle TXTs
    txt_files = glob.glob(os.path.join(directory, "*.txt"))
    
    file_paths = pdf_files + txt_files
    
    if not file_paths:
        print(f"WARNING: No Document files found in '{directory}'. Please place the 4 AI regulation documents there.")
        return []

    for file_path in file_paths:
        try:
            if file_path.endswith('.pdf'):
                reader = PdfReader(file_path)
                text = ""
                for page in reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    
            documents.append({"text": text, "source": os.path.basename(file_path)})
            print(f"Loaded: {os.path.basename(file_path)} ({len(text)} characters)")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            
    return documents

def chunk_documents(documents, chunk_size=1200, chunk_overlap=300):
    print("Chunking documents...")
    # We justify our chunking strategy: 1200 chars handles a few paragraphs ensuring context isn't lost.
    # 300 overlap ensures continuity across boundary points.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = []
    # For chromadb, we need ids, documents, and metadatas
    for doc in documents:
        splits = text_splitter.split_text(doc["text"])
        for i, split in enumerate(splits):
            chunks.append({
                "id": f"{doc['source']}_chunk_{i}",
                "text": split,
                "metadata": {"source": doc["source"], "chunk_id": i}
            })
            
    print(f"Generated {len(chunks)} chunks in total.")
    return chunks

def build_vector_store(chunks, db_path="chroma_db", collection_name="ai_regulation"):
    print(f"Building vector store at {db_path}...")
    # Initialize chroma client
    client = chromadb.PersistentClient(path=db_path)
    
    # We use all-MiniLM-L6-v2 as our embedding model. It is small, fast and locally run.
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    # Reset collection if exists to start fresh
    try:
        client.delete_collection(name=collection_name)
    except:
        pass
        
    collection = client.create_collection(
        name=collection_name, 
        embedding_function=sentence_transformer_ef
    )
    
    if chunks:
        # Prepare lists for chroma
        ids = [chunk["id"] for chunk in chunks]
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
        
        # Batch add to avoid potential limits if too many chunks
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            collection.add(
                ids=ids[i:i+batch_size],
                documents=texts[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size]
            )
        print("Successfully embedded and indexed chunks into ChromaDB.")
    else:
        print("No chunks to index.")
        
    return collection

if __name__ == "__main__":
    docs = load_documents()
    chunks = chunk_documents(docs)
    build_vector_store(chunks)
    print("Ingestion pipeline complete.")
