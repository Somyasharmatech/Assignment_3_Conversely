import os
import chromadb
from dotenv import load_dotenv

load_dotenv()
from chromadb.utils import embedding_functions
from groq import Groq
from router import classify_query

def get_db_collection(db_path="chroma_db", collection_name="ai_regulation"):
    client = chromadb.PersistentClient(path=db_path)
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    try:
        return client.get_collection(name=collection_name, embedding_function=sentence_transformer_ef)
    except Exception as e:
        print(f"Warning: Could not get collection. E={e}")
        return None

def retrieve_chunks(collection, query, k):
    results = collection.query(
        query_texts=[query],
        n_results=k
    )
    if not results["documents"] or len(results["documents"][0]) == 0:
        return []
    
    # Return as list of text chunks with their source
    chunks = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append(f"Source Document: {meta['source']}\nContent: {doc}")
    return chunks

def generate_answer(query: str, collection):
    # Step 1: Explicitly inspectable routing
    route_decision = classify_query(query)
    category = route_decision["category"]
    reasoning = route_decision["reasoning"]
    
    # Log the routing decision precisely
    print(f"[ROUTER START] Query: '{query}'")
    print(f"[ROUTER DECISION] Category: {category}")
    print(f"[ROUTER REASONING] {reasoning}")
    
    # Step 2: Execution Path Delegation
    if category == "Out of scope":
        return {
            "answer": "I cannot answer this query because the provided documents do not contain enough information or the query is off-topic.",
            "category": category,
            "chunks_used": []
        }
        
    elif category == "Factual":
        # Narrow retrieval for localized facts
        k = 3
        chunks = retrieve_chunks(collection, query, k=k)
    elif category == "Synthesis":
        # Broad retrieval spanning multiple contexts/files
        k = 8
        chunks = retrieve_chunks(collection, query, k=k)
    else:
        return {"answer": "Unhandled routing category.", "category": "Error", "chunks_used": []}
        
    # Check if retrieval yielded info
    if not chunks:
        return {
            "answer": "The information is not available in the documents.",
            "category": category,
            "chunks_used": []
        }
        
    # Step 3: Generate the structured answer
    context_text = "\n\n---\n\n".join(chunks)
    
    prompt = f"""You are a precise AI Regulation Assistant. Answer the user's query strictly based on the context provided below.

User Query: {query}

Instructions & Constraints:
- This is a {category} query. Make sure your answer structure matches this intent (e.g., direct fact vs comprehensive comparison).
- Rely ONLY on the provided context. If the context does not explicitly contain the answer, do NOT hallucinate; explicitly state that the information is not present.
- If the sources provide contradictory information, acknowledge the contradiction and state what each source claims.
- Cite the source names when providing information ("According to [Source Document]...").
- Keep it concise, professional, and clear.

Context: 
{context_text}
"""
    
    try:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
             return {"answer": "Error: GROQ_API_KEY is not set.", "category": category, "chunks_used": chunks}
             
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1024
        )
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        answer = f"Error generating text: {e}"
        
    return {
        "answer": answer,
        "category": category,
        "chunks_used": chunks
    }

if __name__ == "__main__":
    collection = get_db_collection()
    if collection:
        q = "Compare the EU AI act penalties with US guidelines."
        res = generate_answer(q, collection)
        print(f"\n[GENERATED ANSWER]\n{res['answer']}")
    else:
        print("Knowledge base not ready. Run ingest.py first.")
