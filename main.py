import os
import argparse
from ingest import load_documents, chunk_documents, build_vector_store
from generator import get_db_collection, generate_answer
from evaluate import evaluate_system

def interactive_chat():
    collection = get_db_collection()
    if not collection:
        print("Knowledge base not found. Please run ingest first.")
        return
        
    print("\n" + "="*50)
    print("Welcome to the Agentic RAG System - AI Regulation")
    print("Type 'exit' or 'quit' to stop.")
    print("="*50)
    
    while True:
        query = input("\nQuery> ")
        if query.lower() in ['exit', 'quit']:
            break
            
        print("\nThinking...")
        response = generate_answer(query, collection)
        
        print(f"\n[Category: {response['category']}]")
        print("-" * 50)
        print(response['answer'])
        print("-" * 50)
        if response['category'] != "Out of scope":
            print(f"[{len(response['chunks_used'])} chunks retrieved for context]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agentic RAG System")
    parser.add_argument("--ingest", action="store_true", help="Run the ingestion pipeline")
    parser.add_argument("--evaluate", action="store_true", help="Run the evaluation framework")
    parser.add_argument("--chat", action="store_true", help="Run the interactive chat")
    
    args = parser.parse_args()
    
    if args.ingest:
        docs = load_documents()
        chunks = chunk_documents(docs)
        build_vector_store(chunks)
    elif args.evaluate:
        evaluate_system()
    elif args.chat:
        interactive_chat()
    else:
        parser.print_help()
