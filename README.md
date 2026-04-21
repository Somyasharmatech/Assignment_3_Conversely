# Agentic RAG system for AI Regulation

This repository contains an agentic RAG system designed to read, classify, and answer queries regarding a supplied dataset of AI Regulation documents. The system explicitly reasons about *how* to answer a user's query before performing retrieval and generating a grounded response.

## 🏗️ System Architecture

1. **Ingestion Pipeline (`ingest.py`)**: Loads PDF documents, chunks them deterministically, and embeds them into a local vector store (ChromaDB) using HuggingFace sentence-transformers.
2. **Explicit Query Router (`router.py`)**: intercepts the user query and uses a zero-shot LLM (Gemini 1.5) bounded by strict structural constraints to classify the query into exactly one of three logic paths: **Factual**, **Synthesis**, or **Out of scope**.
3. **Generation Wrapper (`generator.py`)**: Takes the routing decision and adjusts the retrieval strategy accordingly (e.g., retrieving `k=3` localized chunks for factual queries, `k=8` spanning chunks for synthesis, or immediately cancelling retrieval for out-of-scope queries). It then prompts the LLM to provide a final grounded answer, explicitly referencing contradictions if found.
4. **Evaluation Harness (`evaluate.py`)**: A programmatic evaluation loop testing the system against 15 predefined queries (5 per category).

## 📊 Design & Logic Choices

### Chunking and Embedding
- **Strategy**: Files are split via `RecursiveCharacterTextSplitter` into chunks of **1200 characters** with **300 characters overlap**.
- **Justification**: Legal document structures flow in long paragraphs. A 1200 character window encompasses full legal definitions, while the 300 overlap ensures continuity across arbitrary splits. 
- **Embeddings**: We use `all-MiniLM-L6-v2` locally via HuggingFace's `sentence-transformers`. This operates at zero cost locally and offers exceptionally strong performance for semantic similarity search over text blocks.

### Inspectable Routing Logic
- The assignment rigidly forbids "black box" LangChain agents that self-determine execution paths invisibly. 
- **Solution**: We pass the query through a constrained LLM wrapper that *must* output `{"category": "...", "reasoning": "..."}` as JSON. 
- The system then uses standard, inspectable Python `if/elif/else` statements executing entirely different logic blocks. This proves the routing is fully explicit and explainable.

## 📈 Evaluation Methodology
The script runs 15 diverse queries through the pipeline and gathers automated metrics:
- **Routing Accuracy**: Binary match between predicted route vs expected ground truth.
- **Retrieval Accuracy**: Logic test ensuring we do NOT retrieve documents if a query is Off-Topic, but DO retrieve valid content for Factual/Synthesis ones.
- **Answer Quality**: We calculate `ROUGE-1` F-scores to measure keyword overlap between generated and manually seeded factual answers.

(Check `FAILURES.md` for a deeper dive into the system's edge cases).

## ⚙️ How to Run

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Prepare Data:**
   - Put your 4 AI Regulation `.pdf` files into the `data/` folder.
3. **Set API Keys:**
   ```bash
   export GEMINI_API_KEY="your_api_key_here"
   ```
4. **Run the CLI:**
   - Step 1: Ingest Documents into DB
     ```bash
     python main.py --ingest
     ```
   - Step 2: Test the Chatbot Interactive Mode
     ```bash
     python main.py --chat
     ```
   - Step 3: Run Automation & Evaluation Matrix
     ```bash
     python main.py --evaluate
     ```
