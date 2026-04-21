# Failure Analysis

Building an agentic RAG system that deterministically categorizes and answers queries highlights several structural limitations of "vanilla" RAG wrapped with a static router. Below are three key areas where this system fails or underperforms, along with root cause analysis and future improvements.

## 1. Zero-shot Routing on Ambiguous Context
**Failure Case**: If a user asks *"What are the rules?"* or *"Tell me the summary"*, the explicit Router might classify this as **Out of scope** because it strictly looks for AI regulation terminology, or it might struggle to discern if it's *Factual* vs *Synthesis*.
**Root Cause**: The LLM zero-shot router evaluates the query entirely in isolation without conversational memory. It assumes user queries are hyper-specific.
**What I would do differently**: I would implement short-term conversational history. By injecting the last 3 turns of conversation into the routing prompt, the LLM could infer that *"What are the rules?"* implies *"What are the rules [regarding the EU AI Act discussed previously]?"*, significantly improving routing robustness.

## 2. Naive Contradiction Handling in Semantic Retrieval
**Failure Case**: The documents intentionally contain contradictory information (e.g., varying penalty amounts or differing definitions of "high-risk" across legal jurisdictions). While our prompt *asks* the LLM to highlight contradictions, standard semantic search (k=3 or k=8) might retrieve three chunks from *only one* author/document if they are lexically similar to the query, entirely missing the contradictory document.
**Root Cause**: Dense embeddings (`all-MiniLM-L6-v2`) prioritize semantic similarity of the text to the query, ignoring document diversity.
**What I would do differently**: Implement **Maximal Marginal Relevance (MMR)** or **HyDE** (Hypothetical Document Embeddings) to diversify retrieval. Fetching chunks from distinctly different sources ensures the generation phase actually sees the contradictory evidence so it can synthesize it properly.

## 3. Strict Lexical Evaluation (ROUGE-1) Mismatches
**Failure Case**: An answer is functionally perfect (e.g., generating *"Monetary penalty"*), but the automated evaluation framework scores it `0.0` for **Answer Quality** because the expected keyword list was `["fine", "fee"]`.
**Root Cause**: Using `rouge-score` metrics locally represents a rigid lexical matching process. Standard quantitative metrics fall short of capturing semantic meaning and truthfulness.
**What I would do differently**: Use an **LLM-as-a-judge** workflow (e.g., calling Gemini/GPT to grade the answer on a scale from 1-100 given a detailed rubric) or using specialized embeddings (e.g., calculating cosine similarity between the expected ideal answer embedding and the generated answer embedding).
