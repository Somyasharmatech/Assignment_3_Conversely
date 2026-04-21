import pandas as pd
from rouge_score import rouge_scorer
import sklearn
from generator import generate_answer, get_db_collection

# 15 Test Questions (5 Factual, 5 Synthesis, 5 Out of scope)
TEST_CASES = [
    # Factual
    {"query": "What is the maximum penalty for non-compliance under the EU AI Act?", "expected_category": "Factual", "keywords": ["fine", "penalty", "million", "percent", "compliance"]},
    {"query": "Which regulatory body oversees the enforcement of the AI laws?", "expected_category": "Factual", "keywords": ["body", "authority", "commission", "agency"]},
    {"query": "When does the new AI regulation take effect?", "expected_category": "Factual", "keywords": ["date", "effect", "year", "timeline"]},
    {"query": "How is a 'high-risk' AI system defined?", "expected_category": "Factual", "keywords": ["high-risk", "defined", "criteria", "health", "safety", "rights"]},
    {"query": "Are there any exemptions for open-source AI models?", "expected_category": "Factual", "keywords": ["open-source", "exempt", "free", "license"]},
    
    # Synthesis
    {"query": "Compare the US and EU definitions of high-risk AI models.", "expected_category": "Synthesis", "keywords": ["US", "EU", "european", "states", "compare", "differ", "contrast"]},
    {"query": "Summarize the major differences in penalties between the different jurisdictions mentioned.", "expected_category": "Synthesis", "keywords": ["penalty", "fine", "differences", "jurisdictions", "summary", "compare"]},
    {"query": "How do the provided documents contrast on the topic of AI model transparency?", "expected_category": "Synthesis", "keywords": ["transparency", "contrast", "differ", "openness"]},
    {"query": "What are the common themes regarding data privacy across all regulations?", "expected_category": "Synthesis", "keywords": ["privacy", "common", "data", "theme", "across"]},
    {"query": "Synthesize the requirements for general-purpose AI systems based on the texts.", "expected_category": "Synthesis", "keywords": ["general-purpose", "GPAI", "synthesize", "requirements"]},
    
    # Out of scope
    {"query": "How do I install PyTorch for deep learning?", "expected_category": "Out of scope", "keywords": ["cannot answer", "not available", "off-topic"]},
    {"query": "What is the recipe for a chocolate cake?", "expected_category": "Out of scope", "keywords": ["cannot answer", "not available", "off-topic"]},
    {"query": "Explain the plot of the movie The Matrix.", "expected_category": "Out of scope", "keywords": ["cannot answer", "not available", "off-topic"]},
    {"query": "Which stock is the best to buy right now?", "expected_category": "Out of scope", "keywords": ["cannot answer", "not available", "off-topic"]},
    {"query": "What are the best beaches to visit in Hawaii?", "expected_category": "Out of scope", "keywords": ["cannot answer", "not available", "off-topic"]},
]

def calculate_rouge_overlap(generated_text, expected_keywords):
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    target_text = " ".join(expected_keywords)
    scores = scorer.score(target_text, generated_text)
    return scores['rouge1'].fmeasure

def evaluate_system():
    print("Starting Evaluation of Agentic RAG System...")
    collection = get_db_collection()
    if not collection:
        print("Vector database is empty or not found. Cannot perform evaluation.")
        return
        
    results = []
    
    for i, test in enumerate(TEST_CASES):
        print(f"[{i+1}/15] Evaluating Query: {test['query']}")
        
        response = generate_answer(test["query"], collection)
        
        # 1. Routing Accuracy
        pred_category = response["category"]
        routing_correct = (pred_category == test["expected_category"])
        
        # 2. Retrieval Accuracy 
        # Metric: If Factual/Synthesis, we expect chunks. If Out of scope, we expect 0 chunks.
        chunks = response["chunks_used"]
        if test["expected_category"] == "Out of scope":
            retrieval_correct = len(chunks) == 0
        else:
            # We consider retrieval successful if it grabbed at least 1 document for in-scope queries
            retrieval_correct = len(chunks) > 0
            
        # 3. Answer Quality
        # Using keyword overlap via F-measure ROUGE-1
        answer_text = response["answer"]
        rouge_score = calculate_rouge_overlap(answer_text, test["keywords"])
        
        results.append({
            "Query": test["query"],
            "Expected Category": test["expected_category"],
            "Predicted Category": pred_category,
            "Routing Correct?": routing_correct,
            "Chunks Retrieved": len(chunks),
            "Retrieval Correct?": retrieval_correct,
            "ROUGE-1 F-score (Keyword Overlap)": round(rouge_score, 4)
        })

    df = pd.DataFrame(results)
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(df.to_string(index=False))
    
    df.to_csv("evaluation_results.csv", index=False)
    print("\nResults saved to evaluation_results.csv")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY METRICS")
    print("="*80)
    print(f"Total Routing Accuracy: {df['Routing Correct?'].mean() * 100:.2f}%")
    print(f"Total Retrieval Accuracy: {df['Retrieval Correct?'].mean() * 100:.2f}%")
    print(f"Average Keyword ROUGE Score: {df['ROUGE-1 F-score (Keyword Overlap)'].mean():.4f}")

if __name__ == "__main__":
    evaluate_system()
