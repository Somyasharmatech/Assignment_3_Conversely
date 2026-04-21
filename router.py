import os
import json
import google.generativeai as genai

def classify_query(query: str):
    """
    Classifies a user query into one of three categories: Factual, Synthesis, or Out of scope.
    This explicit classification routing happens BEFORE retrieval.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)
        
    prompt = f"""You are a query classification router for an AI Regulation Knowledge Base.
Your task is to analyze the user's query and classify it into exactly ONE of the following three categories.

Categories:
1. "Factual": The question asks for a specific fact, rule, or penalty. It can likely be found directly in a single location within AI Regulation documents. Examples: "What is the penalty for non-compliance in the EU?", "When does the AI act take effect?".
2. "Synthesis": The query requires combining, comparing, or summarizing information from multiple different angles, sources, or documents regarding AI Regulation. Examples: "Compare the EU and US approaches to AI regulation", "What are the common themes across the different regulations?".
3. "Out of scope": The query is NOT related to AI regulation, AI policy, AI laws, or the topic is entirely unrelated to the assumed knowledge base. Examples: "What is the capital of France?", "How do I write a fast API server?".

Analyze the following query:
'{query}'

Return your output strictly as a valid JSON object with no other text, fenced markdown, or formatting:
{{
  "category": "<Factual OR Synthesis OR Out of scope>",
  "reasoning": "<Short explanation of why>"
}}
"""

    try:
        if not os.environ.get("GEMINI_API_KEY"):
            raise ValueError("GEMINI_API_KEY is not set.")
            
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        text = response.text.replace("```json", "").replace("```", "").strip()
        result = json.loads(text)
        
        category = result.get("category", "Out of scope")
        if category not in ["Factual", "Synthesis", "Out of scope"]:
             category = "Out of scope"
             
        return {"category": category, "reasoning": result.get("reasoning", "")}
    except Exception as e:
        print(f"Routing Error: {e}")
        # Default or fallback logic
        return {"category": "Out of scope", "reasoning": str(e)}

if __name__ == "__main__":
    # Test cases if API key is set
    test_queries = [
        "What is the capital of France?",
        "What is the maximum fine under the EU AI Act?",
        "Compare the AI regulation principles between Europe and the United States."
    ]
    for q in test_queries:
        print(f"Query: {q}")
        print(f"Routing Decision: {classify_query(q)}")
        print("-" * 20)
