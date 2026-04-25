"""
CogniFuse — LLM Service (OpenRouter)
Handles interactions with OpenRouter API using the primary model and a fallback model.
"""

import os
import json
import re
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Configure OpenRouter client using the OpenAI SDK
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# Configuration — use reliable models
# Primary: google/gemini-2.0-flash-lite-001 (Fast, reliable, good free limits)
# Fallback: openai/gpt-4o-mini
PRIMARY_MODEL = "google/gemini-2.0-flash-lite-001"
FALLBACK_MODEL = "openai/gpt-4o-mini"

def generate_content_with_fallback(prompt: str) -> str:
    """
    Attempts to generate content using the PRIMARY_MODEL.
    If it fails, attempts the FALLBACK_MODEL.
    Returns the response text.
    """
    try:
        response = client.chat.completions.create(
            model=PRIMARY_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
    except Exception as e:
        error_msg = str(e)
        print(f"[OpenRouter] Primary model ({PRIMARY_MODEL}) failed: {error_msg}. Trying fallback...")
        
        # Propagate auth errors immediately if the API key is completely wrong
        if "401" in error_msg:
             raise ValueError("API_KEY_INVALID")

        try:
            # Attempt fallback
            fallback_response = client.chat.completions.create(
                model=FALLBACK_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            return fallback_response.choices[0].message.content
        except Exception as fallback_e:
            fallback_err_msg = str(fallback_e)
            print(f"[OpenRouter] Fallback model ({FALLBACK_MODEL}) also failed: {fallback_err_msg}")
            
            if "401" in fallback_err_msg:
                raise ValueError("API_KEY_INVALID")
            elif "429" in fallback_err_msg:
                raise ValueError("QUOTA_EXCEEDED")
            raise ValueError(f"API_ERROR: {fallback_err_msg}")


def _clean_json_response(text: str) -> str:
    """Strip markdown fences and extra whitespace from responses."""
    if not text:
        return ""
    text = text.strip()
    # Remove ```json ... ``` or ``` ... ```
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    return text.strip()


import spacy

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("[NLP] spaCy model 'en_core_web_sm' not found. Ignoring NER enrichment.")
    nlp = None

def extract_triplets(text: str) -> list[dict]:
    """
    Extract concept relationships from text as triplets.
    Returns list of {subject, relation, object}.
    """
    context_str = ""
    if nlp is not None:
        doc = nlp(text)
        # Filter entities based on generic labels that usually map to topics
        entities = [ent.text for ent in doc.ents if len(ent.text) > 2]
        if entities:
            # Provide NER context to hint the LLM 
            context_str = f"Hint: Key entities identified in the text are: {', '.join(set(entities))}.\n"

    prompt = (
        "You are an expert Educational Graph Architect. Extract all specific concept relationships "
        "explicitly mentioned in this text.\n\n"
        "FORMAT:\n"
        "- Return ONLY a JSON array of triplets: {\"subject\": \"...\", \"relation\": \"...\", \"object\": \"...\"}.\n"
        "- Focus on specific facts, definitions, and rules in the text.\n"
        f"{context_str}\n"
        f"Text to process: {text}"
    )
    try:
        response_text = generate_content_with_fallback(prompt)
        raw = _clean_json_response(response_text)
        triplets = json.loads(raw)
        # Validate structure
        validated = []
        for t in triplets:
            if isinstance(t, dict) and "subject" in t and "relation" in t and "object" in t:
                validated.append({
                    "subject": str(t["subject"]).strip(),
                    "relation": str(t["relation"]).strip(),
                    "object": str(t["object"]).strip()
                })
        return validated
    except ValueError as ve:
        # Re-raise known API flow control errors (401, 429) so main.py handles them
        if str(ve) in ["API_KEY_INVALID", "QUOTA_EXCEEDED"]:
            raise ve
        print(f"[OpenRouter] ValueError extract_triplets: {ve}")
        return []
    except Exception as e:
        print(f"[OpenRouter] Error parsing triplets JSON: {e}")
        return []


def get_foundational_prerequisites(topic: str) -> list[dict]:
    """
    Identifies all critical foundational concepts a student must master to learn a topic easily.
    Uses a dedicated pedagogical prompt.
    """
    prompt = (
        f"You are a pedagogical expert. A student needs to master '{topic}'.\n"
        f"Identify ALL critical foundational concepts and prerequisites they must understand to learn {topic} easily.\n\n"
        "REQUIREMENTS:\n"
        "1. Identify concepts from the 'roots' of the subject (e.g., if the topic is Polynomials, the roots are Arithmetic and Basic Algebra).\n"
        "2. Create relationships that show the 'Learning Path' (e.g., 'Arithmetic is a prerequisite for Algebra').\n"
        "3. Ensure at least one triplet directly connects a foundational root to the main topic.\n\n"
        "Return ONLY a JSON array of triplets: [{\"subject\": \"...\", \"relation\": \"...\", \"object\": \"...\"}].\n"
        "No explanation, no markdown."
    )
    try:
        response_text = generate_content_with_fallback(prompt)
        raw = _clean_json_response(response_text)
        triplets = json.loads(raw)
        return [
            {
                "subject": str(t["subject"]).strip(),
                "relation": str(t["relation"]).strip(),
                "object": str(t["object"]).strip()
            }
            for t in triplets
            if isinstance(t, dict) and "subject" in t and "relation" in t and "object" in t
        ]
    except Exception as e:
        print(f"[OpenRouter] Error in get_foundational_prerequisites: {e}")
        return []


def analyze_mistake(question: str, user_answer: str, concept: str) -> dict:
    """
    Analyzes a student's mistake in a quiz. 
    Determines if it is due to a lack of foundational clarity.
    Returns {analysis: str, suggestion: str, foundational_concept: str | None}.
    """
    prompt = (
        f"A student is learning '{concept}'. They failed this quiz question:\n"
        f"Question: {question}\n"
        f"Student's Wrong Answer: {user_answer}\n\n"
        "Analyze this mistake. Determine if it is a simple error or a lack of foundational clarity.\n"
        "If they lack foundational clarity, identify the EXACT prerequisite concept they should study instead.\n\n"
        "Return ONLY a JSON object with:\n"
        "- 'analysis': A short (1-2 sentence) explanation of the mistake.\n"
        "- 'suggestion': Advice on what to do next.\n"
        "- 'foundational_concept': The name of the foundational concept to learn first (if applicable, otherwise null).\n"
        "No explanation, no markdown."
    )
    try:
        response_text = generate_content_with_fallback(prompt)
        raw = _clean_json_response(response_text)
        return json.loads(raw)
    except Exception as e:
        print(f"[OpenRouter] Error in analyze_mistake: {e}")
        return {
            "analysis": "You missed this question. Let's review the concept again.",
            "suggestion": "Review the concept details before trying again.",
            "foundational_concept": None
        }


def generate_flashcards(ordered_concepts: list[str], graph_data: dict) -> list[dict]:
    """
    Generate flashcards for concepts in topological order using batch processing (chunks of 8).
    This prevents JSON truncation issues on large graphs.
    """
    edges_context = ""
    if graph_data and "edges" in graph_data:
        for edge in graph_data["edges"]:
            edges_context += f"- {edge['source']} --[{edge['relation']}]--> {edge['target']}\n"

    all_flashcards = []
    chunk_size = 8
    
    for i in range(0, len(ordered_concepts), chunk_size):
        chunk = ordered_concepts[i:i + chunk_size]
        print(f"[OpenRouter] Generating batch {i//chunk_size + 1} ({len(chunk)} concepts)...")
        
        prompt = (
            "You are a tutoring AI. Generate flashcards for the following specific concepts.\n"
            "Use the provided relationship context to make the flashcards educational.\n\n"
            "TARGET CONCEPTS:\n"
            + "\n".join(f"- {c}" for c in chunk)
            + "\n\nRELATIONSHIP CONTEXT:\n" + edges_context
            + "\n\nFor each TARGET CONCEPT, return a JSON object with:\n"
            "- 'concept': the concept name\n"
            "- 'front': a clear question or prompt about the concept\n"
            "- 'back': a concise, educational explanation (2-4 sentences)\n\n"
            "Return ONLY a JSON array of objects with keys: concept, front, back.\n"
            "No markdown, no explanation."
        )
        
        try:
            response_text = generate_content_with_fallback(prompt)
            raw = _clean_json_response(response_text)
            batch_cards = json.loads(raw)
            
            if isinstance(batch_cards, list):
                for fc in batch_cards:
                    if isinstance(fc, dict):
                        all_flashcards.append({
                            "concept": str(fc.get("concept", "")).strip(),
                            "front": str(fc.get("front", "")).strip(),
                            "back": str(fc.get("back", "")).strip()
                        })
        except Exception as e:
            print(f"[OpenRouter] Error in batch {i//chunk_size + 1}: {e}")
            # Fallback for this specific chunk
            for c in chunk:
                all_flashcards.append({
                    "concept": c,
                    "front": f"What is {c}?",
                    "back": f"A key concept identified in your study material."
                })
                
    return all_flashcards


def generate_quiz(concept: str, context: str = "") -> dict:
    """
    Generate a multiple-choice quiz question for a concept.
    Returns {question, options: [A,B,C,D], answer}.
    """
    prompt = (
        f"Generate a multiple-choice quiz question about the concept: '{concept}'.\n"
        + (f"Context: {context}\n" if context else "")
        + "\nReturn ONLY a JSON object with:\n"
        "- 'question': the question text\n"
        "- 'options': array of exactly 4 answer choices (strings)\n"
        "- 'answer': the correct answer (must be one of the options exactly)\n\n"
        "No markdown, no explanation. Just the JSON object."
    )
    try:
        response_text = generate_content_with_fallback(prompt)
        raw = _clean_json_response(response_text)
        quiz = json.loads(raw)
        return {
            "question": str(quiz.get("question", "")),
            "options": [str(o) for o in quiz.get("options", [])],
            "answer": str(quiz.get("answer", ""))
        }
    except Exception as e:
        print(f"[OpenRouter] Error generating quiz: {e}")
        return {
            "question": f"What is {concept}?",
            "options": [
                f"{concept} is a fundamental concept",
                f"{concept} is not important",
                f"{concept} is unrelated to the topic",
                f"{concept} cannot be defined"
            ],
            "answer": f"{concept} is a fundamental concept"
        }


def mutate_card(concept: str, previous_explanation: str) -> dict:
    """
    Generate a new flashcard explanation from a different angle.
    Returns {front, back} with a fresh perspective.
    """
    prompt = (
        f"A student is struggling with the concept '{concept}'.\n"
        f"The previous explanation was:\n\"{previous_explanation}\"\n\n"
        "Generate a NEW flashcard with a completely different angle/explanation.\n"
        "Use analogies, real-world examples, or a simpler breakdown.\n\n"
        "Return ONLY a JSON object with:\n"
        "- 'front': a new question or prompt about the concept\n"
        "- 'back': a new explanation from a different perspective (2-4 sentences)\n\n"
        "No markdown, no explanation."
    )
    try:
        response_text = generate_content_with_fallback(prompt)
        raw = _clean_json_response(response_text)
        card = json.loads(raw)
        return {
            "front": str(card.get("front", f"Explain {concept} in simple terms")),
            "back": str(card.get("back", f"A different way to understand {concept}."))
        }
    except Exception as e:
        print(f"[OpenRouter] Error mutating card: {e}")
        return {
            "front": f"Can you explain {concept} using an analogy?",
            "back": f"Think of {concept} as a building block - it connects to other ideas in the topic."
        }


def generate_summary(concept: str, neighbors: list[str]) -> str:
    """
    Generate a 2-3 sentence summary for a concept and its neighborhood.
    """
    neighbor_text = ", ".join(neighbors) if neighbors else "no directly connected concepts"
    prompt = (
        f"Write a 2-3 sentence educational summary of the concept '{concept}'.\n"
        f"Related concepts: {neighbor_text}.\n"
        "Be concise and informative. Return ONLY the summary text, no JSON, no markdown."
    )
    try:
        response_text = generate_content_with_fallback(prompt)
        return response_text.strip()
    except Exception as e:
        print(f"[OpenRouter] Error generating summary: {e}")
        return f"{concept} is a key concept in this topic, connected to {neighbor_text}."
