"""
CogniFuse — Gemini AI Service
Handles all interactions with Google Generative AI (Gemini 2.0 Flash)
"""

import os
import json
import re
from dotenv import load_dotenv

load_dotenv()

import google.generativeai as genai

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")


def _clean_json_response(text: str) -> str:
    """Strip markdown fences and extra whitespace from Gemini responses."""
    text = text.strip()
    # Remove ```json ... ``` or ``` ... ```
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    return text.strip()


def extract_triplets(text: str) -> list[dict]:
    """
    Extract concept relationships from text as triplets.
    Returns list of {subject, relation, object}.
    """
    prompt = (
        "Extract all concept relationships from this text as a JSON array of triplets.\n"
        "Each triplet: {\"subject\": \"...\", \"relation\": \"...\", \"object\": \"...\"}.\n"
        "Return ONLY the JSON array, no explanation, no markdown.\n"
        f"Text: {text}"
    )
    try:
        response = model.generate_content(prompt)
        raw = _clean_json_response(response.text)
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
    except Exception as e:
        error_msg = str(e)
        print(f"[Gemini] Error extracting triplets: {error_msg}")
        if "API_KEY_INVALID" in error_msg or "API key not valid" in error_msg:
            raise ValueError("API_KEY_INVALID")
        elif "429" in error_msg or "quota" in error_msg.lower():
            raise ValueError("QUOTA_EXCEEDED")
        return []


def generate_flashcards(ordered_concepts: list[str], graph_data: dict) -> list[dict]:
    """
    Generate flashcards for concepts in topological order.
    Returns list of {concept, front, back}.
    """
    # Build context from graph edges
    edges_context = ""
    if graph_data and "edges" in graph_data:
        for edge in graph_data["edges"]:
            edges_context += f"- {edge['source']} --[{edge['relation']}]--> {edge['target']}\n"

    prompt = (
        "You are a tutoring AI. Generate flashcards for each concept below.\n"
        "The concepts are in prerequisite order (learn first → learn last).\n\n"
        "Concepts (in order):\n"
        + "\n".join(f"{i+1}. {c}" for i, c in enumerate(ordered_concepts))
        + "\n\nRelationships:\n" + edges_context
        + "\n\nFor each concept, create a flashcard with:\n"
        "- 'concept': the concept name\n"
        "- 'front': a clear question or prompt about the concept\n"
        "- 'back': a concise, educational explanation (2-4 sentences)\n\n"
        "Return ONLY a JSON array of objects with keys: concept, front, back.\n"
        "No markdown, no explanation."
    )
    try:
        response = model.generate_content(prompt)
        raw = _clean_json_response(response.text)
        flashcards = json.loads(raw)
        return [
            {
                "concept": str(fc.get("concept", "")).strip(),
                "front": str(fc.get("front", "")).strip(),
                "back": str(fc.get("back", "")).strip()
            }
            for fc in flashcards
            if isinstance(fc, dict)
        ]
    except Exception as e:
        print(f"[Gemini] Error generating flashcards: {e}")
        # Fallback: generate basic cards
        return [
            {"concept": c, "front": f"What is {c}?", "back": f"A concept related to the study material."}
            for c in ordered_concepts
        ]


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
        response = model.generate_content(prompt)
        raw = _clean_json_response(response.text)
        quiz = json.loads(raw)
        return {
            "question": str(quiz.get("question", "")),
            "options": [str(o) for o in quiz.get("options", [])],
            "answer": str(quiz.get("answer", ""))
        }
    except Exception as e:
        print(f"[Gemini] Error generating quiz: {e}")
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
        response = model.generate_content(prompt)
        raw = _clean_json_response(response.text)
        card = json.loads(raw)
        return {
            "front": str(card.get("front", f"Explain {concept} in simple terms")),
            "back": str(card.get("back", f"A different way to understand {concept}."))
        }
    except Exception as e:
        print(f"[Gemini] Error mutating card: {e}")
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
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"[Gemini] Error generating summary: {e}")
        return f"{concept} is a key concept in this topic, connected to {neighbor_text}."
