"""
CogniFuse — FastAPI Backend
Main application with all API endpoints.
"""

import os
# Fix for OMP: Error #15 (Multiple OpenMP runtimes)
# Must be set before importing libraries like torch, faiss, or numpy
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import json
from datetime import datetime, date
import time

load_dotenv()

from llm_service import (
    extract_triplets,
    get_foundational_prerequisites,
    analyze_mistake,
    generate_flashcards,
    generate_quiz,
    mutate_card,
    generate_summary,
)
from graph import KnowledgeGraph
from ocr_service import extract_text_from_image, extract_text_from_pdf
from supabase_client import get_supabase
import jwt  # PyJWT

supabase = get_supabase()

# ── App Setup ──────────────────────────────────────────────
app = FastAPI(title="CogniFuse API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi import Depends, Header

# ── Auth Dependency ──────────────────────────────────────
async def get_current_user(authorization: str = Header(None)):
    """Verifies the Supabase JWT and returns the user object."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header.")
    
    token = authorization.split(" ")[1]
    try:
        # Verify token with Supabase
        user = supabase.auth.get_user(token)
        if not user or not user.user:
            raise HTTPException(status_code=401, detail="Invalid session.")
        return user.user
    except Exception as e:
        print(f"[Auth Error] {e}")
        raise HTTPException(status_code=401, detail="Authentication failed.")

# ── Data Scoping Helpers ──────────────────────────────────
async def get_user_graph(user_id: str):
    """Loads a user's knowledge graph from Supabase."""
    res = supabase.table("knowledge_graphs").select("graph_json").eq("user_id", user_id).execute()
    kg = KnowledgeGraph()
    if res.data and len(res.data) > 0:
        kg.load_json(res.data[0]["graph_json"])
    return kg

async def save_user_graph(user_id: str, kg: KnowledgeGraph):
    """Saves a user's knowledge graph to Supabase."""
    graph_data = kg.get_graph_data()
    supabase.table("knowledge_graphs").upsert({
        "user_id": user_id,
        "graph_json": graph_data,
        "updated_at": "now()"
    }).execute()

async def get_user_history(user_id: str):
    """Loads user stats and SM-2 progress."""
    res = supabase.table("profiles").select("*").eq("id", user_id).execute()
    profile = res.data[0] if res.data else {"streak_count": 0, "last_active_date": None}
    
    sm2_res = supabase.table("concept_mastery").select("*").eq("user_id", user_id).execute()
    sm2_data = {item["concept"]: item for item in sm2_res.data} if sm2_res.data else {}
    
    return {
        "streak": {"count": profile.get("streak_count", 0), "last_active": profile.get("last_active_date")},
        "sm2": sm2_data
    }

# ── Request/Response Models ───────────────────────────────
class TextInput(BaseModel):
    text: str


class QuizRequest(BaseModel):
    concept: str


class AnswerSubmission(BaseModel):
    concept: str
    correct: bool
    score: float


class SummaryRequest(BaseModel):
    concept: str


class MistakeAnalysisRequest(BaseModel):
    question: str
    user_answer: str
    concept: str



# ── Helper: Run Full Pipeline ─────────────────────────────
async def run_pipeline(text: str, user: any) -> dict:
    """
    1. Extract triplets (Textual & Foundational)
    2. Update Knowledge Graph
    3. Topological sort
    4. Generate flashcards in order
    5. Save everything to Supabase
    """
    user_id = user.id

    # Step 1: Extract Textual Triplets
    try:
        current_triplets = extract_triplets(text)
    except ValueError as e:
        if str(e) == "API_KEY_INVALID":
            raise HTTPException(status_code=401, detail="Invalid API Key.")
        elif str(e) == "QUOTA_EXCEEDED":
            raise HTTPException(status_code=429, detail="Quota exceeded.")
        raise e

    if not current_triplets:
        raise HTTPException(status_code=400, detail="No concepts found.")

    # Step 2: Extract Foundational Triplets
    main_topic = current_triplets[0]["subject"] if current_triplets else "the topic"
    base_triplets = get_foundational_prerequisites(main_topic)

    # Step 3: Load current user graph and update
    knowledge_graph = await get_user_graph(user_id)
    knowledge_graph.build_from_triplets(base_triplets, concept_type="base")
    knowledge_graph.build_from_triplets(current_triplets, concept_type="current")
    await save_user_graph(user_id, knowledge_graph)

    # Step 4: Topological sort
    ordered_concepts = knowledge_graph.topological_order()

    # Step 5: Generate flashcards
    graph_data = knowledge_graph.get_graph_data()
    flashcards = generate_flashcards(ordered_concepts, graph_data)
    
    # Save flashcards to DB
    for fc in flashcards:
        supabase.table("flashcards").upsert({
            "user_id": user_id,
            "concept": fc["concept"],
            "front": fc["front"],
            "back": fc["back"]
        }).execute()

    # Log action
    supabase.table("activity_logs").insert({
        "user_id": user_id,
        "action": "processed_content",
        "details": {"main_topic": main_topic, "node_count": len(graph_data["nodes"])}
    }).execute()

    return {
        "triplets": current_triplets + base_triplets,
        "graph": graph_data,
        "flashcards": flashcards,
        "ordered_concepts": ordered_concepts,
    }



# ── Endpoints ─────────────────────────────────────────────

@app.post("/process-text")
async def process_text(input_data: TextInput, user: any = Depends(get_current_user)):
    """Process raw text input through the full pipeline."""
    if not input_data.text.strip():
        raise HTTPException(status_code=400, detail="Text input cannot be empty.")
    return await run_pipeline(input_data.text, user)


@app.post("/process-file")
async def process_file(file: UploadFile = File(...), user: any = Depends(get_current_user)):
    """Process uploaded file (PDF or image) through the full pipeline."""
    contents = await file.read()
    filename = file.filename.lower() if file.filename else ""

    # Detect file type and extract text
    if filename.endswith(".pdf"):
        text = extract_text_from_pdf(contents)
    elif filename.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")):
        text = extract_text_from_image(contents)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    if not text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text.")

    return await run_pipeline(text, user)


@app.post("/generate-quiz")
async def api_generate_quiz(request: QuizRequest, user: any = Depends(get_current_user)):
    """Generate an MCQ quiz for a specific concept."""
    knowledge_graph = await get_user_graph(user.id)
    neighbors = knowledge_graph.get_neighbors(request.concept)
    context = f"Related concepts: {', '.join(neighbors)}" if neighbors else ""
    quiz = generate_quiz(request.concept, context)
    return quiz


@app.post("/submit-answer")
async def submit_answer(submission: AnswerSubmission, user: any = Depends(get_current_user)):
    """
    Submit a quiz answer with user-specific persistence.
    """
    user_id = user.id
    history = await get_user_history(user_id)
    knowledge_graph = await get_user_graph(user_id)
    
    concept = submission.concept
    quality = 5 if submission.correct else 1
    
    # 1. Update Streak
    today = str(date.today())
    last_active = history["streak"].get("last_active")
    streak_count = history["streak"].get("count", 0)
    
    if last_active != today:
        if last_active:
            last_date = datetime.strptime(last_active, "%Y-%m-%d").date()
            if (date.today() - last_date).days == 1:
                streak_count += 1
            else:
                streak_count = 1
        else:
            streak_count = 1
        
        # Update Profiles (Streak)
        supabase.table("profiles").upsert({
            "id": user_id,
            "email": user.email,
            "streak_count": streak_count,
            "last_active_date": today
        }).execute()

    # 2. SM-2 Logic
    sm2_data = history["sm2"].get(concept, {"interval": 0, "repetitions": 0, "ease_factor": 2.5})
    # (Existing SM-2 math logic stays same...)
    if quality >= 3:
        if sm2_data["repetitions"] == 0: sm2_data["interval"] = 1
        elif sm2_data["repetitions"] == 1: sm2_data["interval"] = 6
        else: sm2_data["interval"] = int(sm2_data["interval"] * sm2_data["ease_factor"])
        sm2_data["repetitions"] += 1
    else:
        sm2_data["repetitions"] = 0
        sm2_data["interval"] = 1
    
    sm2_data["ease_factor"] = sm2_data["ease_factor"] + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
    if sm2_data["ease_factor"] < 1.3: sm2_data["ease_factor"] = 1.3
    sm2_data["next_review"] = time.time() + (sm2_data["interval"] * 86400)
    
    # 3. Save Mastery/SM-2 to DB
    mastery_score = knowledge_graph.get_mastery(concept)
    if submission.correct:
        mastery_score = min(100, mastery_score + submission.score)
    else:
        mastery_score = max(0, mastery_score - 10)
    
    knowledge_graph.update_mastery(concept, mastery_score)
    await save_user_graph(user_id, knowledge_graph)

    supabase.table("concept_mastery").upsert({
        "user_id": user_id,
        "concept": concept,
        "mastery_score": mastery_score,
        "sm2_interval": sm2_data["interval"],
        "sm2_repetitions": sm2_data["repetitions"],
        "sm2_ease_factor": sm2_data["ease_factor"],
        "next_review": sm2_data["next_review"]
    }).execute()

    # 4. Result Processing (Root Cause / Mutation)
    result = {"mastery_updated": True, "root_cause": None, "mutated_card": None}
    if not submission.correct:
        root_cause = knowledge_graph.reverse_bfs_root_cause(concept)
        result["root_cause"] = root_cause
        
        # Log failure
        supabase.table("activity_logs").insert({
            "user_id": user_id, 
            "action": "quiz_failure", 
            "details": {"concept": concept}
        }).execute()

    return result


@app.get("/graph")
async def api_get_graph(user: any = Depends(get_current_user)):
    """Return user-specific graph data."""
    kg = await get_user_graph(user.id)
    return kg.get_graph_data()


@app.get("/flashcards")
async def api_get_flashcards(user: any = Depends(get_current_user)):
    """Return user-specific flashcards."""
    res = supabase.table("flashcards").select("*").eq("user_id", user.id).execute()
    return res.data


@app.post("/summary")
async def api_get_summary(request: SummaryRequest, user: any = Depends(get_current_user)):
    """Generate a summary for a concept."""
    kg = await get_user_graph(user.id)
    neighbors = kg.get_neighbors(request.concept)
    summary = generate_summary(request.concept, neighbors)
    return {"concept": request.concept, "summary": summary}


@app.get("/recommend-next")
async def recommend_next(user: any = Depends(get_current_user), top_k: int = 5):
    """Get GNN-powered recommendations for next study topics."""
    kg = await get_user_graph(user.id)
    if not kg.gnn.is_trained():
        return {"active": False, "recommendations": []}
    return {
        "active": True,
        "recommendations": kg.get_recommended_concepts(top_k)
    }


@app.get("/semantic-search")
async def semantic_search(query: str, user: any = Depends(get_current_user), top_k: int = 5):
    """Search for concepts using meaning (FAISS embeddings)."""
    kg = await get_user_graph(user.id)
    return kg.semantic_search(query, top_k)


# ── Analytics Endpoints ───────────────────────────────────

@app.get("/analytics/streak")
async def api_get_streak(user: any = Depends(get_current_user)):
    """Get user study streak."""
    history = await get_user_history(user.id)
    return history["streak"]


@app.get("/analytics/weak-zones")
async def api_get_weak_zones(user: any = Depends(get_current_user)):
    """Get weak areas via GNN."""
    kg = await get_user_graph(user.id)
    return kg.get_weak_areas()


@app.get("/analytics/mastery")
async def api_get_mastery(user: any = Depends(get_current_user)):
    """Get overall mastery."""
    kg = await get_user_graph(user.id)
    nodes = kg.get_graph_data()["nodes"]
    if not nodes: return {"average": 0, "total_concepts": 0}
    avg = sum(n["mastery"] for n in nodes) / len(nodes)
    return {"average": round(avg, 1), "total_concepts": len(nodes)}



@app.post("/analyze-mistake")
async def api_analyze_mistake(request: MistakeAnalysisRequest, user: any = Depends(get_current_user)):
    """Analyze a quiz mistake using AI."""
    # We log the mistake analysis request
    supabase.table("activity_logs").insert({
        "user_id": user.id,
        "action": "mistake_analysis",
        "details": {"concept": request.concept}
    }).execute()
    return analyze_mistake(request.question, request.user_answer, request.concept)


@app.post("/reset")
async def reset_state(user: any = Depends(get_current_user)):
    """Clear all data for this user."""
    user_id = user.id
    supabase.table("knowledge_graphs").delete().eq("user_id", user_id).execute()
    supabase.table("flashcards").delete().eq("user_id", user_id).execute()
    supabase.table("concept_mastery").delete().eq("user_id", user_id).execute()
    supabase.table("profiles").update({"streak_count": 0, "last_active_date": None}).eq("id", user_id).execute()
    return {"message": "User data reset successful"}



# ── Run ───────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
